import glob, math, re, sys, os, os.path as osp
import uuid
from webbrowser import get
import ROOT
import numpy as np
import seutils
import itertools
from array import array
from contextlib import contextmanager
import xgboost as xgb
from typing import List

try:
    import click
except ImportError:
    print('First install click:\npip install click')
    raise

# Add this directory to the python path, so the imports below work
sys.path.append(osp.dirname(osp.abspath(__file__)))
import bdtcode

import bdtcode
import bdtcode.histogramming as H
import bdtcode.crosssections as crosssections
from bdtcode.utils import *


@click.group()
def cli():
    pass


def split_by_category(things, labels, pats=['QCD', 'TTJets', 'WJets', 'ZJets', 'mz']):
    """
    Takes a flat list of `things` (anything), and a list of labels.
    Returns a nested of list of things, split by category
    """
    def index_for_label(label):
        for i, pat in enumerate(pats):
            if pat in label:
                return i
        raise Exception(f'Could not categorize {label}')
    returnable = [[] for i in range(len(pats))]
    for thing, label in zip(things, labels):
        returnable[index_for_label(label)].append(thing)
    return returnable


class Sample:
    """
    Container for a sample.

    A sample can be a specific set of background events, e.g. 
    TTJets_HT-600to800, QCD_Pt_800to1000_TuneCP5, mz250, etc.

    The container stores a simply dictionary in self.d and
    has a few convenience methods to interact with 
    """

    def __init__(self, label, d):
        self.d = d
        self.label = label

    @property
    def mz(self):
        if not hasattr(self, '_mz'): 
            match = re.search(r'mz(\d+)', self.label)
            self._mz = int(match.group(1)) if match else None
        return self._mz
 
    @property
    def is_sig(self):
        return self.mz is not None

    @property
    def is_bkg(self):
        return self.mz is None

    @property
    def genjetpt_efficiency(self):
        if self.is_bkg: return 1.
        return bdtcode.crosssections.genjetpt_eff(self.mz)

    @property
    def crosssection(self):
        """
        Returns inclusive cross section based on the label
        """
        return bdtcode.crosssections.label_to_xs(self.label)

    def mt(self, min_score=None):
        """Returns mt, with the option of cutting on the score here"""
        return self.d['mt'] if min_score is None else self.d['mt'][self.score > min_score]

    @property
    def score(self):
        return self.d['score']

    def bdt_efficiency(self, min_score=None):
        return 1. if min_score is None else (self.score > min_score).sum() / len(self)

    @property
    def preselection_efficiency(self):
        return self.d['preselection']/self.d['total']

    def nevents_after_preselection(self, lumi=137.2*1e3):
        return self.crosssection * lumi * self.preselection_efficiency * self.genjetpt_efficiency

    def nevents_after_bdt(self, min_score=None, lumi=137.2*1e3):
        return self.nevents_after_preselection(lumi) * self.bdt_efficiency(min_score)

    def __len__(self):
        """Returns number of entries in the underlying dict"""
        return len(self.score)


def sample_to_mt_histogram(sample: Sample, min_score=None, mt_binning=None, name=None):
    try_import_ROOT()
    import ROOT
    mt = sample.mt(min_score)
    binning = array('f', crosssections.MT_BINNING if mt_binning is None else mt_binning)
    if name is None: name = str(uuid.uuid4())
    h = ROOT.TH1F(name, name, len(binning)-1, binning)
    ROOT.SetOwnership(h, False)
    [ h.Fill(x) for x in mt ]
    H.normalize(h, sample.nevents_after_bdt(min_score))
    return h


def get_samples_from_postbdt_directory(directory) -> List[List[Sample]]:
    print(f'Building samples from {directory}')
    # Gather all the available labels
    labels = list(set(osp.basename(s) for s in glob.iglob(osp.join(directory, '*/*'))))
    labels.sort()
    # Actually combine all npz files into one dict and create the overarching Sample class
    samples = [
        Sample(l, H.combine_npzs(glob.iglob(osp.join(directory, f'*/*{l}*/*.npz'))))
        for l in labels
        ]
    # Split the samples by category
    samples = split_by_category(samples, labels)
    return samples


@cli.command()
@click.argument('postbdtdir')
def print_statistics(postbdtdir):
    def clean_label(label):
        """Strips off some verbosity from the label for printing purposes"""
        for p in [
            'Autumn18.', '_TuneCP5', '_13TeV', '_pythia8', '-pythia8',
            '-madgraphMLM', '-madgraph', 'genjetpt375_', '_mdark10_rinv0.3',
            'ToLNu', 'ToNuNu',
            ]:
            label = label.replace(p, '')
        return label
    cutflow_keys = [
        'total',
        '>=2jets',
        'eta<2.4',
        'trigger',
        'ecf>0',
        'rtx>1.1',
        'nleptons==0',
        'metfilter',
        'preselection',
        ]
    header_column = ['label'] + cutflow_keys + ['xs', 'genpt>375', 'n137_presel']

    def print_cutflow(samples: List[Sample]):
        table = [header_column]
        for sample in samples:
            column = [clean_label(sample.label)] # , f'{xs:.1f}']
            column.extend(f"{100*sample.d[key]/sample.d['total']:.2f}%" for key in cutflow_keys)
            column.append(f'{sample.crosssection:.1f}')
            column.append(f'{100.*sample.genjetpt_efficiency:.3f}%')
            column.append(f'{sample.nevents_after_preselection():.0f}')
            table.append(column)
        print_table(table, transpose=True)

    samples = get_samples_from_postbdt_directory(postbdtdir)
    for sample_group in samples:
        print_cutflow(sample_group)
        print('')

    samples = flatten(*samples)
    b = sum(sample.nevents_after_preselection() for sample in samples if sample.is_bkg)
    for sample in samples:
        if sample.is_bkg: continue
        s = sample.nevents_after_preselection()
        print(f'mz={sample.mz} {s=:.0f}, {b=:.0f}, s/sqrt(b)={s/np.sqrt(b):.3f}, s/b={s/b:.3f}')


@cli.command()
@click.option('-o', '--rootfile', default='test.root')
@click.argument('postbdt-dir')
def make_histograms(rootfile, postbdt_dir):
    *bkgs, sigs = get_samples_from_postbdt_directory(postbdt_dir)

    # binning = MT_BINNING
    left = 210.
    #right = 500.
    right = 800.
    bin_width = 8.
    binning = [left+i*bin_width for i in range(math.ceil((right-left)/bin_width))]

    def get_group_name(label):
        for pat in ['qcd', 'ttjets', 'wjets', 'zjets']:
            if pat in label.lower():
                return pat
        raise Exception(f'No group name for {label}')

    with open_root(rootfile, 'RECREATE') as f:
        for min_score in [None, 0.1, 0.2, 0.3, 0.4]:
            tdir = f.mkdir(f'bsvj_{0 if min_score is None else min_score:.1f}'.replace('.','p'))
            tdir.cd()
            # Loop over the mz' mass points
            for sig in sigs:
                h = sample_to_mt_histogram(sig, min_score=min_score, mt_binning=binning, name=f'SVJ_mZprime{sig.mz:.0f}_mDark10_rinv03_alphapeak')
                print(f'Writing {h.GetName()} --> {rootfile}/{tdir.GetName()}')
                h.Write()

            hs_bkg = []
            for bkg_group in bkgs:
                h = H.sum_th1s(
                    get_group_name(bkg_group[0].label),
                    (sample_to_mt_histogram(s, min_score=min_score, mt_binning=binning) for s in bkg_group)
                    )
                print(f'Writing {h.GetName()} --> {rootfile}/{tdir.GetName()}')
                h.Write()
                hs_bkg.append(h)

            # Finally write the combined bkg histogram
            h = H.sum_th1s('Bkg', hs_bkg)
            print(f'Writing {h.GetName()} --> {rootfile}/{tdir.GetName()}')
            h.Write()


def combine_dirs_with_weights(directories, weights):
    ds = [H.combine_npzs(glob.glob(osp.join(directory, '*.npz'))) for directory in directories]
    return H.combine_ds_with_weights(ds, weights)


@cli.command()
def process_mz250_locally(model):
    """
    Calculates the BDT scores for mz250 locally and dumps it to a .npz
    """
    rootfiles = seutils.ls_wildcard(
        'root://cmseos.fnal.gov//store/user/lpcdarkqcd/MCSamples_Summer21/TreeMaker'
        '/genjetpt375_mz250_mdark10_rinv0.3/*.root'
        )
    H.dump_score_npzs_mp(model, rootfiles, 'mz250_mdark10_rinv0p3.npz')


@cli.command()
def process_qcd_locally(model):
    for qcd_dir in seutils.ls_wildcard(
        'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BKG/bkg_May04_year2018/*QCD_Pt*'
        ):
        print(f'Processing {qcd_dir}')
        outfile = osp.basename(qcd_dir + '.npz')
        rootfiles = seutils.ls_wildcard(osp.join(qcd_dir, '*.root'))
        H.dump_score_npzs_mp(model, rootfiles, outfile)


if __name__ == '__main__':
    cli()
