from asyncio.log import logger
import glob, math, re, sys, os, os.path as osp
import uuid
import numpy as np
import seutils
from array import array
from typing import List

try:
    import click
except ImportError:
    print('First install click:\npip install click')
    raise

# Add this directory to the python path, so the imports below work
sys.path.append(osp.dirname(osp.abspath(__file__)))
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

def get_samples_from_postbdt_directory(directory, debug=False) -> List[List[bdtcode.sample.Sample]]:
    print(f'Building samples from {directory}')
    # Gather all the available labels
    labels = list(set(osp.basename(s) for s in glob.iglob(osp.join(directory, '*/*'))))
    labels.sort()
    # Actually combine all npz files into one dict and create the overarching Sample class
    if debug:
        import pprint
        samples = []
        for l in labels:
            npz_files = glob.glob(osp.join(directory, f'*/*{l}*/*.npz'))[:5]
            s = bdtcode.sample.Sample(l, H.combine_npzs(npz_files))
            print(l + '\n  ' + '\n  '.join(npz_files))
            samples.append(s)
    else:
        samples = [
            bdtcode.sample.Sample(l, H.combine_npzs(glob.iglob(osp.join(directory, f'*/*{l}*/*.npz'))))
            for l in labels
            ]
    # Split the samples by category
    samples = split_by_category(samples, labels)
    return samples

def clean_label(label):
    """Strips off some verbosity from the label for printing purposes"""
    for p in [
        'Autumn18.', '_TuneCP5', '_13TeV', '_pythia8', '-pythia8',
        '-madgraphMLM', '-madgraph', 'genjetpt375_', '_mdark10_rinv0.3',
        'ToLNu', 'ToNuNu',
        ]:
        label = label.replace(p, '')
    label = label.replace('SingleLeptFromT', '1l_T')
    label = label.replace('DiLept', '2l')
    label = label.replace('_genMET-80', '_met80')
    return label


def xsweighted_bdt_efficiency(samples, min_score):
    """
    Takes a list of Samples and calculates the cross section weighted effiency of
    a particular bdt score.
    """
    total_xs = sum(s.crosssection for s in samples)
    return sum(s.bdt_efficiency(min_score)*s.crosssection for s in samples) / total_xs

def bdt_efficiency_table(bkgs, sigs, bdt_scores=None):
    if bdt_scores is None: bdt_scores = [.1*i for i in range(11)]
    # For the backgrounds, just make a cross-section weighted average per sample
    bkg_eff = [ xsweighted_bdt_efficiency(bkgs, min_score) for min_score in bdt_scores ]
    # For the signal we should not merge different mass points
    # Let's just make a dict, sig_eff[mz_mass] = [ list of bdt efficiencies per bdt value ]
    sig_eff = { sig.mz : [ sig.bdt_efficiency(min_score) for min_score in bdt_scores ] for sig in sigs }
    return bdt_scores, bkg_eff, sig_eff

@cli.command()
@click.argument('postbdtdir')
def print_quantiles(postbdtdir):
    *bkgs, sigs = get_samples_from_postbdt_directory(postbdtdir)
    bkgs = flatten(*bkgs) # Undo the grouping ttjets/qcd/wjets/zjets; just make a flat list
    bdt_scores, bkg_eff, sig_eff = bdt_efficiency_table(bkgs, sigs)
    # Print out - make a table
    def floats_to_strs(list_of_floats):
        return [f'{100.*number:.2f}%' for number in list_of_floats]
    mzs = [ s.mz for s in sigs ]
    header = ['bdt_cut', 'bkg_eff'] + [f'mz{mz:.0f}_eff' for mz in mzs]
    table = [ bdt_scores, floats_to_strs(bkg_eff) ] + [floats_to_strs(sig_eff[mz]) for mz in mzs]
    # Insert the header title at the first place of every row
    [ table[i].insert(0, head) for i, head in enumerate(header) ]
    print_table(table, transpose=True)

@cli.command()
@click.argument('postbdtdir')
def plot_sb(postbdtdir):
    import matplotlib.pyplot as plt
    *bkgs, sigs = get_samples_from_postbdt_directory(postbdtdir)
    bkgs = flatten(*bkgs) # Undo the grouping ttjets/qcd/wjets/zjets; just make a flat list

    bdt_scores = np.linspace(0., 1., 11)

    B = np.array([b.nevents_after_bdt(bdt_scores) for b in bkgs]).sum(axis=0)
    sqB = np.sqrt(B)
    S_per_mz = { s.mz : s.nevents_after_bdt(bdt_scores) for s in sigs }

    fig = plt.Figure()
    ax = fig.gca()

    for s in sigs:
        S = S_per_mz[s.mz]
        ax.plot(bdt_scores, S/sqB, label=f'$m_{{Z\prime}}={s.mz:.0f}$')






@cli.command()
@click.argument('postbdtdir')
@click.option('--group-summary', is_flag=True)
@click.option('--old-cuts', is_flag=True)
def print_statistics(postbdtdir, group_summary, old_cuts):
    new_cuts = not old_cuts
    if old_cuts:
        cutflow_keys = [
            '>=2jets',
            'eta<2.4',
            'trigger',
            'ecf>0',
            'rtx>1.1',
            'nleptons==0',
            'metfilter',
            'preselection',
            ]
    else:
        cutflow_keys = [
            '>=2jets',
            'eta<2.4',
            'jetak8>550',
            'trigger',
            'ecf>0',
            'rtx>1.1',
            'nleptons==0',
            'metfilter',
            'preselection',
            ]
    header_column = ['label', 'total'] + cutflow_keys + ['xs', 'genpt>375', 'n137_presel']
    if new_cuts: header_column.insert(-1, 'ttstitch')
    # print(f'{len(header_column)=}, {header_column=}')

    def format_group_column(samples):
        group_name = get_group_name(samples[0].label)
        group_xs = sum(s.crosssection for s in samples)
        def xs_weighted_sum(things_to_sum):
            return sum((s.crosssection/group_xs) * thing for s, thing in zip(samples, things_to_sum))
        column = [group_name, '100.00%']
        combined_cutflow = [
            xs_weighted_sum(s.d[key]/s.n_mc for s in samples) for key in cutflow_keys
            ]
        column.extend(f"{100*eff:.2f}%" for eff in combined_cutflow)
        column.append(f'{group_xs:.1f}')
        group_genjetpteff = xs_weighted_sum(s.genjetpt_efficiency for s in samples)
        column.append(f'{100.*group_genjetpteff:.2f}%')
        if new_cuts: column.append(f'{100.*s.ttstitch_efficiency:.2f}%')
        column.append(f"{sum(s.nevents_after_preselection() for s in samples):.0f}")
        return column

    def format_single_sample_column(sample):
        column = [clean_label(sample.label), '100.00%'] # , f'{xs:.1f}']
        
        column.extend(f"{100*sample.d[key]/sample.n_mc:.2f}%" for key in cutflow_keys)
        column.append(f'{sample.crosssection:.1f}')
        column.append(f'{100.*sample.genjetpt_efficiency:.3f}%')
        if new_cuts: column.append(f'{100.*sample.ttstitch_efficiency:.2f}%')
        column.append(f'{sample.nevents_after_preselection():.0f}')
        # print(f'{len(column)=}, {column=}')
        return column

    samples = get_samples_from_postbdt_directory(postbdtdir, debug=False)
    for s in flatten(*samples):
        for cutflow_key in cutflow_keys:
            if cutflow_key not in s.d:
                print(f'{cutflow_key=} not in {s.label}; {s.d=}')
                s.d[cutflow_key] = 0

    if group_summary:
        # Only print the cutflows per group, ignore individual sample level
        table = [header_column]
        # Combined background column
        table.append(format_group_column(flatten(*samples[:-1])))
        table[-1][0] = 'bkg'
        for sample_group in samples:
            if sample_group[0].is_sig:
                for sample in sample_group:
                    table.append(format_single_sample_column(sample))
            else:
                table.append(format_group_column(sample_group))
        print_table(table, transpose=True)
        return

    def print_cutflow(samples: List[bdtcode.sample.Sample]):
        table = [header_column]
        # Summary column for the whole group
        if samples[0].is_bkg: table.append(format_group_column(samples))
        # Individual samples
        for sample in samples: 
            table.append(format_single_sample_column(sample))
        print_table(table, transpose=True)

    for sample_group in samples:
        print_cutflow(sample_group)
        print('')

    samples = flatten(*samples)
    b = sum(sample.nevents_after_preselection() for sample in samples if sample.is_bkg)
    for sample in samples:
        if sample.is_bkg: continue
        s = sample.nevents_after_preselection()
        print(f'mz={sample.mz} {s=:.0f}, {b=:.0f}, s/sqrt(b)={s/np.sqrt(b):.3f}, s/b={s/b:.3f}')



def get_group_name(label):
    for pat in ['qcd', 'ttjets', 'wjets', 'zjets']:
    #for pat in ['qcd']:
        if pat in label.lower():
            return pat
    raise Exception(f'No group name for {label}')

@cli.command()
@click.option('-o', '--rootfile', default='test.root')
@click.argument('postbdt-dir')
def make_histograms(rootfile, postbdt_dir):
    *bkgs, sigs = get_samples_from_postbdt_directory(postbdt_dir)

    # binning = MT_BINNING
    left = 0.
    #right = 500.
    right = 1000.
    bin_width = 16.
    binning = [left+i*bin_width for i in range(math.ceil((right-left)/bin_width))]

    better_resolution_selectors = dict(pt_min=250, rt_min=1.15, dphi_max=100, eta_max=1)
    #better_resolution_selectors = {}


    with open_root(rootfile, 'RECREATE') as f:
        for min_score in [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        #for min_score in [None]:
            tdir = f.mkdir(f'bsvj_{0 if min_score is None else min_score:.1f}'.replace('.','p'))
            tdir.cd()
            # Loop over the mz' mass points
            for sig in sigs:
                h = bdtcode.sample.sample_to_mt_histogram(
                    sig, min_score=min_score, mt_binning=binning,
                    name=f'SVJ_mZprime{sig.mz:.0f}_mDark10_rinv03_alphapeak',
                    **better_resolution_selectors
                    )
                print(f'Writing {h.GetName()} --> {rootfile}/{tdir.GetName()}')
                h.Write()

            hs_bkg = []
            for bkg_group in bkgs:
                print(bkg_group[0].label)
                h = H.sum_th1s(
                  get_group_name(bkg_group[0].label),
                  (bdtcode.sample.sample_to_mt_histogram(s, min_score=min_score, mt_binning=binning, **better_resolution_selectors) for s in bkg_group)
                  )
                print(f'Writing {h.GetName()} --> {rootfile}/{tdir.GetName()}')
                h.Write()
                hs_bkg.append(h)

            # Finally write the combined bkg histogram
            h = H.sum_th1s('Bkg', hs_bkg)
            print(f'Writing {h.GetName()} --> {rootfile}/{tdir.GetName()}')
            h.Write()

            # Also write data_obs, which is the exact same thing but with a different name
            h.SetNameTitle('data_obs', 'data_obs')
            print(f'Writing {h.GetName()} --> {rootfile}/{tdir.GetName()}')
            h.Write()


@cli.command()
@click.argument('rootfile')
@click.option('-o', '--plotdir', default='plots_%b%d')
@click.option('--pdf', is_flag=True)
def plot_mt_histograms(rootfile, plotdir, pdf):
    import matplotlib.pyplot as plt
    set_matplotlib_fontsizes(18, 20, 24)
    plotdir = mkdir(plotdir)

    def mz(name):
        return int(re.search(r'mZprime(\d+)', name).group(1))

    def mzlabel(name):
        return r'$m_{Z\prime}$ = ' + str(mz(name))

    def save(outname):
        print(f"Saving {outname}{' with pdf' if pdf else ''}")
        plt.savefig(outname, bbox_inches='tight')
        if pdf: plt.savefig(outname.replace('.png', '.pdf'), bbox_inches='tight')
        plt.clf()

    with open_root(rootfile) as f:
        bdt_dirnames = [k.GetName() for k in f.GetListOfKeys()]
        for bdt_dirname in bdt_dirnames:
            d = f.Get(bdt_dirname)
            histograms = {k.GetName() : d.Get(k.GetName()) for k in d.GetListOfKeys()}
            signal_histograms = { k: v for k, v in histograms.items() if k.startswith('SVJ') }
            bkg_histograms = { k: v for k, v in histograms.items() if not k.startswith('SVJ') }

            fig = plt.figure(figsize=(8,8))
            ax = fig.gca()
            for i, (name, h) in enumerate(signal_histograms.items()):
                binning, vals = th1_binning_and_values(h)
                ax.step(binning[:-1], vals, where='post', label=mzlabel(name))
            ax.legend()
            ax.set_ylabel(r'$N_{events}$ / bin')
            ax.set_xlabel(r'$m_{T}$ (GeV)')
            save(osp.join(plotdir, f'{bdt_dirname}_signals.png'))

            fig = plt.figure(figsize=(8,8))
            ax = fig.gca()
            for i, (name, h) in enumerate(bkg_histograms.items()):
                binning, vals = th1_binning_and_values(h)
                ax.step(binning[:-1], vals, where='post', label=name)
            ax.legend()
            ax.set_ylabel(r'$N_{events}$ / bin')
            ax.set_xlabel(r'$m_{T}$ (GeV)')
            ax.set_yscale('log')
            save(osp.join(plotdir, f'{bdt_dirname}_bkgs.png'))

        fig = plt.figure(figsize=(8,8))
        ax = fig.gca()
        for bdt_dirname in bdt_dirnames:
            h_bkg = f.Get(bdt_dirname + '/Bkg')
            binning, vals = th1_binning_and_values(h_bkg)
            ax.step(binning[:-1], vals, where='post', label=bdt_dirname)
        ax.legend()
        ax.set_ylabel(r'$N_{events}$ / bin')
        ax.set_xlabel(r'$m_{T}$ (GeV)')
        # ax.set_yscale('log')
        save(osp.join(plotdir, f'bkg_per_bdtscore.png'))
        
        fig = plt.figure(figsize=(8,8))
        ax = fig.gca()
        for bdt_dirname in bdt_dirnames:
            h = f.Get(bdt_dirname + '/SVJ_mZprime350_mDark10_rinv03_alphapeak')
            binning, vals = th1_binning_and_values(h)
            ax.step(binning[:-1], vals, where='post', label=bdt_dirname)
        ax.legend()
        ax.set_ylabel(r'$N_{events}$ / bin')
        ax.set_xlabel(r'$m_{T}$ (GeV)')
        save(osp.join(plotdir, f'mz350_per_bdtscore.png'))




# 'SVJ_mZprime250_mDark10_rinv03_alphapeak',
# 'SVJ_mZprime300_mDark10_rinv03_alphapeak',
# 'SVJ_mZprime350_mDark10_rinv03_alphapeak',
# 'SVJ_mZprime400_mDark10_rinv03_alphapeak',
# 'SVJ_mZprime450_mDark10_rinv03_alphapeak',
# 'SVJ_mZprime500_mDark10_rinv03_alphapeak',
# 'SVJ_mZprime550_mDark10_rinv03_alphapeak',
# 'qcd',
# 'ttjets',
# 'wjets',
# 'zjets',
# 'Bkg'





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




@cli.command()
def dev_sample():
    directory = 'postbdt/girthreweight_fromumd_feb23'
    labels = list(set(osp.basename(s) for s in glob.iglob(osp.join(directory, '*/*'))))
    labels.sort()
    label = labels[0]
    print(f'Creating sample for {label=}')
    sample = bdtcode.sample.Sample(label, H.combine_npzs(glob.iglob(osp.join(directory, f'*/*{label}*/*.npz'))))
    import pprint
    pprint.pprint(sample.d)

    for key, value in sample.d.items():
        print(f'{key}: {value.shape if is_array(value) else value}')


if __name__ == '__main__':
    cli()
