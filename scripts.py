import glob, math, re
import ROOT
import numpy as np
import xgboost as xgb
import seutils
from combine_hists import *
import itertools
from array import array
from contextlib import contextmanager

try:
    import click
except ImportError:
    print('First install click:\npip install click')
    raise

def get_model(modeljson):
    model = xgb.XGBClassifier()
    model.load_model(modeljson)
    return model


@click.group()
def cli():
    pass

@contextmanager
def open_root(*args, **kwargs):
    try:
        f = ROOT.TFile.Open(*args, **kwargs)
        yield f
    finally:
        f.Close()

def flatten(*args):
    return list(itertools.chain.from_iterable(args))

def repeat_interleave(contents, counts):
    for item, count in zip(contents, counts):
        for i in range(count):
            yield item

# BKG order:
qcd_labels = [
    'Autumn18.QCD_Pt_300to470_TuneCP5_13TeV_pythia8',
    'Autumn18.QCD_Pt_470to600_TuneCP5_13TeV_pythia8',
    'Autumn18.QCD_Pt_600to800_TuneCP5_13TeV_pythia8',
    'Autumn18.QCD_Pt_800to1000_TuneCP5_13TeV_pythia8_ext1',
    'Autumn18.QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8',
    ]
ttjets_labels = [
    'Autumn18.TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8'
    'Autumn18.TTJets_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8'
    'Autumn18.TTJets_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8'
    'Autumn18.TTJets_HT-1200to2500_TuneCP5_13TeV-madgraphMLM-pythia8'
    'Autumn18.TTJets_HT-2500toInf_TuneCP5_13TeV-madgraphMLM-pythia8'
    'Autumn18.TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8'
    'Autumn18.TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8'
    'Autumn18.TTJets_TuneCP5_13TeV-madgraphMLM-pythia8'
    # 'Autumn18.TTJets_DiLept_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8'
    # 'Autumn18.TTJets_SingleLeptFromT_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8'
    # 'Autumn18.TTJets_SingleLeptFromTbar_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8'
    ]
wjets_labels = [
    'Autumn18.WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8',
    'Autumn18.WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8',
    'Autumn18.WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8',
    'Autumn18.WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8',
    'Autumn18.WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8',
    'Autumn18.WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8',
    'Autumn18.WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8',
    ]
zjets_labels = [
    'Autumn18.ZJetsToNuNu_HT-200To400_13TeV-madgraph',
    'Autumn18.ZJetsToNuNu_HT-400To600_13TeV-madgraph',
    'Autumn18.ZJetsToNuNu_HT-600To800_13TeV-madgraph',
    'Autumn18.ZJetsToNuNu_HT-800To1200_13TeV-madgraph',
    'Autumn18.ZJetsToNuNu_HT-1200To2500_13TeV-madgraph',
    'Autumn18.ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph',
    ]

import crosssections
qcd_xs = crosssections.labels_to_xs(qcd_labels)
ttjets_xs = crosssections.labels_to_xs(ttjets_labels)
wjets_xs = crosssections.labels_to_xs(wjets_labels)
zjets_xs = crosssections.labels_to_xs(zjets_labels)

old_qcd_xs = np.array([6826.0, 552.6, 156.6, 26.3, 7.5])
old_ttjets_xs = np.array([831.8*0.105, 1.808, 0.7490, 831.8*0.219, 831.8*0.219])
old_wjets_xs = np.array([1393.0, 409.9, 57.80, 12.94, 5.451, 1.085, 0.008060])
old_zjets_xs = np.array([304.0, 91.68, 13.11, 3.245, 1.497, 0.3425, 0.005263])


bkg_xs = np.concatenate((qcd_xs, ttjets_xs, wjets_xs, zjets_xs))
bkg_labels = flatten(qcd_labels, ttjets_labels, wjets_labels, zjets_labels)

mz_labels = ['mz250', 'mz300', 'mz350']
# all_xs = np.concatenate((qcd_xs, ttjets_xs, wjets_xs, zjets_xs, mz_xs))
# all_labels = flatten(qcd_labels, ttjets_labels, wjets_labels, zjets_labels, mz_labels)

def format_table(table, col_sep=' ', row_sep='\n', transpose=False):
    def format(s):
        try:
            number_f = float(s)
            number_i = int(s)
            if number_f != number_i:
                return f'{s:.2f}'
            else:
                return f'{s:.0f}'
        except ValueError:
            return str(s)
    table = [ [format(c) for c in row ] for row in table ]
    if transpose: table = list(zip(*table))
    col_widths = [ max(map(len, column)) for column in zip(*table) ]
    return row_sep.join(
        col_sep.join(f'{col:{w}s}' for col, w in zip(row, col_widths)) for row in table
        )

def print_table(*args, **kwargs):
    print(format_table(*args, **kwargs))



@cli.command()
def dev_dicts():
    get_dicts_from_postbdt_directory('postbdt_npzs_Oct05')


def get_dicts_from_postbdt_directory(directory, split=False):
    print(f'Building combined dicts from {directory}')
    labels = list(set(osp.basename(s) for s in glob.iglob(osp.join(directory, '*/*'))))
    labels.sort()
    # FIXME: Aren't we double counting with these included?
    # We can include them but we can't just add them on top
    # of the other directories, we have to correct something?
    labels.remove('Autumn18.TTJets_SingleLeptFromT_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8')
    labels.remove('Autumn18.TTJets_SingleLeptFromTbar_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8')
    labels.remove('Autumn18.TTJets_DiLept_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8')
    labels.remove('Autumn18.TTJets_TuneCP5_13TeV-madgraphMLM-pythia8')
    ds = [combine_npzs(glob.iglob(osp.join(directory, f'*/*{l}*/*.npz'))) for l in labels]
    xs = crosssections.labels_to_xs(labels)
    if split:
        # Return splitted by QCD, TTJets, WJets, ZJets, mz
        from operator import itemgetter
        getters = [
            itemgetter(*(i for i, label in enumerate(labels) if pat in label))
            for pat in [ 'QCD', 'TTJets', 'WJets', 'ZJets', 'mz' ]
            ]        
        return [(getter(labels), getter(xs), getter(ds)) for getter in getters]
    return labels, xs, ds


@cli.command()
def print_statistics_Oct11():
    def clean_label(label):
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
        'rtx>1.08',
        'nleptons==0',
        'metfilter',
        'preselection',
        ]
    header_column = ['label', 'xs'] + cutflow_keys
    def print_cutflow(labels, xss, dicts):
        table = [header_column]
        for label, xs, d in zip(labels, xss, dicts):
            column = [clean_label(label), f'{xs:.1f}']
            for cutflow_key in cutflow_keys:
                column.append(f"{100*d[cutflow_key] / d['total']:.2f}%")
            table.append(column)
        print_table(table, transpose=True)

    # for labels, xss, dicts in get_dicts_from_postbdt_directory('postbdt_npzs_Oct11_5masspoints', split=True):
    for labels, xss, dicts in get_dicts_from_postbdt_directory('postbdt_npzs_Oct12', split=True):
        print_cutflow(labels, xss, dicts)


@cli.command()
@click.option('-o', '--rootfile', default='test.root')
@click.option('-d', '--postbdt-dir', default='postbdt_npzs_Oct12')
def make_histograms(rootfile, postbdt_dir):
    *bkgs, sig = get_dicts_from_postbdt_directory(postbdt_dir, split=True)
    def calc_n137(xss, ds):
        return np.array([d['preselection']/d['total'] * xs * 137.2*1e3 for d, xs in zip(ds, xss)])
    bkgs_n137 = [ calc_n137(bkg[1], bkg[2]) for bkg in bkgs ]
    sig_n137 = calc_n137(sig[1], sig[2])

    combined_bkg = flatten(*[bkg[2] for bkg in bkgs])
    combined_bkg_n137 = flatten(*bkgs_n137)

    # Make combined bkg dict, only for calculating the BDT thresholds at various
    # bkg rejection rates
    bkg_weighted = combine_ds_with_weights(combined_bkg, combined_bkg_n137)
    quantiles = .1*np.arange(1,10)
    thresholds = np.quantile(bkg_weighted['score'], quantiles)

    # binning = MT_BINNING
    left = 210.
    #right = 500.
    right = 800.
    bin_width = 8.
    binning = [left+i*bin_width for i in range(math.ceil((right-left)/bin_width))]

    # Now make the histograms for various thresholds
    def make_and_write(*args, **kwargs):
        kwargs['mt_binning'] = binning
        h = make_summed_histogram(*args, **kwargs)
        print(f'Writing {h.GetName()} --> {rootfile}/{tdir.GetName()}')
        h.Write()
        return h

    with open_root(rootfile, 'RECREATE') as f:
        # Dump many bkg rejections
        for threshold, bkg_rejection in zip([None] + list(thresholds), .1*np.arange(10)):
            print(f'Writing histograms @ {bkg_rejection=:.2f} (bdt_score>{threshold})')
            tdir = f.mkdir(f'bkg_rejection_{bkg_rejection:.2f}'.replace('.','p'))
            tdir.cd()

            signal_histograms = []
            for label, _, d, norm in zip(*sig, sig_n137):
                label = re.search(r'mz\d+', label).group()
                h = make_mt_histogram(label, d['mt'], d['score'], threshold, normalization=norm, mt_binning=binning)
                print(f'Writing {h.GetName()} --> {rootfile}/{tdir.GetName()}')
                h.Write()
                signal_histograms.append(h)

            for process, norms, (_, _, dicts) in zip(
                ['qcd', 'ttjets', 'wjets', 'zjets'], bkgs_n137, bkgs
                ):
                make_and_write(process, dicts, norms, threshold=threshold)

            h_bkg = make_and_write('bkg', combined_bkg, combined_bkg_n137, threshold=threshold)


            tdir = f.mkdir('bsvj_{}'.format(int(100.*bkg_rejection)))
            tdir.cd()

            h_bkg.SetNameTitle('data_obs', 'data_obs')
            print(f'Writing {h_bkg.GetName()} --> {rootfile}/{tdir.GetName()}')
            h_bkg.Write()
            h_bkg.SetNameTitle('Bkg', 'Bkg')
            print(f'Writing {h_bkg.GetName()} --> {rootfile}/{tdir.GetName()}')
            h_bkg.Write()

            for h in signal_histograms:
                name = f'SVJ_mZprime{h.GetName().replace("mz","")}_mDark10_rinv03_alphapeak'
                h.SetNameTitle(name, name)
                print(f'Writing {h.GetName()} --> {rootfile}/{tdir.GetName()}')
                h.Write()


@cli.command()
def print_statistics_Oct05():
    labels, xss, dicts = get_dicts_from_postbdt_directory('postbdt_npzs_Oct05')
    table = [['label', 'xs', 'eff_xs', 'n_total', 'n_presel', 'frac', 'N_137']]
    for label, xs, d in zip(labels, xss, dicts):
        n_total = d["n_total"]
        n_presel = d["n_presel"]
        frac = n_presel / n_total
        eff_xs = frac * xs
        N_137 = int(eff_xs * 137.2*1e3)
        table.append([label, xs, eff_xs, n_total, n_presel, frac, N_137])
    print_table(table)


@cli.command()
@click.option('-o', '--rootfile', default='test.root')
def make_histograms_Oct05(rootfile):
    """
    BDT version Sep22, mz=250,300,350, all 4 bkgs
    """
    try_import_ROOT()
    import ROOT
    qcd, ttjets, wjets, zjets, signal = get_dicts_from_postbdt_directory('postbdt_npzs_Oct05', split=True)

    def calc_n_events_at_lumi(dicts, xss, lumi=137.2):
        return np.array([d["n_presel"]/d["n_total"] * xs * lumi*1e3 for d, xs in zip(dicts, xss)])

    qcd_n137 = calc_n_events_at_lumi(qcd[2], qcd[1])
    ttjets_n137 = calc_n_events_at_lumi(ttjets[2], ttjets[1])
    wjets_n137 = calc_n_events_at_lumi(wjets[2], wjets[1])
    zjets_n137 = calc_n_events_at_lumi(zjets[2], zjets[1])
    signal_n137 = calc_n_events_at_lumi(signal[2], signal[1])
    
    bkg = flatten(qcd[2], ttjets[2], wjets[2], zjets[2])
    bkg_n137 = np.concatenate((qcd_n137, ttjets_n137, wjets_n137, zjets_n137))

    print('Expected # events qcd    =', qcd_n137.sum())
    print('Expected # events ttjets =', ttjets_n137.sum())
    print('Expected # events wjets  =', wjets_n137.sum())
    print('Expected # events zjets  =', zjets_n137.sum())
    print('Expected # events bkg    =', bkg_n137.sum())
    print('Expected # events signal =', signal_n137)

    # Make combined bkg dict, only for calculating the BDT thresholds at various
    # bkg rejection rates
    bkg_weighted = combine_ds_with_weights(bkg, bkg_n137)
    quantiles = .1*np.arange(1,10)
    thresholds = np.quantile(bkg_weighted['score'], quantiles)

    # binning = MT_BINNING
    left = 210.
    #right = 500.
    right = 800.
    bin_width = 8.
    binning = [left+i*bin_width for i in range(math.ceil((right-left)/bin_width))]

    # Now make the histograms for various thresholds
    def make_and_write(*args, **kwargs):
        kwargs['mt_binning'] = binning
        h = make_summed_histogram(*args, **kwargs)
        print(f'Writing {h.GetName()} --> {rootfile}')
        h.Write()
        return h

    with open_root(rootfile, 'RECREATE') as f:
        # Dump many bkg rejections
        for threshold, bkg_rejection in zip([None] + list(thresholds), .1*np.arange(10)):
            print(f'Writing histograms @ {bkg_rejection=:.2f} (bdt_score>{threshold})')
            tdir = f.mkdir(f'bkg_rejection_{bkg_rejection:.2f}'.replace('.','p'))
            tdir.cd()
            make_and_write('qcd', qcd[2], qcd_n137, threshold=threshold)
            make_and_write('ttjets', ttjets[2], ttjets_n137, threshold=threshold)
            make_and_write('wjets', wjets[2], wjets_n137, threshold=threshold)
            make_and_write('zjets', zjets[2], zjets_n137, threshold=threshold)
            make_and_write('bkg', bkg, bkg_n137, threshold=threshold)
            for name, d, norm in zip(signal[0], signal[2], signal_n137):
                h = make_mt_histogram(name, d['mt'], d['score'], threshold, normalization=norm, mt_binning=binning)
                print(f'Writing {h.GetName()} --> {rootfile}')
                h.Write()

            # For Sara: Different naming scheme
            tdir = f.mkdir('bsvj_{}'.format(int(100.*bkg_rejection)))
            tdir.cd()
            # Write Bkg and data_obs
            h = make_and_write('Bkg', bkg, bkg_n137, threshold=threshold)
            h.SetNameTitle('data_obs', 'data_obs')
            h.Write()
            # Write the signal histograms
            for name, d, norm in zip(signal[0], signal[2], signal_n137):
                name = f'SVJ_mZprime{name.replace("mz","")}_mDark10_rinv03_alphapeak'
                h = make_mt_histogram(name, d['mt'], d['score'], threshold, normalization=norm, mt_binning=binning)
                print(f'Writing {h.GetName()} --> {rootfile}')
                h.Write()


# ________________________________________________________
# BELOW THIS LINE IS OUTDATED

# Some weights
qcd_presel_eff = np.array([0.00544, 0.07959, 0.10906, 0.08919, 0.07280])
qcd_crosssections = np.array([6826.0, 552.6, 156.6, 26.3, 7.5])
qcd_eff_xs = qcd_presel_eff * qcd_crosssections

# ttjets order:
# Autumn18.TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8              831.8 * 0.105
# Autumn18.TTJets_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8         1.808
# Autumn18.TTJets_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8        0.7490
# Autumn18.TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8     831.8 * 0.219
# Autumn18.TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8  831.8 * 0.219
ttjets_presel_eff = np.array([0.00311, 0.15391, 0.30009, 0.00385, 0.00375])
ttjets_crosssections = np.array([831.8*0.105, 1.808, 0.7490, 831.8*0.219, 831.8*0.219])
ttjets_eff_xs = ttjets_presel_eff * ttjets_crosssections

# FIXME: mz350 cross section extrapolated now, trigger+ptjet1>550 factors taken from mz250 alone
mz_presel_eff = np.array([0.13847, 0.14312, 0.15442])
mz_crosssections = 0.00191*0.233 * np.array([34820, 23430, 23430-(34820-23430)])
mz_eff_xs = mz_presel_eff * mz_crosssections

bkg_crosssections = np.concatenate((qcd_crosssections, ttjets_crosssections))
all_crosssections = np.concatenate((mz_crosssections, qcd_crosssections, ttjets_crosssections))

bkg_eff_xs = np.concatenate((qcd_eff_xs, ttjets_eff_xs))
all_eff_xs = np.concatenate((qcd_eff_xs, ttjets_eff_xs, mz_eff_xs))


def sort_directory_heuristic(directory):
    import re
    directory = osp.basename(directory).lower()
    if 'mz' in directory:
        mz = int(re.search(r'mz(\d+)', directory).group(1))
        return 1e5 + mz, directory
    elif 'qcd' in directory:
        ptlow = int(re.search(r'pt_(\d+)to', directory).group(1))
        return 1e3 + ptlow, directory
    elif 'ttjets' in directory:
        return 1e4, directory
    else:
        return 1e9, directory


def flatten(*args):
    return list(itertools.chain.from_iterable(args))

def repeat_interleave(contents, counts):
    for item, count in zip(contents, counts):
        for i in range(count):
            yield item

# BKG order:
qcd_labels = [
    'Autumn18.QCD_Pt_300to470_TuneCP5_13TeV_pythia8',
    'Autumn18.QCD_Pt_470to600_TuneCP5_13TeV_pythia8',
    'Autumn18.QCD_Pt_600to800_TuneCP5_13TeV_pythia8',
    'Autumn18.QCD_Pt_800to1000_TuneCP5_13TeV_pythia8_ext1',
    'Autumn18.QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8',
    ]
ttjets_labels = [
    'Autumn18.TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8'
    'Autumn18.TTJets_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8'
    'Autumn18.TTJets_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8'
    'Autumn18.TTJets_HT-1200to2500_TuneCP5_13TeV-madgraphMLM-pythia8'
    'Autumn18.TTJets_HT-2500toInf_TuneCP5_13TeV-madgraphMLM-pythia8'
    'Autumn18.TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8'
    'Autumn18.TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8'
    'Autumn18.TTJets_TuneCP5_13TeV-madgraphMLM-pythia8'
    # 'Autumn18.TTJets_DiLept_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8'
    # 'Autumn18.TTJets_SingleLeptFromT_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8'
    # 'Autumn18.TTJets_SingleLeptFromTbar_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8'
    ]
wjets_labels = [
    'Autumn18.WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8',
    'Autumn18.WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8',
    'Autumn18.WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8',
    'Autumn18.WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8',
    'Autumn18.WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8',
    'Autumn18.WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8',
    'Autumn18.WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8',
    ]
zjets_labels = [
    'Autumn18.ZJetsToNuNu_HT-200To400_13TeV-madgraph',
    'Autumn18.ZJetsToNuNu_HT-400To600_13TeV-madgraph',
    'Autumn18.ZJetsToNuNu_HT-600To800_13TeV-madgraph',
    'Autumn18.ZJetsToNuNu_HT-800To1200_13TeV-madgraph',
    'Autumn18.ZJetsToNuNu_HT-1200To2500_13TeV-madgraph',
    'Autumn18.ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph',
    ]

import crosssections
qcd_xs = crosssections.labels_to_xs(qcd_labels)
ttjets_xs = crosssections.labels_to_xs(ttjets_labels)
wjets_xs = crosssections.labels_to_xs(wjets_labels)
zjets_xs = crosssections.labels_to_xs(zjets_labels)

old_qcd_xs = np.array([6826.0, 552.6, 156.6, 26.3, 7.5])
old_ttjets_xs = np.array([831.8*0.105, 1.808, 0.7490, 831.8*0.219, 831.8*0.219])
old_wjets_xs = np.array([1393.0, 409.9, 57.80, 12.94, 5.451, 1.085, 0.008060])
old_zjets_xs = np.array([304.0, 91.68, 13.11, 3.245, 1.497, 0.3425, 0.005263])


bkg_xs = np.concatenate((qcd_xs, ttjets_xs, wjets_xs, zjets_xs))
bkg_labels = flatten(qcd_labels, ttjets_labels, wjets_labels, zjets_labels)

mz_labels = ['mz250', 'mz300', 'mz350']
# all_xs = np.concatenate((qcd_xs, ttjets_xs, wjets_xs, zjets_xs, mz_xs))
# all_labels = flatten(qcd_labels, ttjets_labels, wjets_labels, zjets_labels, mz_labels)

def format_table(table, col_sep=' ', row_sep='\n', transpose=False):
    def format(s):
        try:
            number_f = float(s)
            number_i = int(s)
            if number_f != number_i:
                return f'{s:.2f}'
            else:
                return f'{s:.0f}'
        except ValueError:
            return str(s)
    table = [ [format(c) for c in row ] for row in table ]
    if transpose: table = list(zip(*table))
    col_widths = [ max(map(len, column)) for column in zip(*table) ]
    return row_sep.join(
        col_sep.join(f'{col:{w}s}' for col, w in zip(row, col_widths)) for row in table
        )

def print_table(*args, **kwargs):
    print(format_table(*args, **kwargs))



@cli.command()
def dev_dicts():
    get_dicts_from_postbdt_directory('postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets')


def get_dicts_from_postbdt_directory(directory, split=False):
    print(f'Building combined dicts from {directory}')
    labels = list(set(osp.basename(s) for s in glob.iglob(osp.join(directory, '*/*'))))
    labels.sort()
    # FIXME: Aren't we double counting with these included?
    # We can include them but we can't just add them on top
    # of the other directories, we have to correct something?
    #'''labels.remove('Autumn18.TTJets_SingleLeptFromT_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8')
    #labels.remove('Autumn18.TTJets_SingleLeptFromTbar_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8')
    #labels.remove('Autumn18.TTJets_DiLept_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8')
    #labels.remove('Autumn18.TTJets_TuneCP5_13TeV-madgraphMLM-pythia8')'''
    ds = [combine_npzs(glob.iglob(osp.join(directory, f'*/*{l}*/*.npz'))) for l in labels]
    xs = crosssections.labels_to_xs(labels)
    if split:
        # Return splitted by QCD, TTJets, WJets, ZJets, mz
        from operator import itemgetter
        getters = [
            itemgetter(*(i for i, label in enumerate(labels) if pat in label))
            for pat in [ 'QCD', 'TTJets', 'WJets', 'ZJets', 'mz' ]
            ]        
        return [(getter(labels), getter(xs), getter(ds)) for getter in getters]
    return labels, xs, ds


@cli.command()
@click.argument('postbdtdir')
def print_statistics(postbdtdir):
    def clean_label(label):
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
    header_column = ['label', 'xs'] + cutflow_keys
    def print_cutflow(labels, xss, dicts):
        table = [header_column]
        for label, xs, d in zip(labels, xss, dicts):
            column = [clean_label(label), f'{xs:.1f}']
            for cutflow_key in cutflow_keys:
                column.append(f"{100*d[cutflow_key] / d['total']:.2f}%")
            table.append(column)
        print_table(table, transpose=True)

    for labels, xss, dicts in get_dicts_from_postbdt_directory(postbdtdir, split=True):
        print_cutflow(labels, xss, dicts)


@cli.command()
@click.option('-o', '--rootfile', default='test.root')
@click.argument('postbdt_dir', default='postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets')
def make_histograms(rootfile, postbdt_dir):
    *bkgs, sig = get_dicts_from_postbdt_directory(postbdt_dir, split=True)
    def calc_n137(xss, ds):
        return np.array([d['preselection']/d['total'] * xs * 137.2*1e3 for d, xs in zip(ds, xss)])
    bkgs_n137 = [ calc_n137(bkg[1], bkg[2]) for bkg in bkgs ]
    sig_n137 = calc_n137(sig[1], sig[2])

    combined_bkg = flatten(*[bkg[2] for bkg in bkgs])
    combined_bkg_n137 = flatten(*bkgs_n137)

    # Make combined bkg dict, only for calculating the BDT thresholds at various
    # bkg rejection rates
    bkg_weighted = combine_ds_with_weights(combined_bkg, combined_bkg_n137)
    quantiles = .1*np.arange(1,10)
    thresholds = np.quantile(bkg_weighted['score'], quantiles)

    # binning = MT_BINNING
    left = 210.
    right = 500.
    bin_width = 8.
    binning = [left+i*bin_width for i in range(math.ceil((right-left)/bin_width))]

    # Now make the histograms for various thresholds
    def make_and_write(*args, **kwargs):
        kwargs['mt_binning'] = binning
        h = make_summed_histogram(*args, **kwargs)
        print(f'Writing {h.GetName()} --> {rootfile}/{tdir.GetName()}')
        h.Write()
        return h

    with open_root(rootfile, 'RECREATE') as f:
        # Dump many bkg rejections
        for threshold, bkg_rejection in zip([None] + list(thresholds), .1*np.arange(10)):
            print(f'Writing histograms @ {bkg_rejection=:.2f} (bdt_score>{threshold})')
            tdir = f.mkdir(f'bkg_rejection_{bkg_rejection:.2f}'.replace('.','p'))
            tdir.cd()

            signal_histograms = []
            for label, _, d, norm in zip(*sig, sig_n137):
                label = re.search(r'mz\d+', label).group()
                h = make_mt_histogram(label, d['mt'], d['score'], threshold, normalization=norm, mt_binning=binning)
                print(f'Writing {h.GetName()} --> {rootfile}/{tdir.GetName()}')
                h.Write()
                signal_histograms.append(h)

            for process, norms, (_, _, dicts) in zip(
                ['qcd', 'ttjets', 'wjets', 'zjets'], bkgs_n137, bkgs
                ):
                make_and_write(process, dicts, norms, threshold=threshold)

            h_bkg = make_and_write('bkg', combined_bkg, combined_bkg_n137, threshold=threshold)


            tdir = f.mkdir('bsvj_{}'.format(int(100.*bkg_rejection)))
            tdir.cd()

            h_bkg.SetNameTitle('data_obs', 'data_obs')
            print(f'Writing {h_bkg.GetName()} --> {rootfile}/{tdir.GetName()}')
            h_bkg.Write()
            h_bkg.SetNameTitle('Bkg', 'Bkg')
            print(f'Writing {h_bkg.GetName()} --> {rootfile}/{tdir.GetName()}')
            h_bkg.Write()

            for h in signal_histograms:
                name = f'SVJ_mZprime{h.GetName().replace("mz","")}_mDark10_rinv03_alphapeak'
                h.SetNameTitle(name, name)
                print(f'Writing {h.GetName()} --> {rootfile}/{tdir.GetName()}')
                h.Write()


@cli.command()
def print_statistics_Oct05():
    labels, xss, dicts = get_dicts_from_postbdt_directory('postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets')
    table = [['label', 'xs', 'eff_xs', 'n_total', 'n_presel', 'frac', 'N_137']]
    for label, xs, d in zip(labels, xss, dicts):
        n_total = d["n_total"]
        n_presel = d["n_presel"]
        frac = n_presel / n_total
        eff_xs = frac * xs
        N_137 = int(eff_xs * 137.2*1e3)
        table.append([label, xs, eff_xs, n_total, n_presel, frac, N_137])
    print_table(table)


@cli.command()
@click.option('-o', '--rootfile', default='test.root')
@click.argument('postbdtdir')
def make_histograms_Oct05(postbdtdir, rootfile):
    """
    BDT version Sep22, mz=250,300,350, all 4 bkgs
    """
    try_import_ROOT()
    import ROOT
    qcd, ttjets, wjets, zjets, signal = get_dicts_from_postbdt_directory(postbdtdir, split=True)

    def calc_n_events_at_lumi(dicts, xss, lumi=137.2):
        return np.array([d["preselection"]/d["total"] * xs * lumi*1e3 for d, xs in zip(dicts, xss)])

    qcd_n137 = calc_n_events_at_lumi(qcd[2], qcd[1])
    ttjets_n137 = calc_n_events_at_lumi(ttjets[2], ttjets[1])
    wjets_n137 = calc_n_events_at_lumi(wjets[2], wjets[1])
    zjets_n137 = calc_n_events_at_lumi(zjets[2], zjets[1])
    signal_n137 = calc_n_events_at_lumi(signal[2], signal[1])
    
    bkg = flatten(qcd[2], ttjets[2], wjets[2], zjets[2])
    bkg_n137 = np.concatenate((qcd_n137, ttjets_n137, wjets_n137, zjets_n137))

    print('Expected # events qcd    =', qcd_n137.sum())
    print('Expected # events ttjets =', ttjets_n137.sum())
    print('Expected # events wjets  =', wjets_n137.sum())
    print('Expected # events zjets  =', zjets_n137.sum())
    print('Expected # events bkg    =', bkg_n137.sum())
    print('Expected # events signal =', signal_n137)

    # Make combined bkg dict, only for calculating the BDT thresholds at various
    # bkg rejection rates
    bkg_weighted = combine_ds_with_weights(bkg, bkg_n137)
    quantiles = .1*np.arange(1,10)
    thresholds = np.quantile(bkg_weighted['score'], quantiles)

    # binning = MT_BINNING
    left = 210.
    right = 500.
    bin_width = 8.
    binning = [left+i*bin_width for i in range(math.ceil((right-left)/bin_width))]

    # Now make the histograms for various thresholds
    def make_and_write(*args, **kwargs):
        kwargs['mt_binning'] = binning
        h = make_summed_histogram(*args, **kwargs)
        print(f'Writing {h.GetName()} --> {rootfile}')
        h.Write()
        return h

    with open_root(rootfile, 'RECREATE') as f:
        # Dump many bkg rejections
        for threshold, bkg_rejection in zip([None] + list(thresholds), .1*np.arange(10)):
            print(f'Writing histograms @ {bkg_rejection=:.2f} (bdt_score>{threshold})')
            tdir = f.mkdir(f'bkg_rejection_{bkg_rejection:.2f}'.replace('.','p'))
            tdir.cd()
            make_and_write('qcd', qcd[2], qcd_n137, threshold=threshold)
            make_and_write('ttjets', ttjets[2], ttjets_n137, threshold=threshold)
            make_and_write('wjets', wjets[2], wjets_n137, threshold=threshold)
            make_and_write('zjets', zjets[2], zjets_n137, threshold=threshold)
            make_and_write('bkg', bkg, bkg_n137, threshold=threshold)
            for name, d, norm in zip(signal[0], signal[2], signal_n137):
                h = make_mt_histogram(name, d['mt'], d['score'], threshold, normalization=norm, mt_binning=binning)
                print(f'Writing {h.GetName()} --> {rootfile}')
                h.Write()

            # For Sara: Different naming scheme
            tdir = f.mkdir('bsvj_{}'.format(int(100.*bkg_rejection)))
            tdir.cd()
            # Write Bkg and data_obs
            h = make_and_write('Bkg', bkg, bkg_n137, threshold=threshold)
            h.SetNameTitle('data_obs', 'data_obs')
            h.Write()
            # Write the signal histograms
            for name, d, norm in zip(signal[0], signal[2], signal_n137):
                name = f'SVJ_mZprime{name.replace("mz","")}_mDark10_rinv03_alphapeak'
                h = make_mt_histogram(name, d['mt'], d['score'], threshold, normalization=norm, mt_binning=binning)
                print(f'Writing {h.GetName()} --> {rootfile}')
                h.Write()


# ________________________________________________________
# BELOW THIS LINE IS OUTDATED

# Some weights
qcd_presel_eff = np.array([0.00544, 0.07959, 0.10906, 0.08919, 0.07280])
qcd_crosssections = np.array([6826.0, 552.6, 156.6, 26.3, 7.5])
qcd_eff_xs = qcd_presel_eff * qcd_crosssections

# ttjets order:
# Autumn18.TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8              831.8 * 0.105
# Autumn18.TTJets_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8         1.808
# Autumn18.TTJets_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8        0.7490
# Autumn18.TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8     831.8 * 0.219
# Autumn18.TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8  831.8 * 0.219
ttjets_presel_eff = np.array([0.00311, 0.15391, 0.30009, 0.00385, 0.00375])
ttjets_crosssections = np.array([831.8*0.105, 1.808, 0.7490, 831.8*0.219, 831.8*0.219])
ttjets_eff_xs = ttjets_presel_eff * ttjets_crosssections

# FIXME: mz350 cross section extrapolated now, trigger+ptjet1>550 factors taken from mz250 alone
mz_presel_eff = np.array([0.13847, 0.14312, 0.15442])
mz_crosssections = 0.00191*0.233 * np.array([34820, 23430, 23430-(34820-23430)])
mz_eff_xs = mz_presel_eff * mz_crosssections

bkg_crosssections = np.concatenate((qcd_crosssections, ttjets_crosssections))
all_crosssections = np.concatenate((mz_crosssections, qcd_crosssections, ttjets_crosssections))

bkg_eff_xs = np.concatenate((qcd_eff_xs, ttjets_eff_xs))
all_eff_xs = np.concatenate((qcd_eff_xs, ttjets_eff_xs, mz_eff_xs))


def sort_directory_heuristic(directory):
    import re
    directory = osp.basename(directory).lower()
    if 'mz' in directory:
        mz = int(re.search(r'mz(\d+)', directory).group(1))
        return 1e5 + mz, directory
    elif 'qcd' in directory:
        ptlow = int(re.search(r'pt_(\d+)to', directory).group(1))
        return 1e3 + ptlow, directory
    elif 'ttjets' in directory:
        return 1e4, directory
    else:
        return 1e9, directory

@cli.command()
def preselection_efficiencies_bkg():
    directories = list(sorted(glob.iglob('postbdt_npzs_Oct21_5mass_qcdttjetswjetszjets/*'), key=sort_directory_heuristic))
    flength = max(len(osp.basename(d)) for d in directories)
    for i, directory in enumerate(directories):
        npzs = glob.glob(osp.join(directory, '*.npz'))
        d = combine_npzs(npzs)

        n_total = d["n_total"]
        n_presel = d["n_presel"]
        frac = n_presel / n_total
        eff_xs = frac * all_crosssections[i]
        N_137 = int(eff_xs * 137.2*1e3)

        print(
            f'{osp.basename(directory):{flength}s} :'
            f' n_total={d["n_total"]:9}, n_presel={d["n_presel"]:9},'
            f' frac={frac:.5f}'
            f' N@137.2fb-1={N_137}'
            )

def combine_dirs_with_weights(directories, weights):
    ds = [combine_npzs(glob.glob(osp.join(directory, '*.npz'))) for directory in directories]
    return combine_ds_with_weights(ds, weights)

def get_combined_qcd_bkg():
    print('Reading individual qcd .npzs')
    qcd_npzs = [
        glob.glob(osp.join(directory, '*.npz'))
        for directory in [
            'postbdt_npzs/Autumn18.QCD_Pt_300to470_TuneCP5_13TeV_pythia8',
            'postbdt_npzs/Autumn18.QCD_Pt_470to600_TuneCP5_13TeV_pythia8',
            'postbdt_npzs/Autumn18.QCD_Pt_600to800_TuneCP5_13TeV_pythia8',
            'postbdt_npzs/Autumn18.QCD_Pt_800to1000_TuneCP5_13TeV_pythia8_ext1',
            'postbdt_npzs/Autumn18.QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8',
            ]
        ]
    print('Creating combined dict for all qcd bins')
    qcd_ds = [ combine_npzs(npzs) for npzs in qcd_npzs ]
    print('Combining qcd bins with weights')
    qcd = combine_ds_with_weights(qcd_ds, [136.52, 278.51, 150.96, 26.24, 7.49])
    return qcd

# def get_combined_bkg():
#     qcd = combine_ds_with_weights(qcd_ds, qcd_sel_eff*qcd_crossections)


@cli.command()
@click.option('-o', '--outputfile', default='bkg.npz')
def get_combined_bkg(outputfile):
    print('Reading individual bkg .npzs')
    bkg_npzs = [
        for directory in [
            'postbdt_npzs_Oct21_5mass_qcdttjetswjetszjets/Autumn18.QCD_Pt_300to470_TuneCP5_13TeV_pythia8',
            'postbdt_npzs_Oct21_5mass_qcdttjetswjetszjets/Autumn18.QCD_Pt_470to600_TuneCP5_13TeV_pythia8',
            'postbdt_npzs_Oct21_5mass_qcdttjetswjetszjets/Autumn18.QCD_Pt_600to800_TuneCP5_13TeV_pythia8',
            'postbdt_npzs_Oct21_5mass_qcdttjetswjetszjets/Autumn18.QCD_Pt_800to1000_TuneCP5_13TeV_pythia8_ext1',
            'postbdt_npzs_Oct21_5mass_qcdttjetswjetszjets/Autumn18.QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8',
            'postbdt_npzs_Oct21_5mass_qcdttjetswjetszjets/Autumn18.TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8',
            'postbdt_npzs_Oct21_5mass_qcdttjetswjetszjets/Autumn18.TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8',
            'postbdt_npzs_Oct21_5mass_qcdttjetswjetszjets/Autumn18.TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8',
            'postbdt_npzs_Oct21_5mass_qcdttjetswjetszjets/Autumn18.TTJets_TuneCP5_13TeV-madgraphMLM-pythia8',
            'postbdt_npzs_Oct21_5mass_qcdttjetswjetszjets/Autumn18.TTJets_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8',
            'postbdt_npzs_Oct21_5mass_qcdttjetswjetszjets/Autumn18.TTJets_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8',
            'postbdt_npzs_Oct21_5mass_qcdttjetswjetszjets/Autumn18.TTJets_HT-1200to2500_TuneCP5_13TeV-madgraphMLM-pythia8',
            'postbdt_npzs_Oct21_5mass_qcdttjetswjetszjets/Autumn18.TTJets_HT-2500toInf_TuneCP5_13TeV-madgraphMLM-pythia8',
            ]
        ]
    print('Creating combined dict for all bkg bins')
    bkg_ds = [ combine_npzs(npzs) for npzs in bkg_npzs ]
    print('Combining bkg bins with weights')
    bkg = combine_ds_with_weights(bkg_ds, np.concatenate((qcd_presel_eff*qcd_crosssections, ttjets_presel_eff*ttjets_crosssections)))
    return bkg


@cli.command()
@click.option('-o', '--rootfile', default='test.root')
def make_histograms_3masspoints_qcd_ttjets(rootfile):
    """
    With bdt version trained on only mz250 and qcd
    """
    try_import_ROOT()
    import ROOT

    qcd_dirs = [
        'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.QCD_Pt_300to470_TuneCP5_13TeV_pythia8',
        'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.QCD_Pt_470to600_TuneCP5_13TeV_pythia8',
        'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.QCD_Pt_600to800_TuneCP5_13TeV_pythia8',
        'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.QCD_Pt_800to1000_TuneCP5_13TeV_pythia8_ext1',
        'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8',
        ]
    ttjets_dirs = [
        'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.TTJets_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.TTJets_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8',
        ]

    qcd = combine_dirs_with_weights(qcd_dirs, qcd_eff_xs)
    ttjets = combine_dirs_with_weights(ttjets_dirs, ttjets_eff_xs)
    bkg = combine_dirs_with_weights(qcd_dirs+ttjets_dirs, bkg_eff_xs)

    mz250 = combine_npzs(glob.glob('postbdt_npzs_Sep21_3masspoints_qcdttjets/genjetpt375_mz250_mdark10_rinv0.3/*.npz'))
    mz300 = combine_npzs(glob.glob('postbdt_npzs_Sep21_3masspoints_qcdttjets/genjetpt375_mz300_mdark10_rinv0.3/*.npz'))
    mz350 = combine_npzs(glob.glob('postbdt_npzs_Sep21_3masspoints_qcdttjets/genjetpt375_mz350_mdark10_rinv0.3/*.npz'))

    norm_qcd    = int(qcd_eff_xs.sum() * 137200)
    norm_ttjets = int(ttjets_eff_xs.sum() * 137200)
    norm_bkg    = int(bkg_eff_xs.sum() * 137200)
    norm_mz250  = int(mz_eff_xs[0] * 137200)
    norm_mz300  = int(mz_eff_xs[1] * 137200)
    norm_mz350  = int(mz_eff_xs[2] * 137200)

    print(f'{norm_qcd=}')
    print(f'{norm_ttjets=}')
    print(f'{norm_bkg=}')
    print(f'{norm_mz250=}')
    print(f'{norm_mz300=}')
    print(f'{norm_mz350=}')

    # Compute thresholds for every 10% quantile
    quantiles = np.array([i*.1 for i in range(1,10)])
    thresholds = np.quantile(bkg['score'], quantiles)

    try:
        f = ROOT.TFile.Open(rootfile, 'RECREATE')

        def dump(d, name, threshold=None, use_threshold_in_name=True, norm=None):
            """
            Mini function to write a dictionary to the open root file
            """
            if threshold is not None and use_threshold_in_name:
                name += f'_{float(quantiles[thresholds == threshold]):.3f}'
            print(f'Writing {name} --> {rootfile}')
            h = make_mt_histogram(name, d['mt'], d['score'], threshold, normalization=norm)
            h.Write()

        dump(qcd, 'qcd_mt', norm=norm_qcd)
        dump(ttjets, 'ttjets_mt', norm=norm_ttjets)
        dump(bkg, 'bkg_mt', norm=norm_bkg)
        dump(mz250, 'mz250_mt', norm=norm_mz250)
        dump(mz300, 'mz300_mt', norm=norm_mz300)
        dump(mz350, 'mz350_mt', norm=norm_mz350)

        for threshold in thresholds:
            dump(qcd, 'qcd_mt', threshold, norm=norm_qcd)
            dump(ttjets, 'ttjets_mt', threshold, norm=norm_ttjets)
            dump(bkg, 'bkg_mt', threshold, norm=norm_bkg)
            dump(mz250, 'mz250_mt', threshold, norm=norm_mz250)
            dump(mz300, 'mz300_mt', threshold, norm=norm_mz300)
            dump(mz350, 'mz350_mt', threshold, norm=norm_mz350)

        # For Sara
        tdir = f.mkdir('bsvj')
        tdir.cd()
        sara_threshold = thresholds[quantiles == .8]
        dump(mz250, 'SVJ_mZprime250_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz300, 'SVJ_mZprime300_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz350, 'SVJ_mZprime350_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(bkg, 'Bkg', sara_threshold, False)
        dump(bkg, 'data_obs', sara_threshold, False)

    finally:
        f.Close()





@cli.command()
def make_histograms_mz250_qcd():
    """
    With bdt version trained on only mz250 and qcd
    """
    try_import_ROOT()
    import ROOT

    rootfile = 'test.root'

    qcd_dirs = [
        'postbdt_npzs/Autumn18.QCD_Pt_300to470_TuneCP5_13TeV_pythia8',
        'postbdt_npzs/Autumn18.QCD_Pt_470to600_TuneCP5_13TeV_pythia8',
        'postbdt_npzs/Autumn18.QCD_Pt_600to800_TuneCP5_13TeV_pythia8',
        'postbdt_npzs/Autumn18.QCD_Pt_800to1000_TuneCP5_13TeV_pythia8_ext1',
        'postbdt_npzs/Autumn18.QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8',
        ]
    # qcd_weights = [136.52, 278.51, 150.96, 26.24, 7.49]
    qcd_weights = qcd_presel_eff*qcd_crosssections
    qcd = combine_dirs_with_weights(qcd_dirs, qcd_weights)

    mz250 = np.load('mz250_mdark10_rinv0p3.npz')
    
    # Compute thresholds for every 10% quantile
    thresholds = np.quantile(qcd['score'], [i*.1 for i in range(1,10)])

    try:
        f = ROOT.TFile.Open(rootfile, 'RECREATE')

        def dump(d, name, threshold=None):
            """
            Mini function to write a dictionary to the open root file
            """
            if threshold is not None: name += f'_{threshold:.3f}'
            print(f'Writing {name} --> {rootfile}')
            h = make_mt_histogram(name, d['mt'], d['score'], threshold)
            h.Write()

        dump(qcd, 'qcd_mt')
        dump(mz250, 'mz250_mt')
        for threshold in thresholds:
            dump(qcd, 'qcd_mt', threshold)
            dump(mz250, 'mz250_mt', threshold)

    finally:
        f.Close()



@cli.command()
def process_mz250_locally(model):
    """
    Calculates the BDT scores for mz250 locally and dumps it to a .npz
    """
    rootfiles = seutils.ls_wildcard(
        'root://cmseos.fnal.gov//store/user/lpcdarkqcd/MCSamples_Summer21/TreeMaker'
        '/genjetpt375_mz250_mdark10_rinv0.3/*.root'
        )
    dump_score_npzs_mp(model, rootfiles, 'mz250_mdark10_rinv0p3.npz')


@cli.command()
def process_qcd_locally(model):
    for qcd_dir in seutils.ls_wildcard(
        'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BKG/bkg_May04_year2018/*QCD_Pt*'
        ):
        print(f'Processing {qcd_dir}')
        outfile = osp.basename(qcd_dir + '.npz')
        rootfiles = seutils.ls_wildcard(osp.join(qcd_dir, '*.root'))
        dump_score_npzs_mp(model, rootfiles, outfile)

if __name__ == '__main__':
    cli()
