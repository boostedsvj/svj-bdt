import glob
import numpy as np
import xgboost as xgb
import seutils
from combine_hists import *

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


# Some weights

qcd_sel_eff = np.array([0.0023, 0.0442, 0.0656, 0.0516, 0.0438])
qcd_crossections = np.array([6826.0, 552.6, 156.6, 26.3, 7.5])

#ttjets orders:
'''DiLept_TuneCP5
SingleLeptFromTbar_TuneCP5
SingleLeptFromT_TuneCP5
TuneCP5
HT-600to800_TuneCP5
HT-800to1200_TuneCP5
HT-1200to2500_TuneCP5
HT-2500toInf_TuneCP5'''
ttjets_sel_eff = np.array([0.0004, 0.0014, 0.0012, 0.001, 0.0569, 0.1066, 0.1255, 0.0871])
ttjets_crosssections = np.array([87.339, 182.1642, 182.1642, 831.76, 1.808, 0.749, 0.22, 0.002])

wjets_sel_eff = np.array([0.000216, 0.006872, 0.061298, 0.163376, 0.102858])
wjets_crosssections = np.array([57.80*1.164, 12.94*1.164, 5.451*1.164, 1.085*1.164, 0.008060*1.164])

zjets_sel_eff = np.array([0.00009, 0.01089, 0.18441, 0.51673, 0.36668])
zjets_crosssections = np.array([13.11*1.1421, 3.245*1.1421, 1.497*1.1421, 0.3425*1.1421, 0.005263*1.1421])


# FIXME: mz350 cross section extrapolated now
mz_sel_eff = np.array([0.1165, 0.1209, 0.1307, 0.1331, 0.1374]) #0.13847, 0.14312, 0.15442])
mz_crosssections = np.array([2958*0.456*0.05, 1671*0.456*0.05, 966.8*0.456*0.05, 578.3*0.456*0.05, 383.6*0.456*0.05])


@cli.command()
def preselection_efficiencies_bkg():
    directories = list(sorted(glob.iglob('postbdt_npzs_Oct21_5mass_qcdttjetswjetszjets/*')))
    flength = max(len(osp.basename(d)) for d in directories)
    for directory in directories:
        npzs = glob.glob(osp.join(directory, '*.npz'))
        d = combine_npzs(npzs)
        print(
            f'{osp.basename(directory):{flength}s} :'
            f' n_total={d["total"]:9}, n_presel={d["preselection"]:9},'
            f' frac={d["preselection"]/d["total"]:.5f}'
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
    # qcd = combine_ds_with_weights(qcd_ds, [136.52, 278.51, 150.96, 26.24, 7.49])
    qcd = combine_ds_with_weights(qcd_ds, qcd_sel_eff*qcd_crossections)
    return qcd


@cli.command()
@click.option('-o', '--outputfile', default='bkg.npz')
def get_combined_bkg(outputfile):
    print('Reading individual bkg .npzs')
    bkg_npzs = [
        glob.glob(osp.join(directory, '*.npz'))
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
    bkg = combine_ds_with_weights(bkg_ds, np.concatenate((qcd_sel_eff*qcd_crossections, ttjets_sel_eff*ttjets_crosssections)))
    np.savez(outputfile, bkg)
    return bkg



@cli.command()
@click.option('-o', '--rootfile', default='test_Nov9.root')
def make_histograms_3masspoints_qcd_ttjets(rootfile):
    """
    With bdt version trained on only mz250 and qcd
    """
    try_import_ROOT()
    import ROOT
    print(f'qcd jets')
    qcd_dirs = [
        'postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/Autumn18.QCD_Pt_300to470_TuneCP5_13TeV_pythia8',
        'postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/Autumn18.QCD_Pt_470to600_TuneCP5_13TeV_pythia8',
        'postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/Autumn18.QCD_Pt_600to800_TuneCP5_13TeV_pythia8',
        'postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/Autumn18.QCD_Pt_800to1000_TuneCP5_13TeV_pythia8_ext1',
        'postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/Autumn18.QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8',
        ]
    qcd_weights = qcd_crossections*qcd_sel_eff
    print(f'ttjets')
    ttjets_dirs = [
        'postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/Autumn18.TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/Autumn18.TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/Autumn18.TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/Autumn18.TTJets_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/Autumn18.TTJets_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/Autumn18.TTJets_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/Autumn18.TTJets_HT-1200to2500_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/Autumn18.TTJets_HT-2500toInf_TuneCP5_13TeV-madgraphMLM-pythia8',
        ]
    ttjets_weights = ttjets_crosssections*ttjets_sel_eff
    print(f'wjets')
    wjets_dirs = [
        'postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/Autumn18.WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/Autumn18.WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/Autumn18.WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/Autumn18.WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/Autumn18.WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8',
        ]
    wjets_weights = wjets_crosssections*wjets_sel_eff
    print(f'zjets')
    zjets_dirs = [
        'postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/Autumn18.ZJetsToNuNu_HT-400To600_13TeV-madgraph',
        'postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/Autumn18.ZJetsToNuNu_HT-600To800_13TeV-madgraph',
        'postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/Autumn18.ZJetsToNuNu_HT-800To1200_13TeV-madgraph',
        'postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/Autumn18.ZJetsToNuNu_HT-1200To2500_13TeV-madgraph',
        'postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/Autumn18.ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph',
        ]
    zjets_weights = zjets_crosssections*zjets_sel_eff

    qcd = combine_dirs_with_weights(qcd_dirs, qcd_weights)
    ttjets = combine_dirs_with_weights(ttjets_dirs, ttjets_weights)
    wjets  = combine_dirs_with_weights(wjets_dirs, wjets_weights)
    zjets  = combine_dirs_with_weights(zjets_dirs, zjets_weights)
    bkg = combine_dirs_with_weights(qcd_dirs+ttjets_dirs+zjets_dirs+wjets_dirs, np.concatenate((qcd_weights, ttjets_weights, wjets_weights, zjets_weights)))

    mz250 = combine_npzs(glob.glob('postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/genjetpt375_mz250_mdark10_rinv0.3/*.npz'))
    mz300 = combine_npzs(glob.glob('postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/genjetpt375_mz300_mdark10_rinv0.3/*.npz'))
    mz350 = combine_npzs(glob.glob('postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/genjetpt375_mz350_mdark10_rinv0.3/*.npz'))
    mz400 = combine_npzs(glob.glob('postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/genjetpt375_mz400_mdark10_rinv0.3/*.npz'))
    mz450 = combine_npzs(glob.glob('postbdt_npzs_Nov9_5masspoints_qcdttjetswjetszjets/genjetpt375_mz450_mdark10_rinv0.3/*.npz'))


    # Compute thresholds for every 10% quantile
    quantiles = np.array([i*.1 for i in range(1,10)])
    naming = np.array([i for i in range(1,10)])
    thresholds = np.quantile(bkg['score'], quantiles)

    try:
        f = ROOT.TFile.Open(rootfile, 'RECREATE')

        def dump(d, name, threshold=None, use_threshold_in_name=True):
            """
            Mini function to write a dictionary to the open root file
            """
            if threshold is not None and use_threshold_in_name:
                name += f'_{int(naming[thresholds == threshold]):d}'
            print(f'Writing {name} --> {rootfile}')
            h = make_mt_histogram(name, d['mt'], d['score'], threshold)
            #h2d = make_rtvsmt_histogram(name, d['rt'], d['mt'], d['score'], threshold)
            #h2d.Write()
            h.Write()

        dump(qcd, 'qcd_mt')
        dump(ttjets, 'ttjets_mt')
        dump(wjets, 'wjets_mt')
        dump(zjets, 'zjets_mt')
        dump(bkg, 'bkg_mt')
        dump(mz250, 'mz250_mt')
        dump(mz300, 'mz300_mt')
        dump(mz350, 'mz350_mt')
        dump(mz400, 'mz400_mt')
        dump(mz450, 'mz450_mt')
	#dump(bkg, 'bkg_rtvsmt')

        for threshold in thresholds:
            dump(qcd, 'qcd_mt', threshold)
            dump(ttjets, 'ttjets_mt', threshold)
            dump(wjets, 'wjets_mt', threshold)
            dump(zjets, 'zjets_mt', threshold)
            dump(bkg, 'bkg_mt', threshold)
            dump(mz250, 'mz250_mt', threshold)
            dump(mz300, 'mz300_mt', threshold)
            dump(mz350, 'mz350_mt', threshold)
            dump(mz400, 'mz400_mt', threshold)
            dump(mz450, 'mz450_mt', threshold)
              

        # For Sara
        tdir = f.mkdir('bsvj')
        tdir.cd()
        #sara_threshold = thresholds[quantiles == .4]
        #sara_threshold = thresholds[quantiles == .5]
        #print(sara_threshold)
        print(thresholds[quantiles == .4])
        print(thresholds[quantiles == .6])
        sara_threshold = 0.01963109 # this is equivalent to quantile==0.6
        #wz_threshold = 0
        dump(mz250, 'SVJ_mZprime250_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz300, 'SVJ_mZprime300_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz350, 'SVJ_mZprime350_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz400, 'SVJ_mZprime400_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz450, 'SVJ_mZprime450_mDark10_rinv03_alphapeak', sara_threshold, False)
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
    qcd_weights = qcd_presel_eff*qcd_crossections
    qcd = combine_dirs_with_weights(qcd_dirs, qcd_weights)

    mz250 = np.load('mz250_mdark10_rinv0p3.npz', allow_pickle=True)
    
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
    get_hist_mp(model, rootfiles, 'mz250_mdark10_rinv0p3.npz')


@cli.command()
def process_qcd_locally(model):
    for qcd_dir in seutils.ls_wildcard(
        'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BKG/bkg_May04_year2018/*QCD_Pt*'
        ):
        print(f'Processing {qcd_dir}')
        outfile = osp.basename(qcd_dir + '.npz')
        rootfiles = seutils.ls_wildcard(osp.join(qcd_dir, '*.root'))
        get_hist_mp(model, rootfiles, outfile)



if __name__ == '__main__':
    cli()
