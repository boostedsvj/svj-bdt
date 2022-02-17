import glob
import numpy as np
import xgboost as xgb
import seutils
from combine_hists import *
from scripts import combine_dirs_with_weights

try:
    import click
except ImportError:
    print('First install click:\npip install click')
    raise

@click.group()
def cli():
    pass

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

mz_sel_eff = np.array([0.1165, 0.1209, 0.1307, 0.1331, 0.1374]) #0.13847, 0.14312, 0.15442])
mz_crosssections = np.array([2958*0.456*0.05, 1671*0.456*0.05, 966.8*0.456*0.05, 578.3*0.456*0.05, 383.6*0.456*0.05])


@cli.command()
@click.option('-o', '--rootfile', default='mt_Feb07_bdtcut0p3.root')
def make_histograms_3masspoints_qcd_ttjets(rootfile):
    """
    With bdt version trained on only mz250 and qcd
    """
    try_import_ROOT()
    import ROOT
    print(f'qcd jets')
    qcd_dirs = [
        'postbdt/Autumn18.QCD_Pt_300to470_TuneCP5_13TeV_pythia8',
        'postbdt/Autumn18.QCD_Pt_470to600_TuneCP5_13TeV_pythia8',
        'postbdt/Autumn18.QCD_Pt_600to800_TuneCP5_13TeV_pythia8',
        'postbdt/Autumn18.QCD_Pt_800to1000_TuneCP5_13TeV_pythia8_ext1',
        'postbdt/Autumn18.QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8',
        ]
    qcd_weights = qcd_crossections*qcd_sel_eff
    print(f'ttjets')
    ttjets_dirs = [
        'postbdt/Autumn18.TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt/Autumn18.TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt/Autumn18.TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt/Autumn18.TTJets_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt/Autumn18.TTJets_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt/Autumn18.TTJets_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt/Autumn18.TTJets_HT-1200to2500_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt/Autumn18.TTJets_HT-2500toInf_TuneCP5_13TeV-madgraphMLM-pythia8',
        ]
    ttjets_weights = ttjets_crosssections*ttjets_sel_eff
    print(f'wjets')
    wjets_dirs = [
        'postbdt/Autumn18.WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt/Autumn18.WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt/Autumn18.WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt/Autumn18.WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt/Autumn18.WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8',
        ]
    wjets_weights = wjets_crosssections*wjets_sel_eff
    print(f'zjets')
    zjets_dirs = [
        'postbdt/Autumn18.ZJetsToNuNu_HT-400To600_13TeV-madgraph',
        'postbdt/Autumn18.ZJetsToNuNu_HT-600To800_13TeV-madgraph',
        'postbdt/Autumn18.ZJetsToNuNu_HT-800To1200_13TeV-madgraph',
        'postbdt/Autumn18.ZJetsToNuNu_HT-1200To2500_13TeV-madgraph',
        'postbdt/Autumn18.ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph',
        ]
    zjets_weights = zjets_crosssections*zjets_sel_eff

    qcd = combine_dirs_with_weights(qcd_dirs, qcd_weights)
    ttjets = combine_dirs_with_weights(ttjets_dirs, ttjets_weights)
    wjets  = combine_dirs_with_weights(wjets_dirs, wjets_weights)
    zjets  = combine_dirs_with_weights(zjets_dirs, zjets_weights)

    bkg = combine_dirs_with_weights(qcd_dirs+ttjets_dirs+zjets_dirs+wjets_dirs, np.concatenate((qcd_weights, ttjets_weights, wjets_weights, zjets_weights)))
    np.savez("bkg.npz", bkg)

    mz250_dirs = ['postbdt/genjetpt375_mz250_mdark10_rinv0.3'] 
    mz300_dirs = ['postbdt/genjetpt375_mz300_mdark10_rinv0.3']
    mz350_dirs = ['postbdt/genjetpt375_mz350_mdark10_rinv0.3']
    mz400_dirs = ['postbdt/genjetpt375_mz400_mdark10_rinv0.3']
    mz450_dirs = ['postbdt/genjetpt375_mz450_mdark10_rinv0.3']
    mz500_dirs = ['postbdt/genjetpt375_mz500_mdark10_rinv0.3']
    mz550_dirs = ['postbdt/genjetpt375_mz550_mdark10_rinv0.3']


    


    #matching_efficiencies = [0.4527400, 0.4511200, 0.4452200, 0.4400800, 0.4277200, 0.4287800, 0.4287800] #not used anymore
    genjetpt_efficiencies = [0.0018112, 0.0029704, 0.0039980, 0.0053627, 0.0073413, 0.0083959, 0.0083959]
    preselection_efficiencies = np.array([0.1049, 0.1096, 0.1173, 0.1195, 0.1269, 0.1243, 0.1247])
    xsec = np.array([2958.0, 1671.0, 966.8, 578.3, 383.6, 268.0, 193.6])

    mz250_weight = np.array([0.562004855]) 
    mz300_weight = np.array([0.544003809]) 
    mz350_weight = np.array([0.453395749]) 
    mz400_weight = np.array([0.370599304]) 
    mz450_weight = np.array([0.357365968]) 
    mz500_weight = np.array([0.279687579]) 
    mz550_weight = np.array([0.202693146])

    mz250 = combine_dirs_with_weights(mz250_dirs, mz250_weight)
    mz300 = combine_dirs_with_weights(mz300_dirs, mz300_weight)
    mz350 = combine_dirs_with_weights(mz350_dirs, mz350_weight)
    mz400 = combine_dirs_with_weights(mz400_dirs, mz400_weight)
    mz450 = combine_dirs_with_weights(mz450_dirs, mz450_weight)
    mz500 = combine_dirs_with_weights(mz500_dirs, mz500_weight)
    mz550 = combine_dirs_with_weights(mz550_dirs, mz550_weight)

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
            h.Write()

        tdir = f.mkdir('bsvj_0p0')
        tdir.cd()
        sara_threshold = 0.0
        dump(mz250, 'SVJ_mZprime250_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz300, 'SVJ_mZprime300_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz350, 'SVJ_mZprime350_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz400, 'SVJ_mZprime400_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz450, 'SVJ_mZprime450_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz500, 'SVJ_mZprime500_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz550, 'SVJ_mZprime550_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(qcd, 'QCD', sara_threshold, False)
        dump(ttjets, 'TTJets', sara_threshold, False)
        dump(wjets, 'WJets', sara_threshold, False)
        dump(zjets, 'ZJets', sara_threshold, False)
        dump(bkg, 'Bkg', sara_threshold, False)
        dump(bkg, 'data_obs', sara_threshold, False)

        tdir = f.mkdir('bsvj_0p1')
        tdir.cd()
        sara_threshold = 0.1
        dump(mz250, 'SVJ_mZprime250_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz300, 'SVJ_mZprime300_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz350, 'SVJ_mZprime350_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz400, 'SVJ_mZprime400_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz450, 'SVJ_mZprime450_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz500, 'SVJ_mZprime500_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz550, 'SVJ_mZprime550_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(qcd, 'QCD', sara_threshold, False)
        dump(ttjets, 'TTJets', sara_threshold, False)
        dump(wjets, 'WJets', sara_threshold, False)
        dump(zjets, 'ZJets', sara_threshold, False)
        dump(bkg, 'Bkg', sara_threshold, False)
        dump(bkg, 'data_obs', sara_threshold, False)

        tdir = f.mkdir('bsvj_0p2')
        tdir.cd()
        sara_threshold = 0.2
        dump(mz250, 'SVJ_mZprime250_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz300, 'SVJ_mZprime300_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz350, 'SVJ_mZprime350_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz400, 'SVJ_mZprime400_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz450, 'SVJ_mZprime450_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz500, 'SVJ_mZprime500_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz550, 'SVJ_mZprime550_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(qcd, 'QCD', sara_threshold, False)
        dump(ttjets, 'TTJets', sara_threshold, False)
        dump(wjets, 'WJets', sara_threshold, False)
        dump(zjets, 'ZJets', sara_threshold, False)
        dump(bkg, 'Bkg', sara_threshold, False)
        dump(bkg, 'data_obs', sara_threshold, False)

        tdir = f.mkdir('bsvj_0p3')
        tdir.cd()
        sara_threshold = 0.3
        dump(mz250, 'SVJ_mZprime250_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz300, 'SVJ_mZprime300_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz350, 'SVJ_mZprime350_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz400, 'SVJ_mZprime400_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz450, 'SVJ_mZprime450_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz500, 'SVJ_mZprime500_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz550, 'SVJ_mZprime550_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(qcd, 'QCD', sara_threshold, False)
        dump(ttjets, 'TTJets', sara_threshold, False)
        dump(wjets, 'WJets', sara_threshold, False)
        dump(zjets, 'ZJets', sara_threshold, False)
        dump(bkg, 'Bkg', sara_threshold, False)
        dump(bkg, 'data_obs', sara_threshold, False)

        tdir = f.mkdir('bsvj_0p4')
        tdir.cd()
        sara_threshold = 0.4
        dump(mz250, 'SVJ_mZprime250_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz300, 'SVJ_mZprime300_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz350, 'SVJ_mZprime350_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz400, 'SVJ_mZprime400_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz450, 'SVJ_mZprime450_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz500, 'SVJ_mZprime500_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz550, 'SVJ_mZprime550_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(qcd, 'QCD', sara_threshold, False)
        dump(ttjets, 'TTJets', sara_threshold, False)
        dump(wjets, 'WJets', sara_threshold, False)
        dump(zjets, 'ZJets', sara_threshold, False)
        dump(bkg, 'Bkg', sara_threshold, False)
        dump(bkg, 'data_obs', sara_threshold, False)


        tdir = f.mkdir('bsvj_0p5')
        tdir.cd()
        sara_threshold = 0.5
        dump(mz250, 'SVJ_mZprime250_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz300, 'SVJ_mZprime300_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz350, 'SVJ_mZprime350_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz400, 'SVJ_mZprime400_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz450, 'SVJ_mZprime450_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz500, 'SVJ_mZprime500_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz550, 'SVJ_mZprime550_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(qcd, 'QCD', sara_threshold, False)
        dump(ttjets, 'TTJets', sara_threshold, False)
        dump(wjets, 'WJets', sara_threshold, False)
        dump(zjets, 'ZJets', sara_threshold, False)
        dump(bkg, 'Bkg', sara_threshold, False)
        dump(bkg, 'data_obs', sara_threshold, False)


        tdir = f.mkdir('bsvj_0p6')
        tdir.cd()
        sara_threshold = 0.6
        dump(mz250, 'SVJ_mZprime250_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz300, 'SVJ_mZprime300_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz350, 'SVJ_mZprime350_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz400, 'SVJ_mZprime400_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz450, 'SVJ_mZprime450_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz500, 'SVJ_mZprime500_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz550, 'SVJ_mZprime550_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(qcd, 'QCD', sara_threshold, False)
        dump(ttjets, 'TTJets', sara_threshold, False)
        dump(wjets, 'WJets', sara_threshold, False)
        dump(zjets, 'ZJets', sara_threshold, False)
        dump(bkg, 'Bkg', sara_threshold, False)
        dump(bkg, 'data_obs', sara_threshold, False)

    finally:
        f.Close() 
