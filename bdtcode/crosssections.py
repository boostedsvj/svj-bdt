import numpy as np
import os.path as osp

MT_BINNING = [160.+8.*i for i in range(44)]
# MT_BINNING = [8.*i for i in range(130)]

ttjets_xs = {
    'TTJets_SingleLeptFromT_genMET-80'    : 32.23*1.677,
    'TTJets_SingleLeptFromTbar_genMET-80' : 31.78*1.677,
    'TTJets_DiLept_genMET-80'             : 22.46*1.677,
    'TTJets_SingleLeptFromT'              : 182.72,
    'TTJets_SingleLeptFromTbar'           : 182.72,
    'TTJets_DiLept'                       : 88.34,
    'TTJets_HT-600to800'                  : 2.685,
    'TTJets_HT-800to1200'                 : 1.096,
    'TTJets_HT-1200to2500'                : 0.194,
    'TTJets_HT-2500toInf'                 : 0.002,
    'TTJets'                              : 831.76,
    }
qcd_xs = {
    'QCD_Pt_300to470_TuneCP5'   : 6832.0,
    'QCD_Pt_470to600_TuneCP5'   : 552.1,
    'QCD_Pt_600to800_TuneCP5'   : 156.7,
    'QCD_Pt_800to1000_TuneCP5'  : 26.25,
    'QCD_Pt_1000to1400_TuneCP5' : 7.465,
    'QCD_Pt_1400to1800_TuneCP5' : 0.6487,
    'QCD_Pt_1800to2400_TuneCP5' : 0.08734,
    'QCD_Pt_2400to3200_TuneCP5' : 0.005237,
    'QCD_Pt_3200toInf_TuneCP5' : 0.0001352,
    }
wjets_xs = {
    'WJetsToLNu_HT-70To100_TuneCP5'    : 1264.0 * 1.139,
    'WJetsToLNu_HT-100To200_TuneCP5'   : 1393.0,
    'WJetsToLNu_HT-200To400_TuneCP5'   : 409.9,
    'WJetsToLNu_HT-400To600_TuneCP5'   : 57.80,
    'WJetsToLNu_HT-600To800_TuneCP5'   : 12.94,
    'WJetsToLNu_HT-800To1200_TuneCP5'  : 5.451,
    'WJetsToLNu_HT-1200To2500_TuneCP5' : 1.085,
    'WJetsToLNu_HT-2500ToInf_TuneCP5'  : 0.008060,
    }
zjets_xs = {
    'ZJetsToNuNu_HT-100To200'   : 304.0,
    'ZJetsToNuNu_HT-200To400'   : 91.68,
    'ZJetsToNuNu_HT-400To600'   : 13.11,
    'ZJetsToNuNu_HT-600To800'   : 3.245,
    'ZJetsToNuNu_HT-800To1200'  : 1.497,
    'ZJetsToNuNu_HT-1200To2500' : 0.3425,
    'ZJetsToNuNu_HT-2500ToInf'  : 0.005263,
    }

d_mz_xs = np.load(osp.join(osp.dirname(osp.abspath(__file__)) , 'crosssections_Oct12.npz'))
mz_xs = {f'mz{mz:.0f}' : xs for mz, xs in zip(d_mz_xs['mz'], d_mz_xs['xs'])}

from functools import reduce
def merge(d1, d2):
    r = {}
    r.update(d1)
    r.update(d2)
    return r
all_xs = reduce(merge, [ttjets_xs, qcd_xs, wjets_xs, zjets_xs, mz_xs], {})

def label_to_xs(label):
    for key, value in all_xs.items():
        if key in label:
            return value
    raise ValueError(f'No valid xs found for {label}')

def labels_to_xs(labels):
    return np.fromiter((label_to_xs(l) for l in labels), dtype=np.float32)

def genjetpt_eff(mz):
    # For parameters, see:
    # https://github.com/boostedsvj/matching_efficiencies/blob/main/fit.ipynb
    return 3.66e-8*mz**2 + -6.76e-7*mz + 1.12e-5
