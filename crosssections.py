import numpy as np

ttjets_xs = {
    'TTJets_SingleLeptFromT'              : 831.76 * 0.219,
    'TTJets_SingleLeptFromTbar'           : 831.76 * 0.219,
    'TTJets_DiLept'                       : 831.76 * 0.105,
    'TTJets_SingleLeptFromT_genMET-80'    : 32.23,
    'TTJets_SingleLeptFromTbar_genMET-80' : 31.78,
    'TTJets_DiLept_genMET-80'             : 22.46,
    'TTJets_HT-600to800'                  : 1.808,
    'TTJets_HT-800to1200'                 : 0.7490,
    'TTJets_HT-1200to2500'                : 0.1315,
    'TTJets_HT-2500toInf'                 : 0.001420,
    'TTJets'                              : 831.76,
    }
qcd_xs = {
    'QCD_Pt_300to470_TuneCP5'   : 6826.0,
    'QCD_Pt_470to600_TuneCP5'   : 552.6,
    'QCD_Pt_600to800_TuneCP5'   : 156.6,
    'QCD_Pt_800to1000_TuneCP5'  : 26.3,
    'QCD_Pt_1000to1400_TuneCP5' : 7.5,
    }
wjets_xs = {
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

raw_mz_xs = {
    'mz1000' : 2.273e+01,
    'mz150'  : 1.374e+04,
    'mz250'  : 2.925e+03,
    'mz300'  : 1.657e+03,
    'mz450'  : 3.794e+02,
    'mz500'  : 2.655e+02,
    'mz600'  : 1.425e+02,
    'mz650'  : 1.082e+02,
    'mz700'  : 8.365e+01,
    'mz800'  : 5.151e+01,
    'mz900'  : 3.379e+01,
    'mz50'   : 2.754e+05,
    }

# FIXME: Inter- and extrapolations
raw_mz_xs['mz350'] = raw_mz_xs['mz300'] - (raw_mz_xs['mz250']-raw_mz_xs['mz300'])
raw_mz_xs['mz400'] = raw_mz_xs['mz450'] + (raw_mz_xs['mz300']-raw_mz_xs['mz450'])/3.
raw_mz_xs['mz230'] = raw_mz_xs['mz250'] + (raw_mz_xs['mz150']-raw_mz_xs['mz250'])/5.
raw_mz_xs['mz270'] = raw_mz_xs['mz250'] - (raw_mz_xs['mz150']-raw_mz_xs['mz250'])/5.
raw_mz_xs['mz330'] = raw_mz_xs['mz300'] - (raw_mz_xs['mz300']-raw_mz_xs['mz450'])/5.

# FIXME: Apply the right genjetpt375 efficiencies per mass point
# Now applying the mz250 efficiencies on all! This is a very bad approximation!
# Need to recalculate!
mz_xs = { k : 0.00191*0.233*v for k, v in raw_mz_xs.items() }

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
