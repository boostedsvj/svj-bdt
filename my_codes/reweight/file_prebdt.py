#extra packages
import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
import xgboost as x
from xgb import get_bkg_features
from all_variables import get_allbkg_features
from reweight import get_reweighted_bkg_features
from reweight_allvariables import get_reweighted_allbkg_features
import glob


qcd_labels = ['QCD_Pt_300to470', 'QCD_Pt_470to600', 'QCD_Pt_600to800', 'QCD_Pt_800to1000', 'QCD_Pt_1000to1400']
qcd_weights = [6826.0*0.0023, 552.6*0.0442, 156.6*0.0656, 26.32*0.0516, 7.50*0.0438]
#sets_of_npzs = [ glob.iglob(f'/home/snabili/data/bdt/svj-bdt/my_codes/reweight/prebdt_Nov30_rtx108/*{l}*/*.npz') for l in qcd_labels]
#sets_of_npzs = [ glob.iglob(f'/home/snabili/data/bdt/svj-bdt/my_codes/parton_flavor/prebdt_Nov15/*{l}*/*.npz') for l in qcd_labels]
sets_of_npzs = [ glob.iglob(f'/home/snabili/data/bdt/svj-bdt/my_codes/reweight/prebdt_Dec10_RTx11_offsetconstitute/*{l}*/*.npz') for l in qcd_labels]

sig_label = ['mz400']
benchmark_weight = [0.0053627*578.3]
sig_benchmark = [ glob.iglob(f'/home/snabili/data/bdt/svj-bdt/my_codes/reweight/prebdt_Nov30_rtx108/*{l}*/*.npz') for l in sig_label]

# all variables order: 
#  X.append([subl.ptD, subl.axismajor, subl.multiplicity,subl.girth, subl.axisminor, subl.metdphi,subl.ecfM2b1, subl.ecfD2b1, subl.ecfC2b1, subl.ecfN2b2, ak4partFlav.partonFlovor,subl.pt, subl.eta, subl.phi, subl.energy, subl.rt, subl.mt, met, subl.smd])
X_bkg = get_bkg_features(sets_of_npzs, qcd_weights) #score
#X_bkg_all = get_allbkg_features(sets_of_npzs, qcd_weights) #all variables
#X_bkgreweighted = get_reweighted_bkg_features(sets_of_npzs, qcd_weights, sig_benchmark, benchmark_weight)#score
#X_allbkgreweighted = get_reweighted_allbkg_features(sets_of_npzs, qcd_weights, sig_benchmark, benchmark_weight) #all variables


#no reweight score
#model_noreweight = x.XGBClassifier()
#model_noreweight.load_model('svjbdt_Nov15.json') #no reweight
#model_noreweight.load_model('svjbdt_Dec02.json') #reweight
#score_noreweight = model_noreweight.predict_proba(np.array(X_bkg))[:,1]

#pt reweight score
model_ptreweight = x.XGBClassifier()
#model_ptreweight.load_model('svjbdt_Nov22.json') #pt reweight
model_ptreweight.load_model('/home/snabili/data/bdt/svj-bdt/my_codes/reweight/weight_study/pt_multi/rtx11_sigweight_svjbdt_ptreweight_Dec10.json') #pt reweight
score_ptreweight = model_ptreweight.predict_proba(np.array(X_bkg))[:,1]

# noweight score and allvariables
# all variables order: 
#  X.append([subl.ptD, subl.axismajor, subl.multiplicity,subl.girth, subl.axisminor, subl.metdphi,subl.ecfM2b1, subl.ecfD2b1, subl.ecfC2b1, subl.ecfN2b2, ak4partFlav.partonFlovor,subl.pt, subl.eta, subl.phi, subl.energy, subl.rt, subl.mt, subl.met])
np.savez('npzfiles/Dec10/ptweight_score_new.npz', score_ptreweight)
#np.savez('npzfiles/Dec10/ptweight_allvarialbles.npz', X_bkg_all[:,:])

# pt re-weighted files
#np.savez('npzfiles/Nov29/ptreweight_score.npz', score_ptreweight)
#np.savez('npzfiles/Nov29/ptreweight_allvariables.npz', X_allbkgreweighted[:,:])


'''
#check signal benchmark variables
lab = ['mz400']
sig_benchmark = [glob.iglob(f'/home/snabili/data/bdt/svj-bdt/my_codes/parton_flavor/prebdt_Nov15/*{l}*/*.npz') for l in lab]
benchmark_weight = [1]
X_signal = get_allbkg_features(sig_benchmark, benchmark_weight)
np.savez('signal.npz', X_signal)

# check one qcd bin score
lab1 = ['QCD_Pt_300to470']
qcd1 = [glob.iglob(f'/home/snabili/data/bdt/svj-bdt/my_codes/parton_flavor/prebdt_Nov15/*{l}*/*.npz') for l in lab1]
qcd1_weight = [6826.0*0.0023]
X_qcd1 = get_allbkg_features(qcd1, qcd1_weight)
np.savez('qcd1.npz', X_qcd1)
'''
