import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#import glob

score_rew = np.load('npzfiles/ptreweight_score.npz')
score_now = np.load('npzfiles/noweight_score.npz')

all_var_reweight = np.load('npzfiles/ptreweight_allvariables.npz')
all_var_noweight = np.load('npzfiles/noweight_allvarialbles.npz')

# all variables order: 
#  X.append([subl.ptD, subl.axismajor, subl.multiplicity,subl.girth, subl.axisminor, subl.metdphi,subl.ecfM2b1, subl.ecfD2b1, subl.ecfC2b1, subl.ecfN2b2, ak4partFlav.partonFlovor,subl.pt, subl.eta, subl.phi, subl.energy, subl.rt, subl.mt])
mt_reweight = all_var_reweight['arr_0'][:,16]
pt_reweight = all_var_reweight['arr_0'][:,11]
ecfd2b1_reweight = all_var_reweight['arr_0'][:,7]
girth_reweight = all_var_reweight['arr_0'][:,3]
score_reweight = score_rew['arr_0']

mt_noweight = all_var_noweight['arr_0'][:,16]
pt_noweight = all_var_noweight['arr_0'][:,11]
ecfd2b1_noweight = all_var_noweight['arr_0'][:,7]
girth_noweight = all_var_noweight['arr_0'][:,3]
multiplicity_noweight = all_var_noweight['arr_0'][:,2]
ptD_noweight = all_var_noweight['arr_0'][:,0]
score_noweight = score_now['arr_0']

high1_score = score_noweight > 0.5
high2_score = score_noweight < 0.7
low1_score = score_noweight < 0.01

plt.figure('multiplicity_noweight')
plt.hist(ptD_noweight[high1_score & high2_score], bins=40)
plt.hist(ptD_noweight[low1_score], bins=40, alpha=0.4)
plt.xlabel('multiplicity')
plt.grid(True)
plt.title('QCD multiplicity: lower & higher bdt peak')
plt.show()
plt.savefig('pngs/rewvsnow/pngs_nw/multiplicity_lowhighpeak.pdf')

'''plt.figure('scorevsmt_noweight')
plt.hist2d(mt_noweight, score_noweight, bins=40, norm=LogNorm())
plt.xlabel('$M_T$ (GeV)')
plt.ylabel('score')
plt.grid(True)
plt.title('QCD Score vs $M_T$ no weight')
plt.colorbar()
plt.savefig('pngs/rewvsnow/pngs_nw/scorevsmt.pdf')

plt.figure('scorevsmt_reweight')
plt.hist2d(mt_reweight, score_reweight, bins=40, norm=LogNorm())
plt.xlabel('$M_T$ (GeV)')
plt.ylabel('score')
plt.grid(True)
plt.title('QCD Score vs $M_T$ reweighted')
plt.colorbar()
plt.savefig('pngs/rewvsnow/pngs_rw/scorevsmt.pdf')

plt.figure('ecfd2b1vsmt_noweight')
plt.hist2d(mt_noweight, ecfd2b1_noweight, bins=40, norm=LogNorm())
plt.xlabel('$M_T$ (GeV)')
plt.ylabel('ecf$D_2b_1$')
plt.grid(True)
plt.title('QCD ecf$D_2b_1$ vs $M_T$ no weight')
plt.colorbar()
plt.savefig('pngs/rewvsnow/pngs_nw/ecfd2b1vsmt.pdf')

plt.figure('ecfd2b1vsmt_reweight')
plt.hist2d(mt_reweight, ecfd2b1_reweight, bins=40, norm=LogNorm())
plt.xlabel('$M_T$ (GeV)')
plt.ylabel('ecf$D_2b_1$')
plt.grid(True)
plt.title('QCD ecf$D_2b_1$ vs $M_T$ reweighted')
plt.colorbar()
plt.savefig('pngs/rewvsnow/pngs_rw/ecfd2b1vsmt.pdf')

plt.figure('girthvsmt_noweight')
plt.hist2d(mt_noweight, girth_noweight, bins=40, norm=LogNorm())
plt.xlabel('$M_T$ (GeV)')
plt.ylabel('girth')
plt.grid(True)
plt.title('QCD girth vs $M_T$ no weight')
plt.colorbar()
plt.savefig('pngs/rewvsnow/pngs_nw/girthvsmt.pdf')

plt.figure('girthvsmt_reweight')
plt.hist2d(mt_reweight, girth_reweight, bins=40, norm=LogNorm())
plt.xlabel('$M_T$ (GeV)')
plt.ylabel('girth')
plt.grid(True)
plt.title('QCD girth vs $M_T$ reweighted')
plt.colorbar()
plt.savefig('pngs/rewvsnow/pngs_rw/girthvsmt.pdf')

plt.figure('MT')
plt.hist(mt_reweight[score_reweight<0.00005],bins=40, density=True, label=['score<0.1'])
plt.hist(mt_reweight[score_reweight<0.01],bins=40, density=True, label=['score<0.01'],alpha=0.7)
plt.hist(mt_reweight[score_reweight<0.0001],bins=40, density=True, label=['score<0.0001'],alpha=0.5)
plt.hist(mt_reweight[score_reweight<0.00005],bins=40, density=True, label=['score<0.00005'],alpha=0.3)
plt.legend()
plt.xlabel('$M_T$ (GeV)')
plt.grid(True)
plt.title('QCD $M_T$ reweighted')
plt.savefig('pngs/rewvsnow/mt.pdf')

plt.figure('Scores')
plt.hist(score_reweight,bins=40, density=True, label=['pt re-weight'])
plt.hist(score_noweight,bins=40, density=True, label=['no re-weight'],alpha=0.3)
plt.legend()
plt.xlabel('$BDT_{score}$')
plt.grid(True)
plt.yscale('log')
plt.title('QCD $BDT_{score}$ re-weight & no-weight')
plt.savefig('pngs/rewvsnow/scores.pdf')'''
