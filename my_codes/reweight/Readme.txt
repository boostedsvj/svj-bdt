file_prebdt.py is to make the npz files 
it runs over the prebdt_npz files stored at umd in: /home/snabili/hadoop/NPZFiles/NPZFiles_QCD_PartonFlavor_bkg_*_year2018

the output of running file_prebdt.py code will be four npz files:
noweight_score.npz --> the bdt_score without any pt_weight applied
ptreweight_score.npz --> bdt_score with pt-reweight applied
noweight_allvarialbles.npz --> all varialbles weighted with only xsec
reweight_allvarialbles.npz --> all variables weighted with xsec and pt-reweight

running sculpt.py script to make diagnostic plots and compare the effect of pt-reweighting 
