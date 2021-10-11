import ROOT
import math
from ROOT import TFile, TTree, TH1F
from array import array
from math import *
import os.path as osp, os


#TTJest
ttjets_dilept_genMET = [line.rstrip('\n') for line in open('TXT/TTJets_DiLept_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8.txt')]
ttjets_dilept        = [line.rstrip('\n') for line in open('TXT/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8.txt')]
ttjets_dilept_sinleptbar_genMET = [line.rstrip('\n') for line in open('TXT/TTJets_SingleLeptFromTbar_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8.txt')]
ttjets_dilept_sinleptbar = [line.rstrip('\n') for line in open('TXT/TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8.txt')]
ttjets_dilept_sinlept_genMET = [line.rstrip('\n') for line in open('TXT/TTJets_SingleLeptFromT_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8.txt')]
ttjets_dilept_sinlept = [line.rstrip('\n') for line in open('TXT/TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8.txt')]
ttjets_tunecp5 = [line.rstrip('\n') for line in open('TXT/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8.txt')]
ttjets_dilept_600to800 = [line.rstrip('\n') for line in open('TXT/TTJets_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8.txt')]
ttjets_dilept_800to1200 = [line.rstrip('\n') for line in open('TXT/TTJets_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8.txt')]
ttjets_dilept_1200to2500 = [line.rstrip('\n') for line in open('TXT/TTJets_HT-1200to2500_TuneCP5_13TeV-madgraphMLM-pythia8.txt')]
ttjets_dilept_2500toInf = [line.rstrip('\n') for line in open('TXT/TTJets_HT-2500toInf_TuneCP5_13TeV-madgraphMLM-pythia8.txt')]

#WJets:
wjets_100to200 = [line.rstrip('\n') for line in open('TXT/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8.txt')]
wjets_200to400 = [line.rstrip('\n') for line in open('TXT/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8.txt')]
wjets_400to600 = [line.rstrip('\n') for line in open('TXT/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8.txt')]
wjets_600to800  = [line.rstrip('\n') for line in open('TXT/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8.txt')]
wjets_800to1200  = [line.rstrip('\n') for line in open('TXT/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8.txt')]
wjets_1200to2500  = [line.rstrip('\n') for line in open('TXT/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8.txt')]
wjets_2500toInf  = [line.rstrip('\n') for line in open('TXT/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8.txt')]

#ZJets:
zjets_100to200 = [line.rstrip('\n') for line in open('TXT/ZJetsToNuNu_HT-100To200_13TeV-madgraph.txt')]
zjets_200to400 = [line.rstrip('\n') for line in open('TXT/ZJetsToNuNu_HT-200To400_13TeV-madgraph.txt')]
zjets_400to600 = [line.rstrip('\n') for line in open('TXT/ZJetsToNuNu_HT-400To600_13TeV-madgraph.txt')]
zjets_600to800 = [line.rstrip('\n') for line in open('TXT/ZJetsToNuNu_HT-600To800_13TeV-madgraph.txt')]
zjets_800to1200 = [line.rstrip('\n') for line in open('TXT/ZJetsToNuNu_HT-800To1200_13TeV-madgraph.txt')]
zjets_1200to2500 = [line.rstrip('\n') for line in open('TXT/ZJetsToNuNu_HT-1200To2500_13TeV-madgraph.txt')]
zjets_2500toInf = [line.rstrip('\n') for line in open('TXT/ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph.txt')]


#QCD_Pt:
qcd_300to470 = [line.rstrip('\n') for line in open('TXT/Autumn18.QCD_Pt_300to470_TuneCP5_13TeV_pythia8.txt')]
qcd_470to600 = [line.rstrip('\n') for line in open('TXT/Autumn18.QCD_Pt_470to600_TuneCP5_13TeV_pythia8.txt')]
qcd_600to800 = [line.rstrip('\n') for line in open('TXT/Autumn18.QCD_Pt_600to800_TuneCP5_13TeV_pythia8.txt')]
qcd_800to1000 = [line.rstrip('\n') for line in open('TXT/Autumn18.QCD_Pt_800to1000_TuneCP5_13TeV_pythia8_ext1.txt')]
qcd_1000to1400 = [line.rstrip('\n') for line in open('TXT/Autumn18.QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8.txt')]
qcd_1400to1800 = [line.rstrip('\n') for line in open('TXT/Autumn18.QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8.txt')]
qcd_1800to2400 = [line.rstrip('\n') for line in open('TXT/Autumn18.QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8.txt')]
qcd_2400to3200 = [line.rstrip('\n') for line in open('TXT/Autumn18.QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8.txt')]
qcd_3200toInf = [line.rstrip('\n') for line in open('TXT/Autumn18.QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8.txt')]


files = [ttjets_dilept_genMET, ttjets_dilept, ttjets_dilept_sinleptbar_genMET, ttjets_dilept_sinleptbar, ttjets_dilept_sinlept_genMET, ttjets_dilept_sinlept, ttjets_tunecp5, ttjets_dilept_600to800, ttjets_dilept_800to1200, ttjets_dilept_1200to2500, ttjets_dilept_2500toInf, wjets_100to200, wjets_200to400, wjets_400to600, wjets_600to800, wjets_800to1200, wjets_1200to2500, wjets_2500toInf, zjets_100to200, zjets_200to400, zjets_400to600, zjets_600to800, zjets_800to1200, zjets_1200to2500, zjets_2500toInf, qcd_300to470, qcd_470to600, qcd_600to800, qcd_800to1000, qcd_1000to1400, qcd_1400to1800, qcd_1800to2400, qcd_2400to3200, qcd_3200toInf]

for f in files:
  print('file in: ', osp.basename(osp.dirname(f[1])))
  #print(f[1])
  ntot = 0
  njets = 0
  neta = 0
  ntrig = 0
  necf = 0
  nrtx = 0
  npre = 0
  nEleMuon = 0
  nMET = 0
  nMETEleMuon = 0
  n = 0
  for i in f:
    n += 1
    if n > 2: continue
    tf_j = ROOT.TFile.Open(i)
    ttree_j = tf_j.Get("TreeMaker2/PreSelection")
    nbkg_j = ttree_j.GetEntries()
    ntot += nbkg_j
    for i in range(nbkg_j):
     ttree_j.GetEntry(i)
     if ttree_j.JetsAK15.size()<2 or ttree_j.JetsAK8.size()<1: continue
     njets += 1
     if ttree_j.JetsAK15[1].Eta()>2.4: continue
     neta += 1
     if ttree_j.JetsAK8[0].Pt()<550: continue
     ntrig += 1
     if ttree_j.JetsAK15_ecfN2b1[1]<0: continue
     necf += 1
     if sqrt(1+ttree_j.MET/ttree_j.JetsAK15[0].Pt()) < 1.08: continue
     nrtx += 1
     npre += 1
     if ttree_j.NElectrons !=0 or ttree_j.NMuons !=0: continue
     nEleMuon += 1
     if ttree_j.HBHENoiseFilter==0 or ttree_j.eeBadScFilter==0 or ttree_j.BadPFMuonFilter==0 or  ttree_j.globalSuperTightHalo2016Filter==0 or ttree_j.ecalBadCalibReducedFilter==0 or ttree_j.BadChargedCandidateFilter ==0: continue
     nMET += 1
     nMETEleMuon += 1
  print("total number of events: ", ntot)
  print("number of events with at least one jet: %d, efficiency of events with at least one jet: %f"%(njets, (njets/ntot)*100))
  print("number of events with |eta|<2.4: %d, efficiency of events with AK15Jets eta<2.4: %f"%(neta, (neta/ntot)*100))
  print("number of events passing trigger: %d, efficiency of trigger: %f"%(ntrig, (ntrig/ntot)*100))
  print("number of events passing ecf>0: %d, efficiency of ecf>0: %f"%(necf, (necf/ntot)*100))
  print("number of events passing RTx>1.08: %d, efficiency of RTx>1.08: %f"%(nrtx, (nrtx/ntot)*100))
  print("NEle, NMuon: %d, efficiency of nEle, NMuons==0: %f"%(nEleMuon, (nEleMuon/ntot)*100))
  print("METFilter: %d, efficiency of METFilter: %f"%(nMET, (nMET/ntot)*100))
  print("*"*100)
  print("*"*100)

