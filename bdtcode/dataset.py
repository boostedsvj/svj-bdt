import os, os.path as osp, glob
import numpy as np
import tqdm
import uptools, seutils
from contextlib import contextmanager
import argparse

import bdtcode
from .utils import is_array


class Bunch:
    def __init__(self, **kwargs):
        self.arrays = kwargs

    def __getattr__(self, name):
       return self.arrays[name]

    def __getitem__(self, where):
        """Selection mechanism"""
        new = self.__class__()
        new.arrays = {k: v[where] for k, v in self.arrays.items()}
        return new

    def __len__(self):
        for k, v in self.arrays.items():
            try:
                return len(v)
            except TypeError:
                return 1


class FourVectorArray:
    """
    Wrapper class for Bunch, with more specific 4-vector stuff
    """
    def __init__(self, pt, eta, phi, energy, **kwargs):
        self.bunch = Bunch(
            pt=pt, eta=eta, phi=phi, energy=energy, **kwargs
            )

    def __getattr__(self, name):
       return getattr(self.bunch, name)

    def __getitem__(self, where):
        new = self.__class__([], [], [], [])
        new.bunch = self.bunch[where]
        return new

    def __len__(self):
        return len(self.bunch)

    @property
    def px(self):
        return np.cos(self.phi) * self.pt

    @property
    def py(self):
        return np.sin(self.phi) * self.pt

    @property
    def pz(self):
        return np.sinh(self.eta) * self.pt


def calc_dphi(phi1, phi2):
    """
    Calculates delta phi. Assures output is within -pi .. pi.
    """
    twopi = 2.*np.pi
    # Map to 0..2pi range
    dphi = (phi1 - phi2) % twopi
    # Map pi..2pi --> -pi..0
    if is_array(dphi):
        dphi[dphi > np.pi] -= twopi
    elif dphi > np.pi:
        dphi -= twopi
    return dphi


def calc_dr(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1-eta2)**2 + calc_dphi(phi1, phi2)**2)


def calculate_mt_rt(jets, met, metphi):
    met_x = np.cos(metphi) * met
    met_y = np.sin(metphi) * met
    jet_x = np.cos(jets.phi) * jets.pt
    jet_y = np.sin(jets.phi) * jets.pt
    # jet_e = np.sqrt(jets.mass2 + jets.pt**2)
    # m^2 + pT^2 = E^2 - pT^2 - pz^2 + pT^2 = E^2 - pz^2
    jet_e = np.sqrt(jets.energy**2 - jets.pz**2)
    mt = np.sqrt( (jet_e + met)**2 - (jet_x + met_x)**2 - (jet_y + met_y)**2 )
    rt = np.sqrt(1+ met / jets.pt)
    return mt, rt

def calculate_mt(jets, met, metphi):
    metx = np.cos(metphi) * met
    mety = np.sin(metphi) * met
    # Actually np.sqrt(jets.mass2 + jets.pt**2), 
    # but mass2 = energy**2 - pt**2 - pz**2
    jets_transverse_e = np.sqrt(jets.energy**2 - jets.pz**2)
    mt = np.sqrt(
        (jets_transverse_e + met)**2
        - (jets.px + metx)**2 - (jets.py + mety)**2
        )
    return mt


def calculate_mass(jets):
    mass = np.sqrt(jets.energy**2 - jets.px**2 - jets.py**2 - jets.pz**2)
    return mass

def calculate_massmet(jets, met, metphi):
    metx = np.cos(metphi) * met
    mety = np.sin(metphi) * met
    mass_viz = np.sqrt(jets.energy**2 - jets.px**2 - jets.py**2 - jets.pz**2)
    metdphi = calc_dphi(jets.phi, metphi)
    massmet = np.sqrt(mass_viz**2 + 2 * met * np.sqrt(jets.pz**2 + jets.pt**2 + mass_viz**2) - 2 * jets.pt * met * np.cos(metdphi))
    return massmet

def calculate_massmetpz(jets, met, metphi):
    metx = np.cos(metphi) * met
    mety = np.sin(metphi) * met
    mass_viz = np.sqrt(jets.energy**2 - jets.px**2 - jets.py**2 - jets.pz**2)
    mass = np.sqrt(mass_viz**2 + 2 * np.sqrt(met**2 + jets.pz**2 ) * np.sqrt(jets.pz**2 + jets.pt**2 + mass_viz**2) - 2 * (jets.pt * met * np.cos(calc_dphi(metphi, jets.phi)) + jets.pz**2))
    return mass

def calculate_massmetpzm(jets, met, metphi):
    metx = np.cos(metphi) * met
    mety = np.sin(metphi) * met
    mass_viz = np.sqrt(jets.energy**2 - jets.px**2 - jets.py**2 - jets.pz**2)
    mass = np.sqrt(2*mass_viz**2 +2*np.sqrt(met**2+jets.pz**2+mass_viz**2)*np.sqrt(jets.pz**2+jets.pt**2+mass_viz**2)-2*(jets.pt*met*np.cos(calc_dphi(metphi, jets.phi))+jets.pz**2))
    return mass


class CutFlowColumn:
    def __init__(self) -> None:
        self.counts = {}

    def plus_one(self, name):
        self.counts.setdefault(name, 0)
        self.counts[name] += 1

    def __getitem__(self, name):
        return self.counts.get(name, 0)


def ttstitch_selection(event, dataset_name, cutflow=None):
    if not 'TTJets' in dataset_name: return True
    mad_ht = event[b'madHT']
    gen_met = event[b'GenMET']
    n_leptons = np.isin( np.abs(event[b'GenParticles_PdgId']), [11, 13, 15] ).sum()

    if 'TTJets_TuneCP5_13TeV-madgraphMLM-pythia8' in dataset_name:
        # Inclusive
        passes = mad_ht < 600. and n_leptons == 0
    elif 'TTJets_HT-' in dataset_name:
        passes = mad_ht >= 600.
    elif 'TTJets_DiLep' in dataset_name or 'TTJets_SingleLep' in dataset_name:
        if not 'genMET' in dataset_name:
            passes = mad_ht < 600. and gen_met < 80.
        else:
            passes = mad_ht < 600. and gen_met >= 80.

    if cutflow and passes: cutflow.plus_one('ttstitch')
    return passes


triggers_2018 = [
    # AK8PFJet triggers
    'HLT_AK8PFJet500_v',
    'HLT_AK8PFJet550_v',
    # CaloJet
    'HLT_CaloJet500_NoJetID_v',
    'HLT_CaloJet550_NoJetID_v',
    # PFJet and PFHT
    'HLT_PFHT1050_v', # but, interestingly, not HLT_PFHT8**_v or HLT_PFHT9**_v, according to the .txt files at least
    'HLT_PFJet500_v',
    'HLT_PFJet550_v',
    # Trim mass jetpt+HT
    'HLT_AK8PFHT800_TrimMass50_v',
    'HLT_AK8PFHT850_TrimMass50_v',
    'HLT_AK8PFHT900_TrimMass50_v',
    'HLT_AK8PFJet400_TrimMass30_v',
    'HLT_AK8PFJet420_TrimMass30_v',
    # MET triggers
    # 'HLT_PFHT500_PFMET100_PFMHT100_IDTight_v',
    # 'HLT_PFHT500_PFMET110_PFMHT110_IDTight_v',
    # 'HLT_PFHT700_PFMET85_PFMHT85_IDTight_v',
    # 'HLT_PFHT700_PFMET95_PFMHT95_IDTight_v',
    # 'HLT_PFHT800_PFMET75_PFMHT75_IDTight_v',
    # 'HLT_PFHT800_PFMET85_PFMHT85_IDTight_v',
    ]


class TriggerEvaluator:
    def __init__(self, rootfile):
        """
        `rootfile` should be a path to a rootfile from which to read the title of the 
        'TriggerPass' branch.
        """
        import uproot3
        title_bstring = uproot3.open(rootfile).get('TreeMaker2/PreSelection')[ b'TriggerPass'].title
        self.titles = title_bstring.decode('utf-8').split(',')
        self.index_map = {self.titles[i] : i for i in range(len(self.titles)) }
        self.year_map = {}

    def get_indices_for_year(self, year):
        if not year in self.year_map:
            titles_this_year = globals()['triggers_{}'.format(year)]
            self.year_map[year] = np.array([self.index_map[title] for title in titles_this_year])
        return self.year_map[year]

    def __call__(self, event, year=2018):
        indices = self.get_indices_for_year(year)
        return np.any(event[b'TriggerPass'][indices] == 1)


def preselection(event, cut_flow=None, trigger_evaluator=None, ul=None):
    if ul is None: ul = bdtcode.UL # if not determined, use the global default

    if cut_flow is None: cut_flow = CutFlowColumn()

    if len(event[b'JetsAK15.fCoordinates.fPt']) < 2:
        return False
    cut_flow.plus_one('>=2jets')

    if abs(event[b'JetsAK15.fCoordinates.fEta'][1]) > 2.4:
        return False
    cut_flow.plus_one('eta<2.4')

    #if len(event[b'JetsAK8.fCoordinates.fPt']) == 0 or event[b'JetsAK8.fCoordinates.fPt'][0] < 550.:
    #    return False
    cut_flow.plus_one('jetak8>550')

    if trigger_evaluator is not None and not(trigger_evaluator(event)):
        return False
    cut_flow.plus_one('trigger')

    '''for ecf in [
        b'JetsAK15_ecfC2b1', b'JetsAK15_ecfD2b1',
        b'JetsAK15_ecfM2b1', b'JetsAK15_ecfN2b2',
        ]:
        try:
            if event[ecf][1] < 0.:
                return False
        except IndexError:
            return False'''
    cut_flow.plus_one('ecf>0')

    # no rtx cut for control region study
    if np.sqrt(1.+event[b'MET']/event[b'JetsAK15.fCoordinates.fPt'][1]) < 1.0:
        return False
    cut_flow.plus_one('rtx>1.0')

    # control region rt<1.15
    '''if np.sqrt(1.+event[b'MET']/event[b'JetsAK15.fCoordinates.fPt'][1]) < 1.0 or np.sqrt(1.+event[b'MET']/event[b'JetsAK15.fCoordinates.fPt'][1]) > 1.15:
        return False
    cut_flow.plus_one('rtx_CR')'''


    #if event[b'Muons'] > 0 or event[b'Electrons'] > 0:
    #    return False
    cut_flow.plus_one('nleptons==0')


    # comment out metfilter 
    # add metfilters as booleans in histogramming.py 
    '''if any(event[b] == 0 for b in [
        b'HBHENoiseFilter',
        b'HBHEIsoNoiseFilter',
        b'eeBadScFilter',
        b'ecalBadCalibFilter' if ul else b'ecalBadCalibReducedFilter',
        b'BadPFMuonFilter',
        b'BadChargedCandidateFilter',
        b'globalSuperTightHalo2016Filter',
        ]):
        return False'''
    cut_flow.plus_one('metfilter')
    cut_flow.plus_one('preselection')
    return True

def get_ak4_subl(event):
    """
    Returns subleading jet
    """
    jets = FourVectorArray(
        event[b'Jets.fCoordinates.fPt'],
        event[b'Jets.fCoordinates.fEta'],
        event[b'Jets.fCoordinates.fPhi'],
        event[b'Jets.fCoordinates.fE'],
        )
    ak4_subl = jets[1]
    return ak4_subl

def get_ak4_lead(event):
    """
    Returns subleading jet
    """
    jets = FourVectorArray(
        event[b'Jets.fCoordinates.fPt'],
        event[b'Jets.fCoordinates.fEta'],
        event[b'Jets.fCoordinates.fPhi'],
        event[b'Jets.fCoordinates.fE'],
        )
    ak4_lead = jets[0]
    return ak4_lead

def get_ak8_lead(event):
    """
    Returns subleading jet
    """
    jets = FourVectorArray(
        event[b'JetsAK8.fCoordinates.fPt'],
        event[b'JetsAK8.fCoordinates.fEta'],
        event[b'JetsAK8.fCoordinates.fPhi'],
        event[b'JetsAK8.fCoordinates.fE'],
        )
    ak8_lead = jets[0]
    return ak8_lead


def get_subl(event):
    """
    Returns subleading jet
    """
    jets = FourVectorArray(
        event[b'JetsAK15.fCoordinates.fPt'],
        event[b'JetsAK15.fCoordinates.fEta'],
        event[b'JetsAK15.fCoordinates.fPhi'],
        event[b'JetsAK15.fCoordinates.fE'],
        ecfC2b1 = event[b'JetsAK15_ecfC2b1'],
        ecfC2b2 = event[b'JetsAK15_ecfC2b2'],
        # ecfC3b1 = event[b'JetsAK15_ecfC3b1'],
        # ecfC3b2 = event[b'JetsAK15_ecfC3b2'],
        ecfD2b1 = event[b'JetsAK15_ecfD2b1'],
        ecfD2b2 = event[b'JetsAK15_ecfD2b2'],
        ecfM2b1 = event[b'JetsAK15_ecfM2b1'],
        ecfM2b2 = event[b'JetsAK15_ecfM2b2'],
        # ecfM3b1 = event[b'JetsAK15_ecfM3b1'],
        # ecfM3b2 = event[b'JetsAK15_ecfM3b2'],
        ecfN2b1 = event[b'JetsAK15_ecfN2b1'],
        ecfN2b2 = event[b'JetsAK15_ecfN2b2'],
        # ecfN3b1 = event[b'JetsAK15_ecfN3b1'],
        # ecfN3b2 = event[b'JetsAK15_ecfN3b2'],
        multiplicity = event[b'JetsAK15_multiplicity'],
        girth = event[b'JetsAK15_girth'],
        ptD = event[b'JetsAK15_ptD'],
        axismajor = event[b'JetsAK15_axismajor'],
        axisminor = event[b'JetsAK15_axisminor'],
        )
    subl = jets[1]
    subl.metdphi = calc_dphi(subl.phi, event[b'METPhi'])
    metdphi = calc_dphi(subl.phi, event[b'METPhi'])
    subl.rt = np.sqrt(1+ event[b'MET']/subl.pt)
    subl.mt = calculate_mt(subl, event[b'MET'], metdphi)
    return subl

def get_zprime(rootfile):
#def get_zprime(event):
    for event in uptools.iter_events(rootfile):
        genparticles = FourVectorArray(
            event[b'GenParticles.fCoordinates.fPt'],
            event[b'GenParticles.fCoordinates.fEta'],
            event[b'GenParticles.fCoordinates.fPhi'],
            event[b'GenParticles.fCoordinates.fE'],
            pdgid=event[b'GenParticles_PdgId'],
            status=event[b'GenParticles_Status']
            )
        zprime = genparticles[genparticles.pdgid == 4900023]
        if len(zprime) == 0: continue
        zprime = zprime[0]
        return zprime

def process_signal(rootfiles, outfile=None):
    n_total = 0
    n_presel = 0
    n_final = 0

    X = []

    for event in uptools.iter_events(rootfiles):
        n_total += 1
        if not preselection(event): continue
        n_presel += 1

        genparticles = FourVectorArray(
            event[b'GenParticles.fCoordinates.fPt'],
            event[b'GenParticles.fCoordinates.fEta'],
            event[b'GenParticles.fCoordinates.fPhi'],
            event[b'GenParticles.fCoordinates.fE'],
            pdgid=event[b'GenParticles_PdgId'],
            status=event[b'GenParticles_Status']
            )

        zprime = genparticles[genparticles.pdgid == 4900023]
        print("11111")
        if len(zprime) == 0: continue
        print("22222")
        #zprime = zprime[0]

        '''dark_quarks = genparticles[(np.abs(genparticles.pdgid) == 4900101) & (genparticles.status == 71)]
        if len(dark_quarks) != 2: continue'''


        if len(event[b'JetsAK15.fCoordinates.fPt']) < 2 or abs(event[b'JetsAK15.fCoordinates.fEta'][1]) > 2.4 or len(event[b'JetsAK8.fCoordinates.fPt']) < 1 or len(event[b'Jets.fCoordinates.fPt']) < 2: continue

        sublak15 = get_subl(event)
        leadak8  = get_ak8_lead(event)
        sublak4  = get_ak4_subl(event)
        leadak4  = get_ak4_lead(event)
        met = event[b'MET']
        metphi = event[b'METPhi'] 
        sublak15.mass = calculate_mass(sublak15)
        zprime.mass = calculate_mass(zprime)
        print("33333")
        leadak15pt = event[b'JetsAK15.fCoordinates.fPt'][0]
        leadak15phi = event[b'JetsAK15.fCoordinates.fPhi'][0]
        leadak15eta = event[b'JetsAK15.fCoordinates.fEta'][0]

        '''# Verify zprime and dark_quarks are within 1.5 of the jet
        if not all(calc_dr(subl.eta, subl.phi, obj.eta, obj.phi) < 1.5 for obj in [
            zprime, dark_quarks[0], dark_quarks[1]
            ]):
            continue

        n_final += 1

        X.append([
            subl.ptD, subl.axismajor, subl.multiplicity, subl.rt, subl.mt,
            subl.girth, subl.axisminor, subl.metdphi,  
            #subl.girth, subl.ptD, subl.axismajor, subl.axisminor,
            subl.ecfM2b1, subl.ecfD2b1, subl.ecfC2b1, subl.ecfN2b2,
            subl.metdphi,
            subl.pt, subl.eta, subl.phi, subl.energy,
            zprime.pt, zprime.eta, zprime.phi, zprime.energy
            ])
        #X.append([subl.rt])'''
        X.append([
            sublak15.pt, sublak15.eta, sublak15.phi, sublak15.mass, sublak15.energy, sublak15.ecfM2b1, sublak15.ecfD2b1, sublak15.ecfC2b1, sublak15.ecfN2b2,
            leadak8.pt,
            leadak15pt, leadak15phi, leadak15eta,
            leadak4.pt, leadak4.eta, leadak4.phi,
            sublak4.pt, sublak4.eta, sublak4.phi,
            zprime.pt, zprime.eta, zprime.phi, zprime.energy, zprime.mass,
            met, metphi
            ])
        print("44444")
    print(f'n_total: {n_total}; n_presel: {n_presel}; n_final: {n_final} ({100.*n_final/float(n_total):.2f}%)')

    if outfile is None: outfile = 'data/signal.npz'
    outdir = osp.abspath(osp.dirname(outfile))
    if not osp.isdir(outdir): os.makedirs(outdir)
    print(f'Saving {n_final} entries to {outfile}')
    np.savez(outfile, X=X)


def process_bkg(rootfiles, outfile=None, chunked_save=None, nmax=None):
    n_total_all = 0
    n_presel_all = 0
    for rootfile in uptools.format_rootfiles(rootfiles):
        X = []
        n_total_this = 0
        n_presel_this = 0
        try:
            for event in uptools.iter_events(rootfile):
                n_total_this += 1
                n_total_all += 1
                if not preselection(event): continue
                n_presel_this += 1
                n_presel_all += 1
                subl = get_subl(event)
                X.append([
                    subl.girth, subl.ptD, subl.axismajor, subl.axisminor,
                    subl.ecfM2b1, subl.ecfD2b1, subl.ecfC2b1, subl.ecfN2b2,
                    subl.metdphi,
                    subl.pt, subl.eta, subl.phi, subl.energy
                    ])
        except IndexError:
            if n_presel_this == 0:
                print(f'Problem with {rootfile}; no entries, skipping')
                continue
            else:
                print(f'Problem with {rootfile}; saving {n_presel_this} good entries')

        outfile = 'data/bkg/{}.npz'.format(dirname_plus_basename(rootfile).replace('.root', ''))
        print(f'n_total: {n_total_this}; n_presel: {n_presel_this} ({(100.*n_presel_this)/n_total_this:.2f}%)')
        outdir = osp.abspath(osp.dirname(outfile))
        if not osp.isdir(outdir): os.makedirs(outdir)
        print(f'Saving {n_presel_this} entries to {outfile}')
        np.savez(outfile, X=X)



def dirname_plus_basename(fullpath):
    return f'{osp.basename(osp.dirname(fullpath))}/{osp.basename(fullpath)}'

@contextmanager
def make_local(rootfile):
    """Copies rootfile to local, and removes when done"""
    tmpfile = f'tmp/{dirname_plus_basename(rootfile)}'
    seutils.cp(rootfile, tmpfile)
    try:
        yield tmpfile
    finally:
        print(f'Removing {tmpfile}')
        os.remove(tmpfile)


def iter_rootfiles_umd(rootfiles):
    for rootfile in rootfiles:
        with make_local(rootfile) as tmpfile:
            yield tmpfile


'''def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, choices=['signal', 'bkg', 'signal_local'])
    args = parser.parse_args


    if args.signal_local:
        process_signal(
            list(sorted(glob.iglob('raw_signal/*.root')))
            )
    elif args.signal:
    if args.signal:
        process_signal(
            iter_rootfiles_umd(
                seutils.ls_wildcard(
                    #'gsiftp://hepcms-gridftp.umd.edu//mnt/hadoop/cms/store/user/snabili/BKG/sig_mz250_rinv0p3_mDark20_Mar31/*.root'
		    #'root://cmseos.fnal.gov//store/user/lpcdarkqcd/MCSamples_Summer21/TreeMaker/genjetpt375_mz250_mdark10_rinv0.3/*.root'
		    '/home/snabili/Bsvj/YiMu_genSamples/finaltreemakersamples/M250.root'
                    ),
                #+ ['gsiftp://hepcms-gridftp.umd.edu//mnt/hadoop/cms/store/user/thomas.klijnsma/qcdtest3/sig_ECF_typeCDMN_Jan29/1.root']
                #),
            ))
    elif args.bkg:
        process_bkg(
            iter_rootfiles_umd(seutils.ls_wildcard(
                'gsiftp://hepcms-gridftp.umd.edu//mnt/hadoop/cms/store/user/snabili/BKG/bkg_May04_year2018/*/*.root'
                )),
            )'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('signal', type=str)
    args = parser.parse_args()

    if args.signal:
      process_signal(
            #list(sorted(glob.iglob('/data/users/snabili/BSVJ/08242020/CMSSW_10_2_21/src/TreeMaker/Production/test/YiMu_genSamples/finaltreemakersamples/treemaker_mz350mDard10rinv0p3.root')))
            list(sorted(glob.iglob('/data/users/snabili/BSVJ/08242020/CMSSW_10_2_21/src/TreeMaker/Production/test/YiMu_genSamples/finaltreemakersamples/M250.root')))
            )

if __name__ == '__main__':
    main()
