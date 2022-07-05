from collections import OrderedDict
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
        self.counts = OrderedDict()

    def keys(self):
        return list(self.counts.keys())

    def values(self):
        return list(self.counts.values())

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

    if len(event[b'JetsAK8.fCoordinates.fPt']) == 0 or event[b'JetsAK8.fCoordinates.fPt'][0] < 550.:
        return False
    cut_flow.plus_one('jetak8>550')

    if trigger_evaluator is not None and not(trigger_evaluator(event)):
        return False
    cut_flow.plus_one('trigger')

    for ecf in [
        b'JetsAK15_ecfC2b1', b'JetsAK15_ecfD2b1',
        b'JetsAK15_ecfM2b1', b'JetsAK15_ecfN2b2',
        ]:
        try:
            if event[ecf][1] < 0.:
                return False
        except IndexError:
            return False
    cut_flow.plus_one('ecf>0')

    if np.sqrt(1.+event[b'MET']/event[b'JetsAK15.fCoordinates.fPt'][1]) < 1.1:
        return False
    cut_flow.plus_one('rtx>1.1')

    if event[b'Muons'] > 0 or event[b'Electrons'] > 0:
        return False
    cut_flow.plus_one('nleptons==0')

    if any(event[b] == 0 for b in [
        b'HBHENoiseFilter',
        b'HBHEIsoNoiseFilter',
        b'eeBadScFilter',
        b'ecalBadCalibFilter' if ul else b'ecalBadCalibReducedFilter',
        b'BadPFMuonFilter',
        b'BadChargedCandidateFilter',
        b'globalSuperTightHalo2016Filter',
        ]):
        return False
    cut_flow.plus_one('metfilter')
    cut_flow.plus_one('preselection')
    return True


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


class Status:
    PASSED = 1
    FAILED_PRESEL = 2
    FAILED_SIGNAL_TRUTH = 3
    FAILED_TTSTITCH = 4

def get_feature_vector(
    event, include_signal_truth=False, check_preselection=True, check_ttstitch=False,
    cutflow=None, trigger_evaluator=None, dataset_name=None
    ):
    if check_ttstitch:
        if dataset_name is None:
            raise Exception('Pass argument dataset_name to perform the ttstitch filter')
        if not ttstitch_selection(event, dataset_name, cutflow):
            return Status.FAILED_TTSTITCH, None

    if check_preselection and not preselection(event, cutflow, trigger_evaluator):
        return Status.FAILED_PRESEL, None

    subl = get_subl(event)

    if include_signal_truth:
        genparticles = FourVectorArray(
            event[b'GenParticles.fCoordinates.fPt'],
            event[b'GenParticles.fCoordinates.fEta'],
            event[b'GenParticles.fCoordinates.fPhi'],
            event[b'GenParticles.fCoordinates.fE'],
            pdgid=event[b'GenParticles_PdgId'],
            status=event[b'GenParticles_Status']
            )

        zprime = genparticles[genparticles.pdgid == 4900023]
        if len(zprime) == 0: return Status.FAILED_SIGNAL_TRUTH, None
        zprime = zprime[0]

        dark_quarks = genparticles[(np.abs(genparticles.pdgid) == 4900101) & (genparticles.status == 71)]
        if len(dark_quarks) != 2: return Status.FAILED_SIGNAL_TRUTH, None

        # Verify zprime and dark_quarks are within 1.5 of the jet
        if not all(calc_dr(subl.eta, subl.phi, obj.eta, obj.phi) < 1.5 for obj in [
            zprime, dark_quarks[0], dark_quarks[1]
            ]):
            return Status.FAILED_SIGNAL_TRUTH, None

    # MUST BE IN SYNC WITH THE VAR "FEATURE_TITLES" BELOW
    X = [
        subl.girth, subl.ptD, subl.axismajor, subl.axisminor,
        subl.ecfM2b1, subl.ecfD2b1, subl.ecfC2b1, subl.ecfN2b2,
        subl.metdphi,
        subl.pt, subl.eta, subl.phi, subl.energy,
        subl.rt, subl.mt
        ]
    
    if include_signal_truth:
        X_truth = [zprime.pt, zprime.eta, zprime.phi, zprime.energy]

    return Status.PASSED, ((X, X_truth) if include_signal_truth else X)

FEATURE_TITLES = [
    'girth', 'ptD', 'axismajor', 'axisminor',
    'ecfM2b1', 'ecfD2b1', 'ecfC2b1', 'ecfN2b2',
    'metdphi',
    'pt', 'eta', 'phi', 'energy',
    'rt', 'mt'
    ]

def del_features(X, names):
    """
    Deletes columns from a feature array by name
    """
    remove_indices = [bdtcode.dataset.FEATURE_TITLES.index(name) for name in names]
    all_indices = np.arange(len(FEATURE_TITLES))
    filtered_indices = all_indices[~np.isin(all_indices, remove_indices)]
    return X[:,filtered_indices]



def vstack(X, *args, **kwargs):
    """
    Wrapper around np.vstack that doesn't crash on empty lists
    """
    return X if X == [] else np.vstack(X, *args, **kwargs)



def apply_bdt(model, rootfiles, outfile, skip_features=['mt', 'rt'], dataset_name=None, nmax=None):
    """
    Applies a BDT (in `model`) on all events in `rootfiles`, and saves output in `outfile` (.npz).

    `rootfiles` can be a list of paths or just a single path to a rootfile, and may be remote.

    `dataset_name` should indicate what type of ttbar sample this is, and should be an empty string
    for non-ttbar samples.

    `skip_features` should be a list of features that were not used in the training of the BDT.
    """
    rootfiles = uptools.format_rootfiles(rootfiles)
    cutflow = CutFlowColumn()
    trigger_evaluator = TriggerEvaluator(rootfiles[0])

    if dataset_name is None:
        dataset_name = '/'.join(rootfiles[0].split('/')[-3:])
        bdtcode.logger.info(f'Using {dataset_name=}')

    X = []
    X_histogram = []

    for rootfile in rootfiles:
        bdtcode.logger.info(f'Start processing {rootfile}')
        try:
            for event in uptools.iter_events(rootfile, nmax=nmax):
                cutflow.plus_one('total')
                status, vector = get_feature_vector(
                    event, include_signal_truth=False,
                    check_preselection=True, check_ttstitch=True, cutflow=cutflow,
                    trigger_evaluator=trigger_evaluator,
                    dataset_name=dataset_name
                    )
                if status != Status.PASSED: continue
                X.append(vector)

                # Get some histogramming variables
                subl = get_subl(event)        
                mt, rt = calculate_mt_rt(subl, event[b'MET'], event[b'METPhi'])

                # TO BE KEPT IN SYNC WITH HISTOGRAMMING_VARIABLE_TITLES BELOW
                X_histogram.append([
                    calculate_mass(subl), mt, rt,
                    subl.energy, subl.pt, subl.eta, subl.phi,
                    event[b'MET'], event[b'METPhi']
                    ])
        except IndexError:
            bdtcode.logger.error(f'Problem with {rootfile}; proceeding with {X.shape[0]} good entries')
        except Exception as e:
            bdtcode.logger.error(f'Error processing {rootfile}; Skipping. Error was: ' + repr(e))

    X = vstack(X)
    X_histogram = vstack(X_histogram)

    n_events = X.shape[0]

    # Now get the scores
    bdtcode.logger.info(f'Applying bdt on {n_events} events')
    scores = model.predict_proba(del_features(X, skip_features))[:,1] if n_events else []

    out_dct = dict(
        X=X, titles=FEATURE_TITLES,
        scores=scores,
        X_histogram=X_histogram, titles_histogram=HISTOGRAMMING_VARIABLE_TITLES,
        cutflow_titles = cutflow.keys(), cutflow_values = cutflow.values()
        )

    if outfile is not None:
        outdir = osp.abspath(osp.dirname(outfile))
        if not osp.isdir(outdir): os.makedirs(outdir)
        bdtcode.logger.info(f'Saving {n_events} entries to {outfile}')
        np.savez(outfile, **out_dct)
    
    return out_dct


# TO BE KEPT IN SYNC WITH ORDER OF VARIABLES IN apply_bdt()
HISTOGRAMMING_VARIABLE_TITLES = [
    'mass', 'mt', 'rt',
    'energy', 'pt', 'eta', 'phi',
    'met', 'metphi'
    ]


def make_feature_npz_signal(rootfiles, outfile=None):
    """
    Takes a list of rootfiles, and builds a combined feature vector from all events
    in those rootfiles that pass certain cuts (see `get_feature_vector`).
    """
    n_total = 0
    n_presel = 0
    n_passed = 0
    X = []
    X_truth = []

    for event in uptools.iter_events(rootfiles):
        n_total += 1
        status, vectors = get_feature_vector(event, include_signal_truth=True)
        if status == Status.FAILED_PRESEL: continue
        n_presel += 1
        if status == Status.FAILED_SIGNAL_TRUTH: continue
        n_passed += 1

        X.append(vectors[0])
        X_truth.append(vectors[1])

    bdtcode.logger.info(f'n_total: {n_total}; n_presel: {n_presel}; n_final: {n_passed} ({100.*n_passed/float(n_total):.2f}%)')

    if outfile:
        outdir = osp.abspath(osp.dirname(outfile))
        if not osp.isdir(outdir): os.makedirs(outdir)
        bdtcode.logger.info(f'Saving {n_passed} entries to {outfile}')
        np.savez(outfile, X=vstack(X), X_truth=vstack(X_truth), titles=FEATURE_TITLES)

    return X, X_truth


def make_feature_npzs_bkg(rootfiles, outfile):
    X = []
    for rootfile in uptools.format_rootfiles(rootfiles):
        bdtcode.logger.info(f'Processing {rootfile}')
        n_total = 0
        n_passed = 0
        try:
            for event in uptools.iter_events(rootfile):
                n_total += 1
                status, vector = get_feature_vector(event, include_signal_truth=False)
                if status == Status.FAILED_PRESEL: continue
                n_passed += 1
                X.append(vector)
        except IndexError:
            if n_total == 0:
                bdtcode.logger.error(f'Problem with {rootfile}; no entries, skipping')
                continue
            else:
                bdtcode.logger.error(f'Problem with {rootfile}; saving {n_passed} good entries')

        bdtcode.logger.info(f'n_total: {n_total}; n_passed: {n_passed} ({(100.*n_passed)/n_total:.2f}%)')

    if not osp.isdir(osp.abspath(osp.dirname(outfile))): os.makedirs(osp.abspath(osp.dirname(outfile)))
    bdtcode.logger.info(f'Saving {n_passed} entries to {outfile}')
    np.savez(outfile, X=vstack(X), titles=FEATURE_TITLES)



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



def outdated():
    make_feature_npz_signal(
        list(sorted(glob.iglob('/data/users/snabili/BSVJ/08242020/CMSSW_10_2_21/src/TreeMaker/Production/test/YiMu_genSamples/finaltreemakersamples/M250.root'))),
        'out.npz'
        )

if __name__ == '__main__':
    # outdated()
    test_make_feature_npz()
