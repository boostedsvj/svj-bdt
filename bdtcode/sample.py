import re, uuid, numpy as np
from array import array
from typing import List

from . import histogramming as H
#from . import histogramming_noscore as H
from .utils import *
from . import crosssections


class Sample:
    """
    Container for a sample.

    A sample can be a specific set of background events, e.g. 
    TTJets_HT-600to800, QCD_Pt_800to1000_TuneCP5, mz250, etc.

    The container stores a simply dictionary in self.d and
    has a few convenience methods to interact with 
    """

    def __init__(self, label, d):
        self.d = d
        self.label = label

    @property
    def mz(self):
        """
        Returns the Z' mass of the sample, based on the label.
        If this sample is background, None is returned.
        """
        if not hasattr(self, '_mz'): 
            match = re.search(r'mz(\d+)', self.label)
            self._mz = int(match.group(1)) if match else None
        return self._mz
 
    @property
    def is_sig(self):
        return self.mz is not None

    @property
    def is_bkg(self):
        return self.mz is None

    @property
    def genjetpt_efficiency(self):
        if self.is_bkg: return 1.
        return crosssections.genjetpt_eff(self.mz)

    @property
    def crosssection(self):
        """
        Returns inclusive cross section based on the label.
        Takes into account the ttstitch efficiency
        """
        return crosssections.label_to_xs(self.label) * self.ttstitch_efficiency

    @property
    def n_mc(self):
        """
        Returns the number of mc events used in the sample.
        If the sample is ttjets, it returns the number of mc events
        _after_ the stitching mask.
        """
        return self.d.get('ttstitch', self.d['total'])

    @property
    def pt(self):
        return np.array(self.d['pt'])

    @property
    def rt(self):
        return np.array(self.d['rt'])

    @property
    def dphi(self):
        return np.array(self.d['dphi'])

    @property
    def eta(self):
        return np.array(self.d['eta'])

    @property
    def trig(self):
        return np.array(self.d['trig'])

    def better_resolution_selection(self, pt_min=None, rt_min=None, dphi_max=None, eta_max=None):
        """
        Returns the selection needed to improve resolution
        """
        selection = np.ones(len(self), dtype=bool)
        # Only select if the parameter is given
        if pt_min is not None:
            selection = (selection & (self.pt > pt_min))
        if rt_min is not None:
            selection = (selection & (self.rt > rt_min))
        if dphi_max is not None:
            selection = (selection & (abs(self.dphi) < dphi_max))
        if eta_max is not None:
            selection = (selection & (abs(self.eta)<eta_max))
        return selection

    def mt(self, min_score=None, **better_resolution_selectors):
        selection = self.better_resolution_selection(**better_resolution_selectors)
        if min_score:
            selection = (selection & (self.score > min_score))
        # print(self.d['mt'], type(self.d['mt']))
        # print(selection, type(selection))
        #return np.array(self.d['mt'])[selection]
        return np.array(self.d['mt'][selection]) #-->new mass variable

    @property
    def score(self):
        return np.array(self.d['score'])

    def bdt_efficiency(self, min_score=None):
        if min_score is None:
            return 1.
        elif is_array(min_score):
            # Multiple scores are requested in one go; turn min_score
            # into a column vector to ensure correct broadcasting
            min_score = np.expand_dims(min_score, -1)
        return (self.score > min_score).sum(axis=-1) / len(self)

    def other_selection_efficiency(self, **better_resolution_selectors):
        selection = self.better_resolution_selection(**better_resolution_selectors)
        return selection.sum() / len(self)

    @property
    def preselection_efficiency(self):
        return self.d.get('preselection', 0) / self.n_mc

    @property
    def ttstitch_efficiency(self):
        return self.n_mc / self.d['total']

    #def nevents_after_preselection(self, lumi=137.2*1e3):
    def nevents_after_preselection(self, lumi=14.027*1e3): #data 2018 PreHEM
        return self.crosssection * lumi * self.preselection_efficiency * self.genjetpt_efficiency

    #def nevents_after_bdt(self, min_score=None, lumi=137.2*1e3):
    def nevents_after_bdt(self, min_score=None, lumi=14.027*1e3): #data 2018 PreHEM
        return self.nevents_after_preselection(lumi) * self.bdt_efficiency(min_score)
        
    #def nevents_after_allcuts(self, min_score=None, lumi=137.2*1e3, **better_resolution_selectors):
    def nevents_after_allcuts(self, min_score=None, lumi=14.027*1e3, **better_resolution_selectors): #data 2018 PreHEM
        return self.nevents_after_bdt(min_score, lumi) * self.other_selection_efficiency(**better_resolution_selectors)

    def __len__(self):
        """Returns number of entries in the underlying dict"""
        return len(self.score)


def sample_to_mt_histogram(sample: Sample, min_score=None, mt_binning=None, name=None, **better_resolution_selectors):
    try_import_ROOT()
    import ROOT
    mt = sample.mt(min_score, **better_resolution_selectors)
    binning = array('f', crosssections.MT_BINNING if mt_binning is None else mt_binning)
    if name is None: name = str(uuid.uuid4())
    h = ROOT.TH1F(name, name, len(binning)-1, binning)
    ROOT.SetOwnership(h, False)
    [ h.Fill(x) for x in mt ]
    H.normalize(h, sample.nevents_after_allcuts(min_score, **better_resolution_selectors))
    return h
