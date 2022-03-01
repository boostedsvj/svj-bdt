import re, uuid, numpy as np
from array import array
from typing import List

from . import histogramming as H
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
        Returns inclusive cross section based on the label
        """
        return crosssections.label_to_xs(self.label)

    @property
    def pt(self):
        return self.d['pt']

    #def mt(self, min_score=None):
    def mt(self, min_score=None, pt_min=None):
        """Returns mt, with the option of cutting on the score here"""
        #return self.d['mt'] if min_score is None else self.d['mt'][self.score > min_score]
        return self.d['mt'][self.pt > pt_min] if min_score is None else self.d['mt'][(self.score > min_score) & (self.pt > pt_min)]

    @property
    def score(self):
        return self.d['score']

    def bdt_efficiency(self, min_score=None):
        if min_score is None:
            return 1.
        elif is_array(min_score):
            # Multiple scores are requested in one go; turn min_score
            # into a column vector to ensure correct broadcasting
            min_score = np.expand_dims(min_score, -1)
        return (self.score > min_score).sum(axis=-1) / len(self)

    @property
    def preselection_efficiency(self):
        return self.d['preselection']/self.d['total']

    def nevents_after_preselection(self, lumi=137.2*1e3):
        return self.crosssection * lumi * self.preselection_efficiency * self.genjetpt_efficiency

    def nevents_after_bdt(self, min_score=None, lumi=137.2*1e3):
        return self.nevents_after_preselection(lumi) * self.bdt_efficiency(min_score)

    def __len__(self):
        """Returns number of entries in the underlying dict"""
        return len(self.score)


#def sample_to_mt_histogram(sample: Sample, min_score=None, mt_binning=None, name=None):
def sample_to_mt_histogram(sample: Sample, pt_min=None, min_score=None, mt_binning=None, name=None):
    try_import_ROOT()
    import ROOT
    #mt = sample.mt(min_score)
    mt = sample.mt(min_score, pt_min)
    binning = array('f', crosssections.MT_BINNING if mt_binning is None else mt_binning)
    if name is None: name = str(uuid.uuid4())
    h = ROOT.TH1F(name, name, len(binning)-1, binning)
    ROOT.SetOwnership(h, False)
    [ h.Fill(x) for x in mt ]
    H.normalize(h, sample.nevents_after_bdt(min_score))
    return h
