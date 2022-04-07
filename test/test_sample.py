import pytest
import bdtcode.sample as sample
import numpy as np

def test_bdt_efficiency():
    s = sample.Sample('label', {'score' : np.array([.11, .11, .21, .31, .31])})
    assert s.bdt_efficiency(None) == 1.
    assert s.bdt_efficiency(.2) == 3/5.
    np.testing.assert_array_equal(s.bdt_efficiency(np.array([.1, .2, .3])), np.array([5, 3, 2])/5.)

def test_nevents_after_bdt():
    s = sample.Sample('label', {'score' : np.array([.11, .11, .21, .31, .31])})
    def fake(self, lumi=None):
        return 100.
    s.nevents_after_preselection = fake
    assert s.nevents_after_bdt(None) == 100.
    assert s.nevents_after_bdt(.2) == 100.*3/5.
    np.testing.assert_array_equal(s.nevents_after_bdt(np.array([.1, .2, .3])), 100.*np.array([5, 3, 2])/5.)
