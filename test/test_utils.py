import bdtcode.utils as utils
import numpy as np

def test_is_array():
    for thing in [ 1, 1., '1', True, [1] ]:
        assert utils.is_array(thing) is False
    assert utils.is_array(np.array([1.])) is True
    assert utils.is_array(np.array([])) is True


def test_np_load_remote():
    d = utils.np_load_remote('root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/postbdt_npzs/stitchedttjets_Mar29/bkg_ttjetsAug04_year2018/Autumn18.TTJets_TuneCP5_13TeV-madgraphMLM-pythia8/0.npz')
