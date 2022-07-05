import pytest
import bdtcode
import bdtcode.dataset as dataset
import numpy as np

def test_calc_dphi():
    pi = np.pi
    twopi = 2.*pi
    assert dataset.calc_dphi(2., 1.) == 1.
    assert dataset.calc_dphi(5., 1.) == 4.-twopi
    assert dataset.calc_dphi(1., 5.) == twopi-4.
    assert dataset.calc_dphi(2., 1.+twopi) == 1.
    assert dataset.calc_dphi(2.+twopi, 1.) == 1.
    assert dataset.calc_dphi(2.+2.*twopi, 1.+5.*twopi) == 1.
    assert dataset.calc_dphi(-twopi+2., 1.) == 1.
    assert dataset.calc_dphi(0., 4.) == twopi-4.
    np.testing.assert_equal(
        dataset.calc_dphi(np.array([2., 2.]), np.array([1., 1.])),
        np.array([1., 1.])
        )
    
def test_calc_dr():    
    assert dataset.calc_dr(1., 1., 2., 2.) == np.sqrt(2.)
    assert dataset.calc_dr(-1., -1., -2., -2.) == np.sqrt(2.)
    twos = np.array([2., 2.])
    ones = np.array([1., 1.])
    np.testing.assert_equal(dataset.calc_dr(twos, twos, ones, ones),  np.sqrt(twos))


# The tests below do not really test whether the functions work, but just if they don't crash
# Writing better tests is something for... later

import os.path as osp
testdir = osp.dirname(osp.abspath(__file__))
bdt_json = osp.join(testdir, 'svjbdt_girthreweight_Jan06.json')
sig_rootfile = osp.join(testdir, 'mz450_example_treemaker.root')
bkg_rootfile = osp.join(testdir, 'ttjets_example_treemaker.root')
data_rootfile = osp.join(testdir, 'data_ntuple_example.root')

def download_test_files():
    """Downloads the testfiles if they are not available yet"""
    import seutils
    if not osp.isfile(sig_rootfile):
        seutils.cp(
            'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/other/bdtcode_testfiles/mz450_example_treemaker.root',
            sig_rootfile
            )
    if not osp.isfile(bkg_rootfile):
        seutils.cp(
            'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/other/bdtcode_testfiles/ttjets_example_treemaker.root',
            bkg_rootfile
            )
    if not osp.isfile(bkg_rootfile):
        seutils.cp(
            'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/other/bdtcode_testfiles/data_ntuple_example.root',
            data_rootfile
            )


class FakeModel:
    def predict_proba(self, X, *args, **kwargs):
        return np.random.rand(X.shape[0], 2)


def test_testfiles():
    download_test_files()
    import uptools
    assert uptools.format_rootfiles(sig_rootfile) == [sig_rootfile]
    events = list(e for e in uptools.iter_events([sig_rootfile], nmax=80))
    assert len(events) == 80
    events = list(e for e in uptools.iter_events([bkg_rootfile], nmax=80))
    assert len(events) == 80

def test_ttstitch_mask():
    download_test_files()
    import uptools
    n_pass = 0
    nmax = 100
    for event in uptools.iter_events(bkg_rootfile, nmax=nmax):
        passes = dataset.ttstitch_selection(event, 'TTJets_TuneCP5_13TeV-madgraphMLM-pythia8')
        assert passes == (
            event[b'GenHT'] < 600. and (event[b'NElectrons'] + event[b'NMuons']) == 0
            )
        n_pass += int(passes)
    assert n_pass > 0 and n_pass < nmax

def test_trigger_evaluator():
    download_test_files()
    import uptools
    trigger_evaluator = dataset.TriggerEvaluator(bkg_rootfile)
    n_pass = sum(trigger_evaluator(event) for event in uptools.iter_events(bkg_rootfile, nmax=2000))
    assert n_pass > 0

def test_dump_score_npz():
    download_test_files()
    import bdtcode, pprint
    model = bdtcode.utils.get_model(bdt_json)
    d = bdtcode.histogramming.dump_score_npz(bkg_rootfile, model, 'test.npz', dataset_name='root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/umd/BKG/bkg_ttjetsSep28_year2018/Autumn18.TTJets_TuneCP5_13TeV-madgraphMLM-pythia8/0.root')
    pprint.pprint(d)
    assert len(d['score']) >= 0
    assert d['ttstitch'] > 0 and d['ttstitch'] < d['total']

def test_new_mass_variables():
    download_test_files()
    import uptools
    for event in uptools.iter_events(sig_rootfile):
        if not dataset.preselection(event): continue
        subl = dataset.get_subl(event)
        met = event[b'MET']
        metphi = event[b'METPhi']
        print(dataset.calculate_mass(subl))
        print(dataset.calculate_massmet(subl, met, metphi))
        print(dataset.calculate_massmetpz(subl, met, metphi))
        print(dataset.calculate_massmetpzm(subl, met, metphi))
        return

def test_data_ntuple():
    download_test_files()
    bdtcode.do_ultra_legacy()
    import uptools
    for event in uptools.iter_events(data_rootfile):
        if not dataset.preselection(event): continue
        subl = dataset.get_subl(event)
        met = event[b'MET']
        metphi = event[b'METPhi']
        print(dataset.calculate_mass(subl))
        print(dataset.calculate_massmet(subl, met, metphi))
        print(dataset.calculate_massmetpz(subl, met, metphi))
        print(dataset.calculate_massmetpzm(subl, met, metphi))
        return

def test_apply_bdt():
    download_test_files()
    bdtcode.do_ultra_legacy()
    model = FakeModel()

    d = dataset.apply_bdt(model, sig_rootfile, outfile=None, nmax=20)

    assert 'scores' in d
    assert 'cutflow_titles' in d
    assert 'X' in d

    assert d['X'].shape[0] == d['cutflow_values'][-1]
    assert d['cutflow_values'][0] == 20
    
