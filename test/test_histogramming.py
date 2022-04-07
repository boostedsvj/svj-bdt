import pytest
import bdtcode.dataset as dataset
import bdtcode.histogramming as H
import numpy as np
from array import array


def test_combine_ds():
    a = { 'array': np.ones(1), 'scalar': 1}
    b = { 'array': np.ones(2), 'scalar': 2}
    c = H.combine_ds([a, b])
    assert c['scalar'] == 3
    np.testing.assert_equal(c['array'], np.ones(3))


def test_optimal_count():
    counts = np.array([ 100, 200, 150 ])
    np.testing.assert_almost_equal(
        optimal_count(counts, [1./3, 1./3, 1./3]),
        np.array([ 100, 100, 100 ])
        )
    np.testing.assert_almost_equal(
        optimal_count(counts, [.2, .4, .4]),
        np.array([ 75, 150, 150 ])
        )
    np.testing.assert_almost_equal(
        optimal_count(counts, counts/counts.sum()),
        counts
        )
    np.testing.assert_almost_equal(
        optimal_count(
            [18307, 104484, 366352, 242363, 163624],
            [136.52, 278.51, 150.96, 26.24, 7.49]
            ),
        [18307, 37347, 20243, 3518, 1004]
        )
    print('Succeeded')


# def test_sum_hists():
#     import ROOT
#     h1 = ROOT.TH1F('h1', 'h1', 10, array('f', np.linspace(200., 400., 11)))
#     h2 = ROOT.TH1F('h2', 'h2', 10, array('f', np.linspace(200., 400., 11)))
#     h1.Fill(250.)
#     h1.Fill(300.)
#     h2.Fill(250.)
#     h2.Fill(350.)
#     def printh(h):
#         contents = []
#         for i in range(h.GetNbinsX()):
#             contents.append(h.GetBinContent(i+1))
#         print(np.array(contents))
#     printh(h1)
#     printh(h2)
#     printh(h1+h2)
#     from operator import add
#     from functools import reduce
#     h_sum = reduce(add, [h1, h2])
#     printh(h_sum)
#     printh(h1)
#     printh(h2)


# # ________________________________________________________
# # Some tests

# def test_get_scores():
#     model = xgb.XGBClassifier()
#     model.load_model('/Users/klijnsma/work/svj/bdt/svjbdt_Aug02.json')
#     rootfile = 'TREEMAKER_genjetpt375_Jul21_mz250_mdark10_rinv0.337.root'
#     d = get_scores(rootfile, model)
#     import pprint
#     pprint.pprint(d, sort_dicts=False)

# def test_dump_score_npz_worker():
#     model = xgb.XGBClassifier()
#     model.load_model('/Users/klijnsma/work/svj/bdt/svjbdt_Aug02.json')
#     rootfile = 'TREEMAKER_genjetpt375_Jul21_mz250_mdark10_rinv0.337.root'
#     dump_score_npz_worker((rootfile, model, 'out.npz'))

# def test_dump_score_npzs_mp():
#     model = xgb.XGBClassifier()
#     model.load_model('/Users/klijnsma/work/svj/bdt/svjbdt_Aug02.json')
#     rootfiles = seutils.ls_wildcard(
#         'root://cmseos.fnal.gov//store/user/lpcdarkqcd/MCSamples_Summer21/TreeMaker'
#         '/genjetpt375_mz250_mdark10_rinv0.3/*.root'
#         )
#     outfile = 'mz250_mdark10_rinv0p3.npz'
#     # dump_score_npz_worker((rootfiles[1], model, 'out.npz'))
#     dump_score_npzs_mp(model, rootfiles, outfile)
