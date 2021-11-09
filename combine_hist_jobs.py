"""# submit
htcondor('request_memory', '4096MB')
import seutils, os.path as osp, itertools

print('Compiling list of rootfiles...')
bkg_rootfiles = [ seutils.ls_wildcard(d + '/*/*.root') for d in [
    # ttjets
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BKG/bkg_ttjetsAug04_year2018',
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BKG/bkg_ttjetsAug18_year2018',
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BKG/bkg_ttjetsAug25_year2018',
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BKG/bkg_ttjetsSep28_year2018',
    # qcd
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BKG/bkg_May04_year2018',
    # wzjets
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BKG/bkg_wjetstolnuAug13_year2018',
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BKG/bkg_wjetstolnuAug26_year2018',
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BKG/bkg_zjetstonunuAug13_year2018',
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BKG/bkg_zjetstonunuAug26_year2018',
    ]]
bkg_rootfiles = list(itertools.chain.from_iterable(bkg_rootfiles))

# mz 250, 300, 350
sig_rootfiles = [ seutils.ls_wildcard(d + '/*.root') for d in [
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/MCSamples_Summer21/TreeMaker/genjetpt375_mz250_mdark10_rinv0.3',
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/MCSamples_Summer21/TreeMaker/genjetpt375_mz300_mdark10_rinv0.3',
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/MCSamples_Summer21/TreeMaker/genjetpt375_mz350_mdark10_rinv0.3',
    # From Long:
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/MCSamples_Sept28/TreeMaker/genjetpt375_mz230_mdark10_rinv0.3',
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/MCSamples_Sept28/TreeMaker/genjetpt375_mz270_mdark10_rinv0.3',
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/MCSamples_Sept28/TreeMaker/genjetpt375_mz330_mdark10_rinv0.3',
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/MCSamples_Sept28/TreeMaker/genjetpt375_mz400_mdark10_rinv0.3',
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/MCSamples_Sept28/TreeMaker/genjetpt375_mz450_mdark10_rinv0.3',
    ]]
sig_rootfiles = list(itertools.chain.from_iterable(sig_rootfiles))

# bdt_json = 'svjbdt_Sep22_fromsara_3masspoints_qcdttjets.json'
bdt_json = 'svjbdt_Oct11_5masspoints_qcdttjets.json'

def submit_chunk(chunk):
    submit(
        rootfiles=chunk,
        bdt_json=bdt_json,
        run_env='condapack:root://cmseos.fnal.gov//store/user/klijnsma/conda-svj-bdt.tar.gz',
        transfer_files=['combine_hists.py', 'dataset.py', bdt_json],
        )

for chunk in qondor.utils.chunkify(bkg_rootfiles, chunksize=20):
    submit_chunk(chunk)

for chunk in qondor.utils.chunkify(sig_rootfiles, chunksize=50):
    submit_chunk(chunk)
"""# endsubmit

import qondor, seutils
from combine_hists import dump_score_npz
import xgboost as xgb
import os.path as osp, os

model = xgb.XGBClassifier()
model.load_model(qondor.scope.bdt_json)

for rootfile in qondor.scope.rootfiles:
    try:
        seutils.cp(rootfile, 'in.root')
        dump_score_npz('in.root', model, 'out.npz')
        outfile = (
            'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/postbdt_npzs/Oct12/'
            + '/'.join(rootfile.split('/')[-3:]).replace('.root', '.npz')
            )
        seutils.cp('out.npz', outfile)

    except Exception as e:
        print('Failed for rootfile ' + rootfile + ':')
        print(e)
        
    finally:
        if osp.isfile('out.npz'): os.remove('out.npz')
        if osp.isfile('in.root'): os.remove('in.root')
