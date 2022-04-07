"""# submit
htcondor('request_memory', '4096MB')
import seutils, os.path as osp, itertools, glob
seutils.MAX_RECURSION_DEPTH = 100

bkg_rootfiles = seutils.ls_wildcard('root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/umd/BKG/bkg_*year2018/*/*.root')
sig_rootfiles = seutils.ls_wildcard('root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/ntuples_Summer21_Sept28_hadd/*.root')

bdt_json = 'svjbdt_girthreweight_Jan06.json'
bdtcode_files = glob.glob('../bdtcode/*.py') + glob.glob('../bdtcode/*.npz')

def submit_chunk(chunk):
    submit(
        rootfiles=chunk,
        bdt_json=bdt_json,
        bdtcode_files=bdtcode_files,
        run_env='condapack:root://cmseos.fnal.gov//store/user/klijnsma/conda-svj-bdt.tar.gz',
        transfer_files=bdtcode_files + [bdt_json],
        )

for chunk in qondor.utils.chunkify(bkg_rootfiles, chunksize=20):
    submit_chunk(chunk)

for rootfile in sig_rootfiles:
    submit_chunk([rootfile])

"""# endsubmit

import os.path as osp, os
import qondor, seutils

# Put all the bdtcode python files back in a directory called 'bdtcode'
os.makedirs('bdtcode/')
for f in qondor.scope.bdtcode_files:
    f = osp.basename(f)
    os.rename(f, 'bdtcode/'+f)
import bdtcode

model = bdtcode.utils.get_model(qondor.scope.bdt_json)

for rootfile in qondor.scope.rootfiles:
    try:
        seutils.cp(rootfile, 'in.root')
        bdtcode.histogramming.dump_score_npz(['in.root'], model, 'out.npz', dataset_name=rootfile)
        outfile = (
            'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/postbdt_npzs/stitchedttjets_Mar31/'
            + '/'.join(rootfile.split('/')[-3:]).replace('.root', '.npz')
            )
        seutils.cp('out.npz', outfile)

    except Exception as e:
        print('Failed for rootfile ' + rootfile + ':')
        print(e)
        
    finally:
        if osp.isfile('out.npz'): os.remove('out.npz')
        if osp.isfile('in.root'): os.remove('in.root')
