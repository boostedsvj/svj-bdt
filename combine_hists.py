import os, os.path as osp, uuid, multiprocessing, shutil, logging
from time import strftime

import numpy as np
import xgboost as xgb
import seutils
import uptools
uptools.logger.setLevel(logging.WARNING)

from dataset import preselection, get_subl, calculate_mt_rt


def get_hist(rootfile, model, outfile):
    '''
    Worker function that reads a single rootfile and dumps events that
    pass the preselection to a .npz file.
    To be used in get_hist_mp.
    '''
    X = []
    X_histogram = []
    n_total = 0
    n_presel = 0
    try:
        for event in uptools.iter_events(rootfile):
            n_total += 1
            if not preselection(event): continue
            n_presel += 1
            subl = get_subl(event)
            mt, rt = calculate_mt_rt(subl, event[b'MET'], event[b'METPhi'])
            X.append([
                subl.girth, subl.ptD, subl.axismajor, subl.axisminor,
                subl.ecfM2b1, subl.ecfD2b1, subl.ecfC2b1, subl.ecfN2b2,
                subl.metdphi
                ])
            X_histogram.append([mt, rt, subl.pt, subl.energy])
    except IndexError:
        print(f'Problem with {rootfile}; saving {n_presel} good entries')
    except:
        print(f'Error processing {rootfile}; Skipping')
    if n_presel == 0:
        print(f'0/{n_total} events passed the preselection for {rootfile}')
        return
    # Get the bdt scores
    score = model.predict_proba(np.array(X))[:,1]
    # Prepare and dump to file
    print(f'Dumping {len(X)} events from {rootfile} to {outfile}')
    X_histogram = np.array(X_histogram)
    outdir = osp.dirname(outfile)
    if outdir and not osp.isdir(outdir): os.makedirs(outdir)
    np.savez(
        outfile,
        score=score,
        **{key: X_histogram[:,index] for index, key in enumerate(['mt', 'rt', 'pt', 'energy'])},
        n_total=n_total,
        n_presel=n_presel
        )


def get_hist_worker(input):
    '''Alias for get_hist that takes only 1 iterable (for mp)'''
    get_hist(*input)


def get_hist_mp(model, rootfiles, outfile, n_threads=12, keep_tmp_files=False):
    '''
    Entrypoint to read a list of rootfiles, compute the BDT scores, and combine it all
    in a single .npz file.
    Uses multiprocessing to speed things up.
    '''
    print(f'Processing {len(rootfiles)} rootfiles to {outfile}')
    tmpdir = strftime(f'TMP_%b%d_%H%M%S_{outfile}')
    os.makedirs(tmpdir)
    # Prepare input data
    data = []
    for rootfile in rootfiles:
        data.append([ rootfile, model, osp.join(tmpdir, str(uuid.uuid4())+'.npz') ])
    # Process data in multiprocessing pool
    # Every thread will dump data into tmpdir/<unique id>.npz
    pool = multiprocessing.Pool(n_threads)
    pool.map(get_hist_worker, data)
    pool.close()
    pool.join()
    # Combine the tmpdir/<unique id>.npz --> outfile and remove tmp files
    combine_npzs(tmpdir, outfile)
    #combine_npzs(tmpdir)
    if not keep_tmp_files:
        print(f'Removing {tmpdir}')
        shutil.rmtree(tmpdir)


def combine_ds(ds):
    """
    Takes a iterable of dict-like objects with the same keys,
    and combines them in a single dict.

    Arrays will be concatenated, scalar values will be summed
    """
    combined = {}
    for d in ds:
        for key, value in d.items():
            if not key in combined: combined[key] = []
            if value.shape and len(value) == 0: continue
            combined[key].append(value)
    # Make proper np arrays
    for key, values in combined.items():
        if len(values[0].shape) == 0:
            combined[key] = np.array(values).sum()
        else:
            combined[key] = np.concatenate(values)
    return combined


def combine_npzs(npzs):
    """
    Like combine_ds, but instead takes an iterable of npz files
    """
    return combine_ds((np.load(npz) for npz in npzs))


def try_import_ROOT():
    try:
        import ROOT
    except ImportError:
        print(
            'ROOT is required to be installed for this operation. Run:\n'
            'conda install -c conda-forge root'
            )
        raise


def combine_ds_with_weights(ds, weights):
    """
    Combines several dicts into a single dict, with weights
    """
    if len(ds) != len(weights): raise ValueError('len ds != len weights')
    counts = [ len(d['score']) for d in ds ]
    #counts = [ len(d['mt']) for d in ds]
    optimal_counts = optimal_count(counts, weights)

    # Purely for debugging:
    print('Counts:')
    for i, (count, opt_count) in enumerate(zip(counts, optimal_counts)):
        print(f'{i} : {count:8} available, using {opt_count}')

    # Combine from an iterator with the dicts cut to size
    return combine_ds(( shrink_dict(d, opt_count) for d, opt_count in zip(ds, optimal_counts) ))


def shrink_dict(d, n):
    """
    Slices all values that are arrays in d up to :n
    """
    return { k : v[:n] if v.shape else v for k, v in d.items()}


# Defauly mt binning
#MT_BINNING = [160.+8.*i for i in range(44)]
MT_BINNING = [190.+8.*i for i in range(40)]
#MT_BINNING = [0.+5.*i for i in range(101)]
RT_BINNING = [1.+0.025*i for i in range(100)]

def make_mt_histogram(name, mt, score=None, threshold=None, mt_binning=None):
    """
    Dumps the mt array to a TH1F. If `score` and `threshold` are supplied, a
    cut score>threshold will be applied.
    """
    try_import_ROOT()
    import ROOT
    from array import array
    if threshold is not None: mt = mt[score > threshold]
    binning = array('f', MT_BINNING if mt_binning is None else mt_binning)
    h = ROOT.TH1F(name, name, len(binning)-1, binning)
    ROOT.SetOwnership(h, False)
    [ h.Fill(x) for x in mt ]
    return h

def make_rtvsmt_histogram(name, mt, rt, score=None, threshold=None, mt_binning=None, rt_binning=None):
    """
    Dumps the rtvsmt array to a TH2F to study sculpting. If `score` and `threshold` are supplied, a
    cut score>threshold will be applied.
    """
    try_import_ROOT()
    import ROOT
    from array import array
    if threshold is not None: mt = mt[score > threshold]
    binning = array('f', MT_BINNING if mt_binning is None else mt_binning)
    rt_binning = array('f', RT_BINNING if rt_binning is None else rt_binning)
    h2d = ROOT.TH2F(name, name, len(rt_binning)-1, rt_binning, len(binning)-1, binning)
    ROOT.SetOwnership(h2d, False)
    [ h2d.Fill(x,y) for x, y in zip(mt, rt) ]
    return h2d


def optimal_count(counts, weights):
    """
    Given an array of counts and an array of desired weights (e.g. cross sections),
    find the highest possible number of events without underrepresenting any bin
    """
    # Normalize weights to 1.
    weights = np.array(weights) / sum(weights)

    # Compute event fractions
    counts = np.array(counts)
    n_total = np.sum(counts)
    fractions = counts / n_total

    imbalance = weights / fractions
    i_max_imbalance = np.argmax(imbalance)
    max_imbalance = imbalance[i_max_imbalance]

    if max_imbalance == 1.:
        # Counts array is exactly balanced; Don't do anything
        return counts

    n_target = counts[i_max_imbalance] / weights[i_max_imbalance]
    optimal_counts = (weights * n_target).astype(np.int32)
    return optimal_counts


# ________________________________________________________
# Some tests

def test_get_hist_worker():
    model = xgb.XGBClassifier()
    model.load_model('/Users/klijnsma/work/svj/bdt/svjbdt_Aug02.json')
    rootfile = 'TREEMAKER_genjetpt375_Jul21_mz250_mdark10_rinv0.337.root'
    get_hist_worker((rootfile, model, 'out.npz'))

def test_get_hist_mp():
    model = xgb.XGBClassifier()
    #model.load_model('/Users/klijnsma/work/svj/bdt/svjbdt_Aug02.json')
    model.load_model('/home/snabili/data/bdt/svj-bdt/svjbdt_Sep22.json')
    rootfiles = seutils.ls_wildcard(
        'root://cmseos.fnal.gov//store/user/lpcdarkqcd/MCSamples_Summer21/TreeMaker'
        '/genjetpt375_mz250_mdark10_rinv0.3/*.root'
        )
    outfile = 'mz250_mdark10_rinv0p3.npz'
    # get_hist_worker((rootfiles[1], model, 'out.npz'))
    get_hist_mp(model, rootfiles, outfile)

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


if __name__ == '__main__':
  #test_get_hist_mp()
  #make_mt_histogram('/mnt/hadoop/cms/store/user/snabili/BKG/BDT/genjetpt375_mz250_mdark10_rinv0.3/TREEMAKER_genjetpt375_Sep07_mz250_mdark10_rinv0.3271.npz',mt)
  combine_npzs('/mnt/hadoop/cms/store/user/snabili/BKG/BDT/genjetpt375_mz250_mdark10_rinv0.3/')
