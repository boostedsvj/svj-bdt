import os, os.path as osp, uuid, multiprocessing, shutil, logging
from time import strftime, perf_counter
from contextlib import contextmanager

import numpy as np
import xgboost as xgb
import seutils
import uptools
import warnings
uptools.logger.setLevel(logging.WARNING)

from .dataset import TriggerEvaluator, preselection, get_subl, get_ak4_subl, get_ak4_lead, calculate_mt_rt, CutFlowColumn, is_array, ttstitch_selection, calculate_mass, get_zprime
from .utils import try_import_ROOT, catchtime


def get_scores(rootfile, model, dataset_name=''):
    '''
    Worker function that reads a single rootfile and returns the
    bdt score, and a few event-level variables to be potentially used
    for histogramming.

    Only uses events that pass the preselection.
    '''
    X = []
    X_histogram = []
    cutflow = CutFlowColumn()
    trigger_evaluator = TriggerEvaluator(uptools.format_rootfiles(rootfile)[0])
    with catchtime() as t: # Measure how long it took
        try:
            for event in uptools.iter_events(rootfile):
                cutflow.plus_one('total')
                if not ttstitch_selection(event, dataset_name, cutflow): continue
                if not preselection(event, cutflow, trigger_evaluator=trigger_evaluator): continue
                # run on data
                #if event[b'RunNum'] < 319077: continue # for Run2018B PostHEM
                #if event[b'RunNum'] >= 319077: continue # for Run2018B PreHEM
                subl = get_subl(event)
                subl.mass = calculate_mass(subl)
                met = event[b'MET']
                metphi = event[b'METPhi']
                mt, rt = calculate_mt_rt(subl, event[b'MET'], event[b'METPhi'])
                # adding leading jets info
                lead_pt = event[b'JetsAK15.fCoordinates.fPt'][0]
                lead_phi = event[b'JetsAK15.fCoordinates.fPhi'][0]
                lead_eta = event[b'JetsAK15.fCoordinates.fEta'][0]
                ak8_pt   = event[b'JetsAK8.fCoordinates.fPt'][0]
                muons = event[b'Muons']
                electrons = event[b'Electrons']
                #print('get zprime')
                zprime = get_zprime(event)
                #z_mass = zprime.mass
                z_pt = zprime.pt
                z_phi = zprime.phi
                z_eta = zprime.eta
                #print('got zprime')
                # adding ak4 jets leading and subleading jets
                ak4_lead = get_ak4_lead(event)
                ak4_subl = get_ak4_subl(event)

                X.append([
                    subl.girth, subl.ptD, subl.axismajor, subl.axisminor,
                    subl.ecfM2b1, subl.ecfD2b1, subl.ecfC2b1, subl.ecfN2b2,
                    subl.metdphi
                    ])
                '''X_histogram.append([
                    mt, rt, subl.pt, subl.energy, met, subl.phi, subl.eta, 
                    subl.mass, metphi, lead_pt, lead_phi, lead_eta,
		    subl.girth, subl.ptD, subl.axismajor, subl.axisminor,
                    subl.ecfM2b1, subl.ecfD2b1, subl.ecfC2b1, subl.ecfN2b2,
                    ak4_lead.eta, ak4_lead.phi, ak4_lead.pt, ak4_subl.eta, ak4_subl.phi, ak4_subl.pt,
		    muons, electrons, ak8_pt])'''

                X_histogram.append([
                    mt, rt, subl.pt, met, subl.phi, subl.eta, 
                    metphi, lead_pt, lead_phi, lead_eta,
                    subl.ecfM2b1, subl.ecfD2b1, subl.ecfC2b1, subl.ecfN2b2,
                    ak4_lead.eta, ak4_lead.phi, ak4_lead.pt, ak4_subl.eta, ak4_subl.phi, ak4_subl.pt,
                    #muons, electrons, ak8_pt])
                    muons, electrons, ak8_pt, z_pt, z_phi, z_eta])
                print(mt.shape, z_pt.shape)
        except IndexError:
            print(f'Problem with {rootfile}; saving {cutflow["preselection"]} good entries')
        except Exception as e:
            print(f'Error processing {rootfile}; Skipping. Error was: ' + repr(e))
    t = t()
    print(f'Processed {cutflow["total"]} events in {t:.3f} seconds ({t/60.:.3f} min)')
    if cutflow['preselection'] == 0:
        print(f'0/{cutflow["total"]} events passed the preselection for {rootfile}')
        '''d = {k : np.array([]) for k in ['score', 'mt', 'rt', 'pt', 'energy', 'met', 'phi', 'eta', 'mass', 'metphi', 'lj_pt', 'lj_phi', 'lj_eta', 
               'subl.girth', 'subl.ptD', 'subl.axismajor', 'subl.axisminor', 'subl.ecfM2b1', 'subl.ecfD2b1', 'subl.ecfC2b1', 'subl.ecfN2b2',
                'ak4_lead.eta', 'ak4_lead.phi', 'ak4_lead.pt', 'ak4_subl.eta', 'ak4_subl.phi', 'ak4_subl.pt', 'muons', 'electrons', 'ak8_pt']}'''

        d = {k : np.array([]) for k in ['score', 'mt', 'rt', 'pt', 'met', 'phi', 'eta', 'metphi', 'lj_pt', 'lj_phi', 'lj_eta', 
               'subl.ecfM2b1', 'subl.ecfD2b1', 'subl.ecfC2b1', 'subl.ecfN2b2',
               'ak4_lead.eta', 'ak4_lead.phi', 'ak4_lead.pt', 'ak4_subl.eta', 'ak4_subl.phi', 'ak4_subl.pt', 'muons', 'electrons', 'ak8_pt',
               'z_pt', 'z_phi', 'z_eta']}

        d.update(**cutflow.counts)
        d['wtime'] = t
        return d
    # Get the bdt scores
    score = model.predict_proba(np.array(X))[:,1]
    # Prepare and dump to file
    X_histogram = np.array(X_histogram)
    '''return dict(
        score=score,
        wtime = t,
        **{key: X_histogram[:,index] for index, key in enumerate(['mt', 'rt', 'pt', 'met', 'phi', 'eta', 'metphi',
                                                                  'lj_pt',  'lj_phi', 'lj_eta', 
                                                                  'subl.ecfM2b1', 'subl.ecfD2b1', 'subl.ecfC2b1', 'subl.ecfN2b2', 
                                                                  'ak4_lead.eta', 'ak4_lead.phi', 'ak4_lead.pt', 'ak4_subl.eta', 'ak4_subl.phi', 'ak4_subl.pt',
                                                                  'muons', 'electrons', 'ak8_pt'])},
        **cutflow.counts
        )'''
    return dict(
        score=score,
        wtime = t,

        **{key: X_histogram[:,index] for index, key in enumerate(['mt', 'rt', 'pt', 'met', 'phi', 'eta', 'metphi',
                                                                  'lj_pt',  'lj_phi', 'lj_eta', 
                                                                  'subl.ecfM2b1', 'subl.ecfD2b1', 'subl.ecfC2b1', 'subl.ecfN2b2', 
                                                                  'ak4_lead.eta', 'ak4_lead.phi', 'ak4_lead.pt', 'ak4_subl.eta', 'ak4_subl.phi', 'ak4_subl.pt',
                                                                  #'muons', 'electrons', 'ak8_pt'])},
                                                                  'muons', 'electrons', 'ak8_pt', 'z_pt', 'z_phi', 'z_eta'])},

        **cutflow.counts
        )


def dump_score_npz(rootfile, model, outfile, dataset_name=''):
    '''    
    Calculates score and dumps events that pass the preselection to a .npz file.
    '''
    d = get_scores(rootfile, model, dataset_name)
    del d['wtime']
    print(f'Dumping {len(d["score"])} events from {rootfile} to {outfile}')
    outdir = osp.dirname(outfile)
    if outdir and not osp.isdir(outdir): os.makedirs(outdir)
    np.savez(outfile, **d)
    return d


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
            # If this value has a length (i.e. is array-like)
            # but it's length is zero, skip it
            try:
                if len(value) == 0: continue
            except TypeError:
                pass
            combined[key].append(value)
    # Make proper np arrays
    for key, values in combined.items():
        try:
            if is_array(values[0]):
                combined[key] = np.concatenate(values)
            else:
                combined[key] = np.array(values).sum()
        except IndexError:
            print(f'Problem with {key=}, {values=}, {type(values)=}; len 0 should not happen here.')
            continue
    return combined


def combine_npzs(npzs):
    """
    Like combine_ds, but instead takes an iterable of npz files
    """
    return combine_ds((np.load(npz) for npz in npzs))

def normalize(h, normalization):
    """
    Takes a ROOT TH1 histogram and sets the integral to `normalization`.
    Includes the under- and overflow bins!
    """
    integral = h.Integral(0, h.GetNbinsX()+1)
    h.Scale(normalization/integral if integral != 0. else 0.)


def make_mt_histogram(name, mt, score=None, threshold=None, mt_binning=None, normalization=None):
    """
    Dumps the mt array to a TH1F. If `score` and `threshold` are supplied, a
    cut score>threshold will be applied.

    Normalization refers to the normalization *before* applying the threshold!
    """
    try_import_ROOT()
    import ROOT
    from array import array
    efficiency = 1.
    if threshold is not None:
        mt = mt[score > threshold]
        efficiency = (score > threshold).sum() / score.shape[0]
        # print(f'{name}: {efficiency=}')
    binning = array('f', MT_BINNING if mt_binning is None else mt_binning)
    h = ROOT.TH1F(name.replace('.','p'), name, len(binning)-1, binning)
    ROOT.SetOwnership(h, False)
    [ h.Fill(x) for x in mt ]
    if normalization is not None:
        integral = h.Integral(0, h.GetNbinsX()+1)
        h.Scale(normalization*efficiency / integral if integral != 0. else 0.)
    return h


def sum_th1s(name, th1s):
    """
    Sums up ROOT TH1s.
    Use reduce and add to create really a new histogram, instead
    of overwriting any existing histogram.
    """
    from operator import add
    from functools import reduce
    h = reduce(add, th1s)
    h.SetNameTitle(name, name)
    return h


def make_summed_histogram(name, ds, norms, threshold=None, mt_binning=None):
    from operator import add
    from functools import reduce
    h = reduce(add, (
        make_mt_histogram(
            str(uuid.uuid4()), d['mt'], d['score'],
            threshold=threshold, normalization=norm, mt_binning=mt_binning
            )
        for d, norm in zip(ds, norms)
        ))
    h.SetNameTitle(name, name)
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
# For local multiprocessing running
# Not recommended, way too slow

def dump_score_npz_worker(input):
    '''    
    Like dump_score_npz but takes a single tuple as input.
    To be used in dump_score_npzs_mp.
    '''
    dump_score_npz(*input)


def dump_score_npzs_mp(model, rootfiles, outfile, n_threads=12, keep_tmp_files=False):
    '''
    Entrypoint to read a list of rootfiles, locally compute the BDT scores, and combine
    it all in a single .npz file.
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
    pool.map(dump_score_npz_worker, data)
    pool.close()
    pool.join()
    # Combine the tmpdir/<unique id>.npz --> outfile and remove tmp files
    combine_npzs(tmpdir, outfile)
    if not keep_tmp_files:
        print(f'Removing {tmpdir}')
        shutil.rmtree(tmpdir)


def combine_ds_with_weights(ds, weights):
    """
    Combines several dicts into a single dict, with weights.

    WARNING: THIS THROWS AWAY PERFECTLY GOOD EVENTS!

    This function should be used only for studying things, not actual
    production of histograms for fitting purposes.
    """
    if len(ds) != len(weights): raise ValueError('len ds != len weights')
    counts = [ len(d['score']) for d in ds ]
    optimal_counts = optimal_count(counts, weights)
    print('Counts:')
    for i, (count, opt_count) in enumerate(zip(counts, optimal_counts)):
        print(f'{i} : {count:8} available, using {opt_count}')
    # Combine from an iterator with the dicts cut to size
    return combine_ds(( shrink_dict(d, opt_count) for d, opt_count in zip(ds, optimal_counts) ))

def shrink_dict(d, n):
    """
    Slices all values that are arrays in d up to :n.
    Integer counts are reduced by the fraction n/len(d)
    """
    len_d = len(d['score']) # Just pick an array key that is always there
    frac = min(float(n/len_d), 1.)
    return { k : v[:n] if v.shape else frac*v for k, v in d.items()}
