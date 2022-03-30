import numpy as np
import itertools
from contextlib import contextmanager
from time import strftime, perf_counter
import os, os.path as osp

def get_model(modeljson):
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(modeljson)
    return model

def try_import_ROOT():
    try:
        import ROOT
    except ImportError:
        print(
            'ROOT is required to be installed for this operation. Run:\n'
            'conda install -c conda-forge root'
            )
        raise

@contextmanager
def open_root(*args, **kwargs):
    try_import_ROOT()
    import ROOT
    try:
        f = ROOT.TFile.Open(*args, **kwargs)
        yield f
    finally:
        f.Close()

def is_array(a):
    """
    Checks if a thing is an array or maybe a number
    """
    try:
        shape = a.shape
        return len(shape) >= 1
    except AttributeError:
        return False

def flatten(*args):
    return list(itertools.chain.from_iterable(args))

def repeat_interleave(contents, counts):
    for item, count in zip(contents, counts):
        for i in range(count):
            yield item

def format_table(table, col_sep=' ', row_sep='\n', transpose=False):
    def format(s):
        try:
            number_f = float(s)
            number_i = int(s)
            if number_f != number_i:
                return f'{s:.2f}'
            else:
                return f'{s:.0f}'
        except ValueError:
            return str(s)
    table = [ [format(c) for c in row ] for row in table ]
    if transpose: table = _transpose(table)
    col_widths = [ max(map(len, column)) for column in zip(*table) ]
    return row_sep.join(
        col_sep.join(f'{col:{w}s}' for col, w in zip(row, col_widths)) for row in table
        )

def transpose(table):
    """
    Turns rows into columns. Does not check if all rows have the same length!
    Example: [[1,2,3], [4,5,6]] --> [[1,4], [2,5], [3,6]]
    """
    # return list(zip(*table))
    return [list(x) for x in zip(*table)]

def _transpose(*args, **kwargs):
    """Same as `transpose` above"""
    return transpose(*args, **kwargs)

def print_table(*args, **kwargs):
    print(format_table(*args, **kwargs))

def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

def mkdir(dirname):
    """
    Makes dir if it doesn't exist, and fills in time formatting
    """
    dirname = strftime(dirname)
    if not osp.isdir(dirname): os.makedirs(dirname)
    return dirname

class colorwheel_root:

    def __init__(self, colors=[2, 3, 4, 6, 7, 8, 30, 38, 40, 41, 42, 46, 48]):
        self.colors = colors
        self._available_colors = self.colors.copy()

    def __call__(self):
        color = self._available_colors.pop(0)
        if not len(self._available_colors):
            self._available_colors = self.colors.copy()
        return color

def th1_binning_and_values(h):
    """
    Returns the binning and values of the histogram.
    Does not include the overflows.
    """
    n_bins = h.GetNbinsX()
    # GetBinLowEdge of the right overflow bin is the high edge of the actual last bin
    binning = np.array([h.GetBinLowEdge(i) for i in range(1,n_bins+2)])
    values = np.array([h.GetBinContent(i) for i in range(1,n_bins+1)])
    return binning, values

def set_matplotlib_fontsizes(small=10, medium=14, large=18):
    import matplotlib.pyplot as plt
    plt.rc('font', size=small)          # controls default text sizes
    plt.rc('axes', titlesize=small)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
    plt.rc('legend', fontsize=small)    # legend fontsize
    plt.rc('figure', titlesize=large)   # fontsize of the figure title

@contextmanager
def catchtime():
    start = perf_counter()
    yield lambda: perf_counter() - start

def np_load_remote(path, **kwargs):
    """
    Like np.load, but works on remote paths
    """
    from io import StringIO
    import seutils
    contents = seutils.cat(path)
    f = StringIO(contents)
    return np.load(f, **kwargs)
