import numpy as np
import itertools
from contextlib import contextmanager

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