import xgboost as xgb
import itertools
from contextlib import contextmanager

def get_model(modeljson):
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
    if transpose: table = list(zip(*table))
    col_widths = [ max(map(len, column)) for column in zip(*table) ]
    return row_sep.join(
        col_sep.join(f'{col:{w}s}' for col, w in zip(row, col_widths)) for row in table
        )

def print_table(*args, **kwargs):
    print(format_table(*args, **kwargs))
