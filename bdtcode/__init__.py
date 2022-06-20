import logging
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

UL = False
def do_ultra_legacy(flag=True):
    global UL
    UL = flag

def setup_logger(name='bdtcode'):
    if name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.info('Logger %s is already defined', name)
    else:
        fmt = logging.Formatter(
            fmt = (
                '\033[36m%(levelname)7s:%(asctime)s:%(module)s:%(lineno)s\033[0m'
                + ' %(message)s'
                ),
            datefmt='%Y-%m-%d %H:%M:%S'
            )
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger
logger = setup_logger()

def debug(flag=True):
    logger.setLevel(logging.DEBUG if flag else logging.INFO)

def set_mpl_fontsize(small=16, medium=20, large=24):
    import matplotlib.pyplot as plt
    plt.rc('font', size=small)        # controls default text sizes
    plt.rc('axes', titlesize=small)   # fontsize of the axes title
    plt.rc('axes', labelsize=medium)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=small)  # fontsize of the tick labels
    plt.rc('legend', fontsize=small)  # legend fontsize
    plt.rc('figure', titlesize=large) # fontsize of the figure title

from . import crosssections
from . import dataset
from . import histogramming
from . import utils
from . import sample
from . import training