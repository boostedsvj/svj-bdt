import glob, os.path as osp
import numpy as np
from time import strftime
import bdtcode

np.random.seed(1001)


class NPZ:
    def __init__(self, npzfile):
        self.npzfile = npzfile
        self.X = None

    @property
    def shape(self):
        self.read()
        return self.X.shape

    @property
    def n(self):
        return self.shape[0]

    @property
    def is_read(self):
        return self.X is not None

    def read(self):
        if not self.is_read: self.X = np.load(self.npzfile)['X']
    
    def read_once(self):
        if self.is_read: return self.X
        return np.load(self.npzfile)['X']

    @property
    def sample(self):
        return osp.basename(osp.dirname(self.npzfile))


def get_n_events(npzs, n):
    n_todo = n
    X = []
    for npz in npzs:
        X.append(npz.read_once()[:n_todo])
        n_todo -= npz.n
        if n_todo <= 0: break
    else:
        bdtcode.logger.warning(f'Requested {n} events, but only got {n-n_todo} out of {len(npzs)} npzs')
    return np.vstack(X)


def get_bkg_X_weighted(n_target=10000):
    """
    Returns a single feature vector, (n_target_events x n_features),
    which has events occuring according to the cross section.
    """
    npzs = [ NPZ(n) for n in glob.iglob('trainingnpzs/*QCD*/*.npz')]
    samples = list(set(npz.sample for npz in npzs))
    samples.remove('QCD_Pt_170to300') # Very few events and skews rest of unweighted sample
    crosssections = np.array([ bdtcode.crosssections.label_to_xs(s) for s in samples ])
    n_target_per_sample = (n_target * crosssections / crosssections.sum()).astype(np.int)
    X = []
    for sample, n_target_sample in zip(samples, n_target_per_sample):
        npzs_sample = [npz for npz in npzs if npz.sample == sample]
        X.append(get_n_events(npzs_sample, n_target_sample))
    X = np.vstack(X)
    bdtcode.logger.info(f'Built vector of {X.shape} from {len(npzs)} npzfiles')
    for sample, n_target_sample in zip(samples, n_target_per_sample):
        bdtcode.logger.info(f'  {sample=:20s} {n_target_sample=}')
    return X

def get_bkg_X(n_target=100000):
    npzs = [ NPZ(n) for n in glob.iglob('trainingnpzs/*QCD*/*.npz')]
    samples = list(set(npz.sample for npz in npzs))
    samples.remove('QCD_Pt_170to300') # Very few events and skews rest of unweighted sample
    crosssections = np.array([ bdtcode.crosssections.label_to_xs(s) for s in samples ])
    n = int(n_target / len(samples))
    X = []
    weights = []
    for sample in samples:
        X_sample = get_n_events([npz for npz in npzs if npz.sample == sample], n)
        weights.append(bdtcode.crosssections.label_to_xs(sample) * np.ones(X_sample.shape[0]))
        X.append(X_sample)
    X = np.vstack(X)
    weights = np.concatenate(weights)
    bdtcode.logger.info(f'Built vector of {X.shape} from {len(npzs)} npzfiles')
    for sample in samples:
        bdtcode.logger.info(f'  {sample=:20s} {n=}')
    return X, weights


def get_sig_X(mz):
    return NPZ(f'trainingnpzs/mz{mz}.npz').read_once()


def get_X(n_bkg_target=200000):
    def make_y(X, val=1):
        return np.ones(X.shape[0], dtype=int) * int(val)

    X_mz250 = get_sig_X(250)
    y_mz250 = make_y(X_mz250)
    X_mz350 = get_sig_X(350)
    y_mz350 = make_y(X_mz350)
    X_mz450 = get_sig_X(450)
    y_mz450 = make_y(X_mz450)

    X_bkg, weights_bkg = get_bkg_X(n_bkg_target)
    y_bkg = make_y(X_bkg, 0)

    weights_bkg /= weights_bkg.sum()
    weights_sig = np.ones(X_mz250.shape[0] + X_mz350.shape[0] + X_mz450.shape[0])
    weights_sig /= weights_sig.sum()

    X = np.vstack((X_mz250, X_mz350, X_mz450, X_bkg))
    y = np.concatenate((y_mz250, y_mz350, y_mz450, y_bkg))
    weights = np.concatenate((weights_sig, weights_bkg))
    weights *= weights.shape[0]

    bdtcode.logger.info(
        f'Created combined bkg/sig vector X={X.shape}, y={y.shape}, weights={weights.shape}; '
        f'{y.sum()=}, {weights[y==1].sum()=}, {weights[y==0].sum()=}'
        )

    return X, y, weights


def train_entrypoint():
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc

    X, y, weights = get_X()
    (
        X_train, X_test,
        y_train, y_test,
        weights_train, weights_test
        ) = train_test_split(X, y, weights, test_size=.2)

    model = xgb.XGBClassifier(
        eta=.05,
        max_depth=4,
        n_estimators=850,
        use_label_encoder=False
        )

    model.fit(X_train, y_train, sample_weight=weights_train)
    model_outfile = strftime('svjbdt_%b%d.json')
    model.save_model(model_outfile)
    bdtcode.logger.info(f'Saved trained model to {model_outfile}')

    # Little bit of validation on the test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    
    bdtcode.logger.info(
        f'Trained BDT; Confusion matrix (first counts, then weighted counts):'
        f'\n{confusion_matrix(y_test, y_pred)}\n{confusion_matrix(y_test, y_pred, sample_weight=weights_test)}'
        )
    bdtcode.logger.info(f'AUC: {roc_auc_score(y_test, y_prob)}')

    import matplotlib.pyplot as plt
    eff_bkg, eff_sig, cuts = roc_curve(y_test, y_prob)

    fig = plt.figure(figsize=(6,6))
    ax = fig.gca()
    ax.plot([0,1], [0,1], c='gray')
    ax.plot(eff_bkg, eff_sig)
    ax.set_ylim(0., 1.)
    ax.set_xlim(0., 1.)
    ax.set_xlabel('Bkg efficiency', fontsize=20)
    ax.set_ylabel('Sig efficiency', fontsize=20)
    plt.savefig('roc.png')
    bdtcode.logger.info('Created roc.png with a basic ROC curve')




    











