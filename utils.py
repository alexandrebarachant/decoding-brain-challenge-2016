import numpy as np
from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin


def epoch_data(data, events, stim_ID, tmin=-.2, tmax=0.399):
    """Epoch data."""
    ix_events = np.where(np.diff(np.int32(events > 0)) == 1)[0] + 1

    ix_min = int(tmin*1000)
    ix_max = int(tmax*1000)
    nsamp = ix_max - ix_min
    X = np.zeros((len(ix_events), data.shape[0], nsamp))
    y = np.int32(events[ix_events] > 50)
    st_id = np.int32(stim_ID[ix_events])

    for i, ix in enumerate(ix_events):
        sl = slice((ix + ix_min), (ix + ix_max))
        tmp = data[:, sl]
        X[i, :, 0:tmp.shape[1]] = tmp
    return X, y, st_id


class DownSampler(BaseEstimator, TransformerMixin):
    """Downsample transformer"""

    def __init__(self, factor=4):
        """Init."""
        self.factor = factor

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[:, :, ::self.factor]


class EpochsVectorizer(BaseEstimator, TransformerMixin):
    """Vectorize epochs."""

    def __init__(self):
        """Init."""

    def fit(self, X, y):
        return self

    def transform(self, X):
        X2 = np.array([x.flatten() for x in X])
        return X2


class CospBoostingClassifier(BaseEstimator, TransformerMixin):
    """Cospectral matrice bagging."""

    def __init__(self, baseclf):
        """Init."""
        self.baseclf = baseclf

    def fit(self, X, y):
        self.clfs_ = []
        for i in range(X.shape[-1]):
            clf = deepcopy(self.baseclf)
            self.clfs_.append(clf.fit(X[:, :, :, i], y))
        return self

    def predict_proba(self, X):
        proba = []
        for i in range(X.shape[-1]):
            proba.append(self.clfs_[i].predict_proba(X[:, :, :, i]))
        proba = np.mean(proba, axis=0)
        return proba

    def transform(self, X):
        proba = []
        for i in range(X.shape[-1]):
            proba.append(self.clfs_[i].predict_proba(X[:, :, :, i]))
        proba = np.concatenate(proba, 1)
        return proba
