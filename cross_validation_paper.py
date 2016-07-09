import numpy as np
import pandas as pd
import mne

from scipy.io import loadmat
from glob import glob
from collections import OrderedDict
from copy import deepcopy

from pyriemann.estimation import (XdawnCovariances, HankelCovariances,
                                  CospCovariances, ERPCovariances)
from pyriemann.spatialfilters import Xdawn, CSP
from pyriemann.tangentspace import TangentSpace
from pyriemann.channelselection import ElectrodeSelection

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score

from utils import (DownSampler, EpochsVectorizer, CospBoostingClassifier,
                   epoch_data)

dataframe1 = pd.read_csv('ecog_train_with_labels.csv')

array_clfs = OrderedDict()

# ERPs models
array_clfs['XdawnCov'] = make_pipeline(XdawnCovariances(6, estimator='oas'),
                                       TangentSpace('riemann'),
                                       LogisticRegression('l2'))

array_clfs['Xdawn'] = make_pipeline(Xdawn(12, estimator='oas'),
                                    DownSampler(5),
                                    EpochsVectorizer(),
                                    LogisticRegression('l2'))

# Induced activity models

baseclf = make_pipeline(ElectrodeSelection(10, metric=dict(mean='logeuclid',
                                                           distance='riemann')),
                        TangentSpace('riemann'),
                        LogisticRegression('l1'))

array_clfs['Cosp'] = make_pipeline(CospCovariances(fs=1000, window=32,
                                                   overlap=0.95, fmax=300,
                                                   fmin=1),
                                   CospBoostingClassifier(baseclf))

array_clfs['HankelCov'] = make_pipeline(DownSampler(2),
                                        HankelCovariances(delays=[2, 4, 8, 12, 16], estimator='oas'),
                                        TangentSpace('logeuclid'),
                                        LogisticRegression('l1'))

array_clfs['CSSP'] = make_pipeline(HankelCovariances(delays=[2, 4, 8, 12, 16],
                                                     estimator='oas'),
                                   CSP(30),
                                   LogisticRegression('l1'))

patients = dataframe1.PatientID.values

index = array_clfs.keys() + ['Ensemble']
columns = ['ca', 'de', 'fp', 'ja', 'mv', 'wc', 'zt']
res_acc = pd.DataFrame(index=index, columns=columns)
res_auc = pd.DataFrame(index=index, columns=columns)

fnames = glob('./fhpred/data/*/*.mat')

for fname in fnames:
    data = loadmat(fname)
    p = fname[-18:-16]
    clfs = deepcopy(array_clfs)

    Nc = data['data'].shape[1]

    ch_names = [str(i) for i in range(Nc)] + ['stim']
    ch_type = ['eeg']*Nc + ['stim']

    info = mne.create_info(ch_names, sfreq=1000, ch_types=ch_type)
    mne_data = data['data']
    mne_data = np.c_[mne_data, data['stim']].T

    mne_data[-1, mne_data[-1] == 101] = 0
    raw = mne.io.RawArray(mne_data, info)

    events = (mne.find_events(raw, verbose=False))
    events[:, -1] = events[:, -1] > 50

    picks = mne.pick_types(raw.info, eeg=True)[::1]

    epochs = mne.Epochs(raw, events, {'c1': 0, 'c2': 1}, tmin=0.099,
                        tmax=0.399, add_eeg_ref=False, preload=True,
                        baseline=None, verbose=False, picks=picks)

    X = epochs._data[:, :-1]
    y = epochs.events[:, -1]

    preds = OrderedDict()

    cv = KFold(len(y), 3)
    for clf in clfs:
        preds[clf] = np.zeros((len(y), 2))
        acc_tmp = []
        auc_tmp = []
        for train, test in cv:
            clfs[clf].fit(X[train], y[train])
            preds[clf][test] = (clfs[clf].predict_proba(X[test]))

            yte = np.argmax(preds[clf][test], 1)
            acc_tmp.append(100 * np.mean(yte == y[test]))
            auc_tmp.append(roc_auc_score(y[test], preds[clf][test, 1]))
        res_acc.loc[clf, p] = np.mean(acc_tmp)
        res_auc.loc[clf, p] = np.mean(auc_tmp)

    yte = np.argmax(np.mean(preds.values(), 0), 1)

    acc_tmp = []
    auc_tmp = []
    for train, test in cv:
        acc_tmp.append(100*np.mean(yte[test] == y[test]))
        pr = np.mean(preds.values(), 0)[test, 1]
        auc_tmp.append(roc_auc_score(y[test], pr))

    res_acc.loc['Ensemble', p] = np.mean(acc_tmp)
    res_auc.loc['Ensemble', p] = np.mean(auc_tmp)
    print res_acc
    print res_auc

res_acc.loc[:, 'Average'] = res_acc.mean(1)
res_auc.loc[:, 'Average'] = res_auc.mean(1)

# save filters
res_acc.to_csv('Results_accuracy_CV_paper.csv')
res_auc.to_csv('Results_AUC_CV_paper.csv')
