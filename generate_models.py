import numpy as np
import pandas as pd
import joblib
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
                                       LogisticRegression('l1'))

array_clfs['ERPCov'] = make_pipeline(ERPCovariances(svd=16, estimator='oas'),
                                     TangentSpace('logdet'),
                                     LogisticRegression('l1'))

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
columns = ['p1', 'p2', 'p3', 'p4']
res_acc = pd.DataFrame(index=index, columns=columns)
res_auc = pd.DataFrame(index=index, columns=columns)


for p in np.unique(patients):

    clfs = deepcopy(array_clfs.values())

    ix = patients==p
    eeg_data = np.float64(dataframe1.loc[ix].values[:,1:-2].T)
    events   = np.int32(dataframe1.Stimulus_Type.loc[ix].values)
    stim_ID  = np.int32(dataframe1.Stimulus_ID.loc[ix].values)
    events[events==101] = 0
    picks = (eeg_data!=-999999).mean(1)

    X, y, st_id = epoch_data(eeg_data[picks==1], events, stim_ID, tmin=0.099, tmax=0.399)

    for clf in clfs:
        clf.fit(X, y)

    joblib.dump(clfs, 'models/models/classifiers_%s.pkl' % p)
    print("Subject %s Done !" % p)
