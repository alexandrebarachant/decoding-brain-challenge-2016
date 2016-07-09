# The script MUST contain a function named azureml_main
# which is the entry point for this module.
#
# The entry point function can contain up to two input arguments:
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame

import numpy as np
import pandas as pd
from pyriemann.estimation import XdawnCovariances, Covariances, HankelCovariances, CospCovariances
from pyriemann.channelselection import ElectrodeSelection
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from pyriemann.spatialfilters import Xdawn, CSP
from sklearn.externals import joblib

from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import KFold
from utils import DownSampler, EpochsVectorizer, CospBoostingClassifier, epoch_data

# this is the hardcoded list of "good" electrodes for each subjects.
# it is just based on the detection of -999999 channels from the training data

picks = {'p1': np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]),
 'p2': np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]),
 'p3': np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]),
 'p4': np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41])}

def azureml_main(dataframe1 = None, dataframe2 = None):

    patients = dataframe1.PatientID.values

    patients_ID = []
    stimulus_ID = []
    prediction_Label = []

    # For each patient, we load the models trained localy
    patients_tot = ['p1', 'p2', 'p3', 'p4']
    clfs = dict()
    for p in patients_tot:
        clfs[p] = joblib.load('.\Script Bundle\models\classifiers_%s.pkl' % p)

    for p in np.unique(patients):

        # Get the data
        ix = patients==p
        eeg_data = np.float64(dataframe1.loc[ix].values[:,1:-2].T)
        events   = np.int32(dataframe1.Stimulus_Type.loc[ix].values)
        stim_ID  = np.int32(dataframe1.Stimulus_ID.loc[ix].values)
        events[events==101] = 0

        # Epoch the data
        X, y, st_id = epoch_data(eeg_data[picks[p]], events, stim_ID, tmin=0.099, tmax=0.399)

        preds = np.zeros((len(y),2))
        # Apply the classifiers
        for clf in clfs[p]:
            preds += clf.predict_proba(X)

        yte = np.argmax(preds,1)
        # store the resutls
        patients_ID.extend([p] * len(yte))
        stimulus_ID.extend(list(st_id))
        prediction_Label.extend(list(yte + 1))

    # create the dataframe
    output1 = pd.DataFrame(columns=['PatientID', 'Stimulus_ID', 'Scored Labels'])
    output1['PatientID'] = patients_ID
    output1['Stimulus_ID'] = stimulus_ID
    output1['Scored Labels'] = np.int64(prediction_Label)

    # Return value must be of a sequence of pandas.DataFrame
    return output1,
