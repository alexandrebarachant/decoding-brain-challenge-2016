# Solution of the Decoding Brain Signals Cortana challenge

This data analysis challenge was organized on the Microsoft Azure ML plateform.
Data consisted in ECoG recording of 4 patients while they were presented picture of Face and House. The task was to build model achieving the highest categorization accuracy between the two type of pictures.

The experiemental protocol was identitcal to the one described in [1], but the dataset was much more challenging (lower quality ?).


## Forewords
There is some general ideas I followed to build my models for this challenge. I will develop them here.

- **Subject specific training**. Due to the difference in electrode implantation as well as the subject specificities of the brain patterns, the best solution was to train models independently on each subjects. While it was possible to fine-tune each model for each subjects, i choose not to do it, for the sake of scalability. The models and their hyper-parameters are common for each patient, but they are trained independently.  
- **No Preprocessing**. Biosignal analysis is generally heavy in preprocessing. Common Average Reference (CAR), notch filters and Baseline correction are one of these common steps that are blindly applied in the majority of the literature. Due to the trials of test data being shuffled, i was reluctant to use continuous filtering, and filtering epoched data is generally sub-optimal (Even more in this case were the end of the epoch was still containing useful information). As a consequence, I benchmarked the use of different preprocessing methods with my pipeline and found that none of them where improving my local validation. Therefore, I chose not to apply any preprocessing of the data and directly feed the raw signal to my models.
- **Local Training and Validation**. The Azure ML studio is not very adapted for exploratory data analysis and validation of the models. Mastering the cross-validation parameters, training subject specific model and ensembling them was a difficult task to setup. I decided to work locally with python, train my model, save them on disk and upload them on the plateform to be used in the webservice. I gained in flexibility but was also limited when it comes to use advanced toolboxes and methods. The scikit-learn distibution available on Azure ML was seriously outdated, and i could not install my own packages. As a result, I did not used any fancy classifier (100% logistic-regression), because it was one of the few model that was compatible both on my machine and the Azure plateform (I could have installed a virtual-env and downgraded my scikit-learn version + heavily modified my own toolbox pyRiemann, but i was lazy).
- **Simpler is Better**. Biosignal challenges are known to be extremely prone to overfitting and can give unconsistent results across data partitions. This is due to the nature of the signals, but also to the cognitive load and level of fatigue of the subjects that can strongly modulates the signal-to-noise ratio. Under this conditions, over-tuning hyper-parameter becomes dangerous and while methods like stacking classifiers or domain adaptation can be a source of improvement, they can results in a higher instability of the model. For this challenge, the amount of data was also small, and the inter-subject variability high. So instead of going [full nuclear](https://github.com/alexandrebarachant/Grasp-and-lift-EEG-challenge) with complex solutions, I chose to stick to a subset of simple model that i will simply ensemble by averaging their probabilities.

## Solution

The final solution is a blend of 6 different models, 3 dedicated to detection of evoked potential, and 3 to induced activity. For all models, data ranging from 100ms to 400ms after the onset of the stimulation have been used. No preprocessing or artifact rejection has been applied.

### ERPs models

#### XdawnCov

```
clf = make_pipeline(XdawnCovariances(6, estimator='oas'),
                    TangentSpace('riemann'),
                    LogisticRegression('l1'))
```

#### ERPCov

```
clf = make_pipeline(ERPCovariances(svd=16, estimator='oas'),
                    TangentSpace('logdet'),
                    LogisticRegression('l1'))
```

#### Xdawn

```
clf = make_pipeline(Xdawn(12, estimator='oas'),
                    DownSampler(5),
                    EpochsVectorizer(),
                    LogisticRegression('l2'))
```

### Induced activity models

#### CospCov

```
clf = make_pipeline(CospCovariances(fs=1000, window=32, overlap=0.95,
                                    fmax=300, fmin=1),
                    CospBoostingClassifier(baseclf))
```

```
baseclf = make_pipeline(ElectrodeSelection(10, metric=dict(mean='logeuclid',
                                                           distance='riemann')),
                        TangentSpace(metric='riemann'),
                        LogisticRegression('l1'))
```

#### HankelCov

```
clf = make_pipeline(DownSampler(2),
                    HankelCovariances(delays = [2, 4, 8, 12, 16],
                                      estimator='oas'),
                    TangentSpace('logeuclid'),
                    LogisticRegression('l1'))
```

#### CSSP

```
clf = make_pipeline(HankelCovariances(delays = [2, 4, 8, 12, 16],
                                      estimator='oas'),
                    CSP(30),
                    LogisticRegression('l1'))
```

## Results

### Challenge

#### Accuracy

|           | p1   | p2   | p3    | p4   | Average |
|-----------|------|------|-------|------|---------|
| XdawnCov  | 85.0 | 83.0 | 84.5  | 82.0 | 83.6    |
| ERPCov    | 82.0 | 88.5 | 95.0  | 81.0 | 86.6    |
| Xdawn     | 88.0 | 88.0 | 90.0  | 76.5 | 85.6    |
| Cosp      | 89.0 | 84.0 | **100.0** | 77.6 | 87.6    |
| HankelCov | 81.0 | 68.0 | 96.5  | 80.0 | 81.4    |
| CSSP      | 76.5 | 62.5 | 88.0  | 74.5 | 75.4    |
| Ensemble  | **93.0** | **94.5** | 99.5  | **83.0** | **92.5**|


#### AUC

### Paper

#### Accuracy
|           | ca   | de   | fp   | ja   | mv   | wc   | zt    | Average |
|-----------|------|------|------|------|------|------|-------|---------|
| XdawnCov  | 89.3 | 90.7 | 93.7 | 93.3 | 94.0 | 97.7 | 98.7  | 93.9    |
| ERPCov    | 96.3 | 88.7 | 97.3 | 96.0 | 96.3 | 98.3 | 99.0  | 96.0    |
| Xdawn     | 90.0 | 95.0 | 84.0 | 94.7 | 85.7 | 97.7 | 99.3  | 92.3    |
| Cosp      | **98.7** | **98.0** | **99.7** | **99.7** | **98.0** | 97.3 | **100.0** | **98.8**|
| HankelCov | 96.0 | 89.7 | 96.3 | 97.7 | 95.7 | 95.0 | 99.3  | 95.7    |
| CSSP      | 81.3 | 87.7 | 91.3 | 88.3 | 88.7 | 95.0 | 97.0  | 89.9    |
| Ensemble  | **98.7** | 96.3 | 97.7 | 98.3 | 97.3 | **98.7** | 99.7  | 98.1    |


## Discussion

### Things that didn't worked
- preprocessing, bagging, stacking, domain adaptation

### Things that could have improved the results
- longer epochs, data leak


## References

[1] Miller, Kai J., Gerwin Schalk, Dora Hermes, Jeffrey G. Ojemann, and Rajesh PN Rao. "Spontaneous Decoding of the Timing and Content of Human Object Perception from Cortical Surface Recordings Reveals Complementary Information in the Event-Related Potential and Broadband Spectral Change." PLoS Comput Biol 12, no. 1 (2016)
