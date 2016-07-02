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

The models used in this challenge were already available as part of the [pyRiemann](http://pythonhosted.org/pyriemann/) toolbox. Almost no custom code has been developed.

### ERPs models

#### XdawnCov

```python
clf = make_pipeline(XdawnCovariances(6, estimator='oas'),
                    TangentSpace(metric='riemann'),
                    LogisticRegression(penalty='l1'))
```

#### ERPCov

```python
clf = make_pipeline(ERPCovariances(svd=16, estimator='oas'),
                    TangentSpace(metric='logdet'),
                    LogisticRegression(penalty='l1'))
```

#### Xdawn

```python
clf = make_pipeline(Xdawn(12, estimator='oas'),
                    DownSampler(5),
                    EpochsVectorizer(),
                    LogisticRegression(penalty='l2'))
```

### Induced activity models

#### CospCov

```python
clf = make_pipeline(CospCovariances(fs=1000, window=32, overlap=0.95, fmax=300, fmin=1),
                    CospBoostingClassifier(baseclf))
```

```python
baseclf = make_pipeline(ElectrodeSelection(10, metric=dict(mean='logeuclid', distance='riemann')),
                        TangentSpace(metric='riemann'),
                        LogisticRegression(penalty='l1'))
```

#### HankelCov

```python
clf = make_pipeline(DownSampler(2),
                    HankelCovariances(delays=[2, 4, 8, 12, 16], estimator='oas'),
                    TangentSpace(metric='logeuclid'),
                    LogisticRegression(penalty='l1'))
```

#### CSSP

```python
clf = make_pipeline(HankelCovariances(delays=[2, 4, 8, 12, 16], estimator='oas'),
                    CSP(30),
                    LogisticRegression(penalty='l1'))
```

## Results

### Challenge

#### Accuracy

|         | XdawnCov | ERPCov | Xdawn | Cosp      | HankelCov | CSSP | Ensemble |
|---------|----------|--------|-------|-----------|-----------|------|----------|
| p1      | 85.0     | 82.0   | 88.0  | 89.0      | 81.5      | 76.0 | **93.0** |
| p2      | 83.0     | 89.0   | 88.0  | 84.0      | 67.0      | 62.5 | **94.5** |
| p3      | 84.5     | 95.0   | 90.0  | **100.0** | 96.0      | 88.0 | 99.5     |
| p4      | 82.5     | 81.0   | 76.5  | 77.6      | 80.0      | 75.0 | **83.0** |
| Average | 83.8     | 86.7   | 85.6  | 87.6      | 81.1      | 75.4 | **92.5** |


#### AUC

|         | XdawnCov | ERPCov | Xdawn | Cosp      | HankelCov | CSSP  | Ensemble  |
|---------|----------|--------|-------|-----------|-----------|-------|-----------|
| p1      | 0.907    | 0.889  | 0.924 | 0.968     | 0.879     | 0.823 | **0.973** |
| p2      | 0.927    | 0.959  | 0.949 | 0.948     | 0.742     | 0.686 | **0.977** |
| p3      | 0.930    | 0.994  | 0.967 | **1.000** | 0.996     | 0.935 | **1.000** |
| p4      | 0.894    | 0.873  | 0.861 | 0.871     | 0.861     | 0.797 | **0.914** |
| Average | 0.915    | 0.929  | 0.925 | 0.947     | 0.869     | 0.810 | **0.966** |

### Paper

#### Accuracy

|         | XdawnCov | ERPCov | Xdawn | Cosp      | HankelCov | CSSP | Ensemble |
|---------|----------|--------|-------|-----------|-----------|------|----------|
| ca      | 89.3     | 96.3   | 90.0  | **98.7**  | 96.0      | 81.3 | **98.7** |
| de      | 90.7     | 88.7   | 95.0  | **98.0**  | 89.3      | 87.7 | 96.3     |
| fp      | 93.3     | 97.3   | 84.0  | **99.7**  | 96.3      | 91.3 | 97.7     |
| ja      | 93.3     | 96.0   | 94.7  | **99.7**  | 97.7      | 88.0 | 98.3     |
| mv      | 94.0     | 96.3   | 85.7  | **98.0**  | 95.7      | 88.7 | 97.3     |
| wc      | 97.7     | 98.3   | 97.7  | 97.3      | 95.0      | 95.0 | **98.7** |
| zt      | 98.7     | 99.0   | 99.3  | **100.0** | 99.3      | 97.0 | 99.7     |
| Average | 93.9     | 96.0   | 92.3  | **98.8**  | 95.6      | 89.9 | 98.1     |

#### AUC

|         | XdawnCov | ERPCov | Xdawn | Cosp  | HankelCov | CSSP  | Ensemble |
|---------|----------|--------|-------|-------|-----------|-------|----------|
| ca      | 0.961    | 0.992  | 0.964 | **0.999** | 0.985     | 0.885 | **0.999**    |
| de      | 0.960    | 0.951  | 0.987 | 0.996 | 0.953     | 0.954 | **0.998**    |
| fp      | 0.982    | 0.996  | 0.935 | **1.000** | 0.998     | 0.975 | 0.997    |
| ja      | 0.981    | 0.996  | 0.987 | **1.000** | 0.995     | 0.949 | 0.999    |
| mv      | 0.977    | 0.989  | 0.936 | 0.983 | 0.984     | 0.952 | **0.994**    |
| wc      | 0.993    | 0.996  | 0.996 | 0.996 | 0.990     | 0.990 | **0.999**    |
| zt      | 0.999    | **1.000**  | 0.995 | **1.000** | **1.000**     | 0.998 | **1.000**    |
| Average | 0.979    | 0.988  | 0.972 | 0.996 | 0.987     | 0.958 | **0.998**    |

## Discussion

### Things that didn't worked
- preprocessing, bagging, stacking, domain adaptation

### Things that could have improved the results
- longer epochs, data leak, better model selection (without CSSP)

## Reproduce the results

## References

[1] Miller, Kai J., Gerwin Schalk, Dora Hermes, Jeffrey G. Ojemann, and Rajesh PN Rao. "Spontaneous Decoding of the Timing and Content of Human Object Perception from Cortical Surface Recordings Reveals Complementary Information in the Event-Related Potential and Broadband Spectral Change." PLoS Comput Biol 12, no. 1 (2016)
