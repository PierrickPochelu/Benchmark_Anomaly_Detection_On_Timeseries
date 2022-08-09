This repo aims to compare different recent unsupervised anomaly detection methods on timeseries.

Typical anomaly detection algorithms are calibrated on anomaly-free historical data samples and used on the remainder of the time series where we are looking for anomalies. Two main approaches are evaluated: real-time anomaly detection and offline anomaly detection. Only univariate anomaly detection with unsupervised learning is evaluated here.

# METHODS

**Offline time series anomaly detection** takes a fixed sequence of input data samples as input named a "frame", and returns if there is an anomaly in the frame. They are convenient methods to detect anomalies in stored files with maximum reliability.
The usual workflow runs the following steps (you can follow them in the lower figure):
1. The input signal is **split between training/testing**. The first samples serve to train/calibrate the detector, and the following serves to test its ability to ring anomalies when needed. For each time series here, I use a  15%/85% ratio of training/testing data samples.
3. The signal is **standardized**. Like it is commonly done, we compute mean &mu; and std &sigma; on the training (anomaly-free) signal. Then we standardize x_train=(x_train-&mu;)/(&sigma;) and x_test=(x_test-&mu;)/(&sigma;)
4. The input signal is **split into frames**. For example, the time series [5,6,2,7,8,9] with 3 length frames, would produce those 4 frames: [5,6,2], [6,2,7], [2,7,8], [7,8,9]. During this step, we may apply data augmentation or features extraction to improve the detector's ability to extract useful information and ignore the noise.
5. The detector is **trained** on the training split.
6. During the **inference phase**, the frame of values is given to the detector.

The Python code below shows a common piece of code to train and predict (5th and 6th steps above).
```python
hyperparameters={"n_estimator":128}
from sklearn.ensemble import IsolationForest
model=IsolationForest(hyperparameters)
model.fit(X_train_frames) # we can iterate on "fit" method to improve the model
anomaly_detection=model.predict(X_test_frames)
```

**Realtime time series anomaly detections** consist in predicting incoming one-by-one values. They are especially useful to detect early an error. However, they seem less accurate than offline methods. The code below is an example of their API.
```python
hyperparameters={"N_bins":5}
from nab.detectors.relative_entropy.relative_entropy_detector import RelativeEntropyDetector
# We don't need to iterate on the training phase. It is common to give training samples to the constructor.
realtime_algo = RelativeEntropyDetector(dataSet=x_NAB_dataset_train, hyperparameters)
anomaly_detection=[]
for v in gen_X_test(): #generator producing 1-per-1 test value
	a=model.predict(v)
	anomaly_detection.append(a)
```


To compare multiple workflows at scale on many time series I design this workflow. It allows to loop easily or distributed the computing of the experiments. Each step needs to be called using their id-name and may require specific hyperparameters.

![Big picture of the workflow](media/workflow.png)

Methods used:
- Features extractor may include: data augmentation (keyword: "DATAAUG"), ROCKET [^1] ("ROCKET"), autoencoder compression ("DENSE_AE","CONV_AE","LSTM_AE"), or no one ("IDENTITY").
- Realtime detectors:  Adaptive Resonance [^2] ("ARTIME"), Conformal Anomaly Detector K-NN [^3] ("CADKNN"), x ("OSE"), Relative Entropy ("RE"), STUMP ("STUMP")
- Offline detectors: Autocoender with loss reconstruction ("DENSE_AE","CONV_AE","LSTM_AE"), IsolationForest [^4] ("IFOREST"), One-class SVM ("ONESVM"), EllipticEnvelope ("ELLIPTIC").


[^1]: "ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels", A. Dempster,  Data Mining and Knowledge Discovery 2020,  https://doi.org/10.1007/s10618-020-00701-z
[^2]: "Unsupervised real-time anomaly detection for streaming data", S. Ahmada et al., Neurocomputing 2017, https://doi.org/10.1016/j.neucom.2017.04.070 
[^3]: "Conformal k-NN Anomaly Detector for Univariate Data Streams", V. Ishimtsev et al., PMLR 2017, http://proceedings.mlr.press/v60/ishimtsev17a/ishimtsev17a.pdf
[^4]: "Isolation forest", ICDM 2008, F. T. Liu, https://doi.org/10.1109/ICDM.2008.17
[^5]: "STUMP", S.M. Law,2019, Journal of Open Source Software, 2019

Notice: all methods have been (hyper-)parametrized either re-using commonly used values or by doing myself some preliminary experiments

Frameworks used:
- tsaug [relevant doc page here](https://tsaug.readthedocs.io/en/stable/notebook/Examples%20of%20augmenters.html):  data augmentation for time series. It includes diverse kinds of noise: speed shift, gaussian multiplicative noise on each value,...
- pyts [relevant doc page here](https://pyts.readthedocs.io/en/stable/modules/transformation.html): data augmentation for time series. It includes recent methods such as ROCKET.
- scikit-learn [relevant doc page here](https://scikit-learn.org/stable/modules/outlier_detection.html): (non-deep) Machine Learning offline anomaly detectors: One-class SVM, Isolation Forest, Elliptic Envelope...
- numenta [github here](https://github.com/numenta/NAB): Realtime anomaly detection. It implements recent methods such as ARTime.
- tensorflow2.9 [relevant doc page here](https://keras.io/examples/timeseries/timeseries_anomaly_detection/): Autoencoder for detecting anomaly based on loss reconstruction, or Autoencoder for extract features.
- stumpy

# DATASETS
The time series are included in this github in ./data/NAB/. Official link is: https://github.com/numenta/NAB/tree/master/data

I tested my algorithms on NAB time series files. 58 files are present but 51 are valid for large-scale experiments. 7 files are invalid for one of those two reasons:
- The timeseries contains too early anomalies. So, we cannot easily calibrate the algorithms on anomaly-free signals.
- The timeseries contains NaN values.

Notice: A few files are fully anomaly-free (e.g., realAWSCloudwatch/ec2_cpu_utilization_c6585a.csv). The F1-score formula would fail due to arithmetic reason (division per zero) but we expect the detector produces no False Positive. I compute the accuracy on them and tag it as "F1 score" in the remaining.

The tested algorithms can be evaluated beyond those 51 files. Further possible investigations would include more time series: multi-variate time series, multi-modal time series, time series clustering, ...





# OFFLINE STRATEGY - EXPERIMENTAL RESULTS

To analyse a result and compare fairly multiple pipelines it is required to evaluate them on a large number of independant timeseries.

## Comparing different features extractors

I compare a few implemented features extractor with a commonly used detector: IsolationForest for its speed and predictions quality.

```python
from experiments import LAUNCH_EXPERIMENTS_AT_SCALE
from strat_map import detector_strat_map,feature_extractor_strat_map
if __name__=="__main__":
    from extract_data import extract_datasets
    datasets=extract_datasets("./data/NAB/")
    detector_strat_name= "IFOREST" # Isolation Forest implemented by scikit-learn
    for fe_strat_name in feature_extractor_strat_map.keys(): # for each known detection strategy
        results=LAUNCH_EXPERIMENTS_AT_SCALE(fe_strat_name, detector_strat_name, datasets)
        print(fe_strat_name, detector_strat_name, results)
```
produces:
```
IDENTITY IFOREST {'time': 33.986, 'tp': 7928, 'tn': 157209, 'fp': 34005, 'fn': 15003, 'f1': 0.2964}
ROCKET IFOREST {'time': 176.478, 'tp': 7557, 'tn': 163776, 'fp': 27438, 'fn': 15374, 'f1': 0.33}
DATAAUG IFOREST {'time': 123.847, 'tp': 6239, 'tn': 170956, 'fp': 20258, 'fn': 16692, 'f1': 0.281}
CONV_AE IFOREST {'time': 2856.494, 'tp': 7124, 'tn': 161476, 'fp': 29738, 'fn': 15807, 'f1': 0.2918}
DENSE_AE IFOREST {'time': 1775.215, 'tp': 8126, 'tn': 155902, 'fp': 35312, 'fn': 14802, 'f1': 0.3044}
LSTM_AE IFOREST {'time': 2545.714, 'tp': 4492, 'tn': 176413, 'fp': 14801, 'fn': 18439, 'f1': 0.2622}
```

** Conclusion: AE is more accurate but at the cost of significant training time. DATAAUG produces hardly small improvement and require a lot of tuning: (choice of noise kinds, probability of the noise, amplitude of the noises...). Indeed, no feature extractor (identity) is a robust, simple and fast strategy.**

More investigations should sweep other technics available. Pyts framework propose easily testable recent features extractors.

## Compare different offline algorithms

AE as features extractor and AE as detectors means two autoencoders are cascading. This cascade is way better than 1 single AE+IsolationForest (f1: 0.4062) but it requires two AE and they are significantly slower than non-deep ML algorithms such as IsolationForest.


## LARGE-SCALE EXPERIMENTAL RESULTS
I loop on all detector algorithms by fixing the features extraction strateg to "IDENTITY"

```python
from experiments import LAUNCH_EXPERIMENTS_AT_SCALE
from strat_map import detector_strat_map,feature_extractor_strat_map
if __name__=="__main__":
    from extract_data import extract_datasets
    datasets=extract_datasets("./data/NAB/")
    feature_extractor="IDENTITY"

    for detector in detector_strat_map.keys(): # for each known detection strategy
        results=LAUNCH_EXPERIMENTS_AT_SCALE(feature_extractor,detector,datasets)
        print(feature_extractor, detector, results)
```
produces:
```
IDENTITY RE {'time': 141.267, 'tp': 878, 'tn': 194490, 'fp': 2714, 'fn': 22540, 'f1': 0.1438}
IDENTITY CADKNN {'time': 1783.387, 'tp': 14517, 'tn': 110398, 'fp': 86806, 'fn': 8901, 'f1': 0.2673}
IDENTITY ARTIME {'time': 81.399, 'tp': 122, 'tn': 197014, 'fp': 190, 'fn': 23296, 'f1': 0.1119}
IDENTITY OSE {'time': 3017.315, 'tp': 5986, 'tn': 169234, 'fp': 27970, 'fn': 17432, 'f1': 0.1947}
IDENTITY STUMP {'time': 114.7, 'tp': 1873, 'tn': 190624, 'fp': 6580, 'fn': 21545, 'f1': 0.1502}
IDENTITY IFOREST {'time': 33.986, 'tp': 7928, 'tn': 157209, 'fp': 34005, 'fn': 15003, 'f1': 0.2964}
IDENTITY ONESVM {'time': 25.76, 'tp': 13101, 'tn': 78298, 'fp': 112916, 'fn': 9830, 'f1': 0.2249}
IDENTITY ELLIPTIC {'time': 673.583, 'tp': 6967, 'tn': 167652, 'fp': 23562, 'fn': 15964, 'f1': 0.2921}

deep learning approaches:
IDENTITY CONV_AE {'time': 3394.014, 'tp': 10083, 'tn': 147247, 'fp': 43967, 'fn': 12848, 'f1': 0.3396}
IDENTITY LSTM_AE {'time': 4575.304, 'tp': 9445, 'tn': 152804, 'fp': 38410, 'fn': 13486, 'f1': 0.3365}
IDENTITY DENSE_AE {'time': 1266.93, 'tp': 11133, 'tn': 142505, 'fp': 48709, 'fn': 11798, 'f1': 0.3336}
```

**Autoencoders is the most accurate at the cost of longer runtime. Isolation Forest is a g ood tradeof between accuracy and speed. The real-time strategies are less accurate than offline ones. This is why I am investigating further the offline ones.**

The lines 

## LARGE-SCALE INSIGHT
Below I plot IDENTITY_DENSE_AE strategy predictions, timeseries and labels.

![Large-scale anomaly detection](./media/IDENTITY_DENSE_AE_mosaic.png)

Click on it for a better view: time series name, F1-score, and detection/ground truth.

Legend:
- grey: anomaly-free training split
- no color: True negative
- green: True positive
- red: False negative error
- orange: False positive error

**The margin for improvement is obvious. On the majority of time series it is better than random, on some others, the method performs equally to random (everything tagged as non-anomaly).**

# STAY TUNED
I am actively developing this project. In next days new datasets (e.g., sound processing), new features extractors, and new detectors will be added... New large scale benchmarks to evaluate all those methods will be shown.
