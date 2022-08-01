# TODO
Data augmentation:
https://tsaug.readthedocs.io/en/stable/index.html


# INTRODUCTION

# METHODS

Offline timeseries anomaly detection consists in producing time frames. It requires splitting the signal into smaller frames. For example, the time series [5,6,2,7,8,9] with 3 length frames, would produce those 4 frames: [5,6,2], [6,2,7], [2,7,8], [7,8,9]. They are convenients methods to detect anomalies on the disk with maximum of reliability.

The offline approach are generally used as following like any unsupervised algorithm:
```
hyperparameters={"n_estimator":128}
from sklearn.ensemble import IsolationForest
model=IsolationForest(hyperparameters)
model.fit(X_train_frames) # we can iterate on fit to improve the model
anomaly_detection=model.predict(X_test_frames)
```

Realtime timeseries anomaly detections the values are incoming one-by-one. They are especially usefull to detect early an error. However, they seem less accurate than offline methods.
```
hyperparameters={"n_estimator":128}
from sklearn.ensemble import IsolationForest
model=IsolationForest(X_train,hyperparameters) # don't need to iterate on the training phase. We passe
anomaly_detection=[]
for v in gen_X_test(): #generator producing 1-per-1 test value
	a=model.predict(v)
	anomaly_detection.append(a)
```


To compare multiple workflows at scale on a large number of timeseries I design this workflow. It allows to loop easily or distributed the computing of the experiments. Each step needs to be called using their id-name and may require specific hyperparameters.

#![Big picture of the workflow](workflow.png)

Methods used:
- Train/Test split: I use 15% of training and the remaining as testing.
- Norm: Like it is commonly done, we compute mean &mu; and std &sigma; on the training (anomaly-free) signal. Then we standardize x_train=(x_train-&mu;)/(&sigma;) and x_test=(x_test-&mu;)/(&sigma;)
- Features extractor may include: data augmentation ("DATAAUG"), ROCKET ("ROCKET"), autoencoder compression ("AE"), or no one ("IDENTITY").
- Realtime detectors: ARTime, CAD-KNN, OSE, Relative Entropy
- Offline detectors: Autocoender with loss reconstruction ("AE"),

Notice: autoencoders are used for two different ways and require different hyperparameters. For features extraction we use 5 conv. layers, and the detector 9 conv. layers.

Frameworks used:
- tsaug <link>:  data augmentation for timeseries. It includes diverse kind of noises: speed shift, gaussian multiplicative noise on each value, ....
- pyts <link>: data augmentation for timeseries. It includes recent methods such as ROCKET.
- scikit-learn <link>: (non-deep) Machine Learning offline anomaly detectors: One-class SVM, Isolation Forest, Elliptic Envelope...
- numenta <link>: Realtime anomaly detection. It implements recent methods such as ARTime.
- tensorflow2.9 <link>: Autoencoder for detect anomaly based on loss reconstruction, or Autoencoder for extract features.


# DATASETS
I tested my algorithms on NAB timeserie files <link>. 58 are present,  I use 51 are valid for my large-scale experiments. I tagged 7 files invalid:
- Files containing too early anormalies. So, it make it impossible to calibrate the algorithms on anomaly-free signals
- Files containing NaN values

Notice: A few files are fully anomaly-free (e.g., realAWSCloudwatch/ec2_cpu_utilization_c6585a.csv). The F1-score formula would failed due to arithmetic reason (division per zero) but we expect the detector produces no False Positive. I compute the accuracy on them and tag it as "F1 score" in the remaining.

The tested algorithms can be evaluated beyond those 51 files. Further possible investigations would include multi-variate time series, multi-modal time series, timeseries clustering with anomaly, ...





# OFFLINE STRATEGY - EXPERIMENTAL RESULTS

# Let's compare a simple strategy on all datasets. It consists to used the normed signal frame-per-frame to IsolationForest. The length of frames is 128 values and 15% of signals is used as unsupervised training ("unsupervised" but we know there is no anomaly in the training split).

```python
import os
from extract_data import extract_datasets
from offline_AD import OFFLINE_AD
from insight import plot_curves
datasets=extract_datasets("./data/NAB/")
paths=[] # we will build a beautiful mosaic
for dataset_name, dataset in datasets.items():
    try:
        stats=None
        stats,details=OFFLINE_AD(dataset,
                train_test_split_rate=0.15,
                frame_size=128,
                normalize_strategy_name="STD",
                FE_frame_strategy_name="IDENTITY",
                AD_strategies_name="IFOREST" #IsolationForest
            )
    except Exception as err:
        print(f"Exception with dataset {dataset_name} type:{type(err)} msg:{err}")
    if stats is not None:
        print(dataset_name, " stats:", stats)
```

It produces:
```
<sub>
realKnownCause/rogue_agent_key_hold.csv  stats: {'tn': 422, 'fp': 861, 'fn': 0, 'tp': 190, 'f1': 0.3062}
realKnownCause/ec2_request_latency_system_failure.csv  stats: {'tn': 1578, 'fp': 1377, 'fn': 130, 'tp': 216, 'f1': 0.2228}
Exception with dataset realKnownCause/machine_temperature_system_failure.csv type:<class 'ValueError'> msg:The begginning of the timeseries should be anomaly-free
realKnownCause/machine_temperature_system_failure.csv  stats: {'tn': 1578, 'fp': 1377, 'fn': 130, 'tp': 216, 'f1': 0.2228}
realKnownCause/cpu_utilization_asg_misconfiguration.csv  stats: {'tn': 12347, 'fp': 1370, 'fn': 408, 'tp': 1091, 'f1': 0.551}
realKnownCause/rogue_agent_key_updown.csv  stats: {'tn': 3698, 'fp': 159, 'fn': 534, 'tp': 0, 'f1': 0.0}
realKnownCause/ambient_temperature_system_failure.csv  stats: {'tn': 2770, 'fp': 2552, 'fn': 140, 'tp': 588, 'f1': 0.304}
realKnownCause/nyc_taxi.csv  stats: {'tn': 5752, 'fp': 1858, 'fn': 609, 'tp': 426, 'f1': 0.2567}
artificialWithAnomaly/art_load_balancer_spikes.csv  stats: {'tn': 2562, 'fp': 336, 'fn': 156, 'tp': 247, 'f1': 0.501}
artificialWithAnomaly/art_daily_jumpsup.csv  stats: {'tn': 2842, 'fp': 54, 'fn': 271, 'tp': 134, 'f1': 0.4519}
artificialWithAnomaly/art_daily_jumpsdown.csv  stats: {'tn': 2873, 'fp': 23, 'fn': 309, 'tp': 96, 'f1': 0.3664}
artificialWithAnomaly/art_daily_nojump.csv  stats: {'tn': 2843, 'fp': 53, 'fn': 395, 'tp': 10, 'f1': 0.0427}
artificialWithAnomaly/art_increase_spike_density.csv  stats: {'tn': 2867, 'fp': 29, 'fn': 332, 'tp': 73, 'f1': 0.288}
artificialWithAnomaly/art_daily_flatmiddle.csv  stats: {'tn': 2632, 'fp': 264, 'fn': 194, 'tp': 211, 'f1': 0.4795}
realTweets/Twitter_volume_IBM.csv  stats: {'tn': 10872, 'fp': 921, 'fn': 1263, 'tp': 327, 'f1': 0.2304}
realTweets/Twitter_volume_KO.csv  stats: {'tn': 10248, 'fp': 1512, 'fn': 1388, 'tp': 199, 'f1': 0.1207}
Exception with dataset realTweets/Twitter_volume_PFE.csv type:<class 'ValueError'> msg:The begginning of the timeseries should be anomaly-free
realTweets/Twitter_volume_PFE.csv  stats: {'tn': 10248, 'fp': 1512, 'fn': 1388, 'tp': 199, 'f1': 0.1207}
realTweets/Twitter_volume_FB.csv  stats: {'tn': 11750, 'fp': 0, 'fn': 1582, 'tp': 0, 'f1': 0.0}
Exception with dataset realTweets/Twitter_volume_AMZN.csv type:<class 'ValueError'> msg:The begginning of the timeseries should be anomaly-free
realTweets/Twitter_volume_AMZN.csv  stats: {'tn': 11750, 'fp': 0, 'fn': 1582, 'tp': 0, 'f1': 0.0}
Exception with dataset realTweets/Twitter_volume_UPS.csv type:<class 'ValueError'> msg:The begginning of the timeseries should be anomaly-free
realTweets/Twitter_volume_UPS.csv  stats: {'tn': 11750, 'fp': 0, 'fn': 1582, 'tp': 0, 'f1': 0.0}
Exception with dataset realTweets/Twitter_volume_CVS.csv type:<class 'ValueError'> msg:The begginning of the timeseries should be anomaly-free
realTweets/Twitter_volume_CVS.csv  stats: {'tn': 11750, 'fp': 0, 'fn': 1582, 'tp': 0, 'f1': 0.0}
realTweets/Twitter_volume_CRM.csv  stats: {'tn': 11355, 'fp': 442, 'fn': 1168, 'tp': 425, 'f1': 0.3455}
realTweets/Twitter_volume_GOOG.csv  stats: {'tn': 11653, 'fp': 254, 'fn': 1105, 'tp': 327, 'f1': 0.3249}
Exception with dataset realTweets/Twitter_volume_AAPL.csv type:<class 'ValueError'> msg:The begginning of the timeseries should be anomaly-free
realTweets/Twitter_volume_AAPL.csv  stats: {'tn': 11653, 'fp': 254, 'fn': 1105, 'tp': 327, 'f1': 0.3249}
artificialNoAnomaly/art_daily_no_noise.csv  stats: {'tn': 3277, 'fp': 24, 'fn': 0, 'tp': 0, 'f1': 0.9927}
Exception with dataset artificialNoAnomaly/art_flatline.csv type:<class 'ValueError'> msg:Error 4032 value(s) have been fond in x
artificialNoAnomaly/art_flatline.csv  stats: {'tn': 3277, 'fp': 24, 'fn': 0, 'tp': 0, 'f1': 0.9927}
artificialNoAnomaly/art_daily_small_noise.csv  stats: {'tn': 3208, 'fp': 93, 'fn': 0, 'tp': 0, 'f1': 0.9718}
artificialNoAnomaly/art_noisy.csv  stats: {'tn': 3019, 'fp': 282, 'fn': 0, 'tp': 0, 'f1': 0.9146}
artificialNoAnomaly/art_daily_perfect_square_wave.csv  stats: {'tn': 3257, 'fp': 44, 'fn': 0, 'tp': 0, 'f1': 0.9867}
realTraffic/occupancy_6005.csv  stats: {'tn': 1565, 'fp': 92, 'fn': 222, 'tp': 17, 'f1': 0.0977}
realTraffic/speed_6005.csv  stats: {'tn': 1689, 'fp': 70, 'fn': 175, 'tp': 64, 'f1': 0.3432}
realTraffic/speed_t4013.csv  stats: {'tn': 1732, 'fp': 12, 'fn': 242, 'tp': 8, 'f1': 0.0593}
realTraffic/TravelTime_387.csv  stats: {'tn': 1459, 'fp': 371, 'fn': 141, 'tp': 27, 'f1': 0.0954}
realTraffic/TravelTime_451.csv  stats: {'tn': 1507, 'fp': 0, 'fn': 204, 'tp': 0, 'f1': 0.0}
realTraffic/speed_7578.csv  stats: {'tn': 490, 'fp': 225, 'fn': 71, 'tp': 45, 'f1': 0.2332}
realTraffic/occupancy_t4013.csv  stats: {'tn': 1691, 'fp': 53, 'fn': 138, 'tp': 116, 'f1': 0.5485}
realAdExchange/exchange-2_cpc_results.csv  stats: {'tn': 582, 'fp': 635, 'fn': 37, 'tp': 0, 'f1': 0.0}
realAdExchange/exchange-2_cpm_results.csv  stats: {'tn': 710, 'fp': 380, 'fn': 157, 'tp': 7, 'f1': 0.0254}
realAdExchange/exchange-4_cpc_results.csv  stats: {'tn': 0, 'fp': 1138, 'fn': 0, 'tp': 132, 'f1': 0.1883}
realAdExchange/exchange-4_cpm_results.csv  stats: {'tn': 50, 'fp': 1077, 'fn': 8, 'tp': 135, 'f1': 0.1993}
realAdExchange/exchange-3_cpc_results.csv  stats: {'tn': 1077, 'fp': 0, 'fn': 104, 'tp': 0, 'f1': 0.0}
realAdExchange/exchange-3_cpm_results.csv  stats: {'tn': 945, 'fp': 81, 'fn': 155, 'tp': 0, 'f1': 0.0}
realAWSCloudwatch/ec2_disk_write_bytes_c0d644.csv  stats: {'tn': 2602, 'fp': 294, 'fn': 330, 'tp': 75, 'f1': 0.1938}
realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv  stats: {'tn': 494, 'fp': 2403, 'fn': 101, 'tp': 303, 'f1': 0.1949}
realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv  stats: {'tn': 0, 'fp': 2898, 'fn': 0, 'tp': 403, 'f1': 0.2176}
realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv  stats: {'tn': 1309, 'fp': 1649, 'fn': 137, 'tp': 206, 'f1': 0.1874}
realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv  stats: {'tn': 2243, 'fp': 654, 'fn': 102, 'tp': 302, 'f1': 0.4441}
realAWSCloudwatch/elb_request_count_8c0756.csv  stats: {'tn': 2873, 'fp': 73, 'fn': 289, 'tp': 66, 'f1': 0.2672}
realAWSCloudwatch/ec2_network_in_5abac7.csv  stats: {'tn': 3246, 'fp': 174, 'fn': 369, 'tp': 105, 'f1': 0.2789}
realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv  stats: {'tn': 2804, 'fp': 93, 'fn': 278, 'tp': 126, 'f1': 0.4045}
realAWSCloudwatch/ec2_disk_write_bytes_1ef3de.csv  stats: {'tn': 3340, 'fp': 79, 'fn': 374, 'tp': 101, 'f1': 0.3084}
realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv  stats: {'tn': 1746, 'fp': 1183, 'fn': 239, 'tp': 133, 'f1': 0.1576}
realAWSCloudwatch/grok_asg_anomaly.csv  stats: {'tn': 234, 'fp': 3102, 'fn': 105, 'tp': 360, 'f1': 0.1833}
realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv  stats: {'tn': 115, 'fp': 2782, 'fn': 104, 'tp': 300, 'f1': 0.1721}
realAWSCloudwatch/iio_us-east-1_i-a2eb1cd9_NetworkIn.csv  stats: {'tn': 0, 'fp': 872, 'fn': 0, 'tp': 58, 'f1': 0.1174}
realAWSCloudwatch/ec2_cpu_utilization_c6585a.csv  stats: {'tn': 3286, 'fp': 15, 'fn': 0, 'tp': 0, 'f1': 0.9955}
realAWSCloudwatch/ec2_network_in_257a54.csv  stats: {'tn': 2894, 'fp': 2, 'fn': 388, 'tp': 17, 'f1': 0.0802}
realAWSCloudwatch/ec2_cpu_utilization_77c1ca.csv  stats: {'tn': 2506, 'fp': 390, 'fn': 323, 'tp': 82, 'f1': 0.187}
realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv  stats: {'tn': 2666, 'fp': 229, 'fn': 304, 'tp': 102, 'f1': 0.2768}
</sub>
```

# Compare different FE algorithms
The training and testing signals have been normed, I fix the dataset (art_daily_jumpsdown) and the anomaly detector step (Isolation Forest). An ideal methodology should compute the F1-score acress dozens of different datasets and different runtimes (random seeds) but would increase significantly the training time.
```python
from offline_AD import OFFLINE_AD
from extract_data import extract_one_dataset
dataset=extract_one_dataset("./data/NAB/", "artificialWithAnomaly/art_daily_jumpsdown.csv")
for feature_extractor in ["IDENTITY",
                          "AE",
                          "ROCKET",
                          "DATAAUG"]:
    stats,details=OFFLINE_AD(dataset,
            train_test_split_rate=0.15,
            frame_size=128,
            normalize_strategy_name="STD",
            FE_frame_strategy_name=feature_extractor,
            AD_strategies_name="IFOREST" #IsolationForest
        )
    print(feature_extractor," stats:", stats)
```
produces:
```
IDENTITY  stats: {'tn': 2866, 'fp': 30, 'fn': 310, 'tp': 95, 'f1': 0.3585}
AE  stats: {'tn': 2842, 'fp': 54, 'fn': 288, 'tp': 117, 'f1': 0.4062}
ROCKET  stats: {'tn': 2652, 'fp': 244, 'fn': 379, 'tp': 26, 'f1': 0.077}
DATAAUG  stats: {'tn': 2886, 'fp': 10, 'fn': 314, 'tp': 91, 'f1': 0.3597}
```

AE is more accurate but at the cost of significant training time. DATAAUG produces hardly small improvement and require a lot of tuning: (choice of noise kinds, probability of the noise, amplitude of the noises...). Indeed, no feature extractor (identity) is a robust, simple and fast strategy.

More investigations should sweep other technics available. Pyts framework propose easily testable recent features extractors.

## Compare different offline algorithms

Now we see AE is a better feature extractors, we will compare different detectors
```python
from offline_AD import OFFLINE_AD
from extract_data import extract_one_dataset
dataset=extract_one_dataset("./data/NAB/", "artificialWithAnomaly/art_daily_jumpsdown.csv")
#for detector in ["AE","ELLIPTIC","ONESVM","IFOREST", [IFOREST,AE]]:
for detector in ["AE","ELLIPTIC","ONESVM","IFOREST"]:
    stats,details=OFFLINE_AD(dataset,
                train_test_split_rate=0.15,
                frame_size=128,
                normalize_strategy_name="STD",
                FE_frame_strategy_name="AE",
                AD_strategies_name=detector #IsolationForest
            )
    print(detector, " stats:", stats)
```
produces
```
AE  stats: {'tn': 2864, 'fp': 32, 'fn': 232, 'tp': 173, 'f1': 0.5672}
ELLIPTIC  stats: {'tn': 2890, 'fp': 6, 'fn': 404, 'tp': 1, 'f1': 0.0049}
ONESVM  stats: {'tn': 1496, 'fp': 1400, 'fn': 187, 'tp': 218, 'f1': 0.2155}
IFOREST  stats: {'tn': 2834, 'fp': 62, 'fn': 290, 'tp': 115, 'f1': 0.3952}
```

# "AE" means we extract features with AE (compression) and detect anomalies with another AE based on loss error. This cascade is way better than 1 single AE (f1: 0.4062)


## LARGE SCALE INSIGHT
I take the simple and fast strategy consisting in applying Isolation Forest on a standardize signal. I use it on all datasets and monitor it.
```python
import os
from extract_data import extract_datasets
from offline_AD import OFFLINE_AD
from insight import plot_curves, mosaic
datasets=extract_datasets("./data/NAB/")
paths=[] # we will build a beautiful mosaic
for dataset_name, dataset in datasets.items():
    try:
        stats=None
        stats,details=OFFLINE_AD(dataset,
                train_test_split_rate=0.15,
                frame_size=128,
                normalize_strategy_name="STD",
                FE_frame_strategy_name="IDENTITY",
                AD_strategies_name="IFOREST"
            )
    except Exception as err:
        print(f"Exception with dataset {dataset_name} type:{type(err)} msg:{err}")
    if stats is not None:
        # Monitor
        print(dataset_name, " stats:", stats)
        name = dataset_name.replace(os.sep, "_").split(".")[0] + "_isolation_forest"
        path = os.path.join("media", name + ".png")
        paths.append(path)
        txt = name + "\nF1-score:" + str(stats["f1"])
        plot_curves(x_train=details["train_dataset"]["x"],
                    x_test=details["test_dataset"]["x"],
                    y_test=details["test_dataset"]["y"],
                    y_pred=details["y_test_pred"],
                    frame_size=details["frame_size"],
                    path=path, txt=txt)
mosaic(paths,"mosaic.png")
```

#![Large-scale anomaly detection](mosaic.png)
Legend:
- grey: training split
- no color: True negative
- green: True positive
- red: False negative error
- orange: False positive error

The margin for improvement is obvious. On the majority of timeseries it is better than random, on some others, the method performs equally to random (everything tagged as non-anomaly).

# ONLINE STRATEGY - EXPERIMENTAL RESULTS
"""
from realtime_AD import REALTIME_AD
from extract_data import extract_one_dataset
dataset=extract_one_dataset("./data/NAB/", "artificialWithAnomaly/art_daily_jumpsdown.csv")
for detector in ["RE", "CADKNN", "ARTIME","OSE"]:
    stats=REALTIME_AD(dataset,
                train_test_split_rate=0.15,
                normalize_strategy_name="STD",
                AD_strategies_name=detector
            )
    print(detector, " stats:", stats)
"""
"""
RE  stats: {'tn': 3004, 'fp': 19, 'fn': 399, 'tp': 6, 'f1': 0.0279}
CADKNN  stats: {'tn': 1568, 'fp': 1455, 'fn': 198, 'tp': 207, 'f1': 0.2003}
ARTIME  stats: {'tn': 3010, 'fp': 13, 'fn': 403, 'tp': 2, 'f1': 0.0095}
OSE  stats: {'tn': 2943, 'fp': 80, 'fn': 378, 'tp': 27, 'f1': 0.1055}
"""

The realtime strategy are less accurate than offline ones. However, they do not require to repeat the training when new data samples are incoming.
ARTime strategy fails while the author published amazing results. I am investigating further.
