import pandas as pd
import numpy as np


class NAB_dataset:
    def __init__(self, d):
        self.data = pd.DataFrame(d['x'], columns=['value'])


def predict_with_strat(realtime_algo, test_dataset,thresh=0.5) -> np.ndarray:
    rings = np.zeros((len(test_dataset['x']),), dtype=np.float32)


    for i, v in enumerate(test_dataset['x']):
        anomaly_score = realtime_algo.handleRecord({"value": v})
        if isinstance(anomaly_score, list) or isinstance(anomaly_score,tuple):  # handle different technologies
            anomaly_score = anomaly_score[0]
        else:
            pass
        rings[i] = 1 if anomaly_score >= thresh else 0
    return rings


def RE(train_dataset, test_dataset):
    x_NAB_dataset_train = NAB_dataset(train_dataset)
    from nab.detectors.relative_entropy.relative_entropy_detector import RelativeEntropyDetector
    realtime_algo = RelativeEntropyDetector(dataSet=x_NAB_dataset_train, probationaryPercent=1.)
    return predict_with_strat(realtime_algo, test_dataset)


def Numenta(train_dataset, test_dataset):
    x_NAB_dataset_train = NAB_dataset(train_dataset)
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    x = "https://github.com/markNZed/ARTimeNAB.jl"
    jl_code = f'import Pkg;Pkg.add(url="{x}")'
    print(jl_code)
    from juliacall import Main as jl
    jl.seval(jl_code)
    jl.seval("using ARTime")

    from nab.detectors.ARTime.ARTime_detector import ARTimeDetector  # may install Julia automatically
    algo = ARTimeDetector(dataSet=x_NAB_dataset_train, probationaryPercent=1.)
    algo.initialize()

    rings=predict_with_strat(algo, train_dataset)

    return predict_with_strat(algo, test_dataset)


def OSE(train_dataset, test_dataset):
    x_NAB_dataset_train = NAB_dataset(train_dataset)


    from nab.detectors.context_ose.cad_ose import ContextualAnomalyDetectorOSE

    class OSE_detector:
        def __init__(self,train_dataset, hyperparameters):
            min_val,max_val=hyperparameters["min_val"],hyperparameters["max_val"]
            contextLength=hyperparameters["context_length"]
            neurons=hyperparameters["neurons"]
            norm_bits=hyperparameters["norm_bits"]
            percentile=hyperparameters["percentile"]
            self.algo = ContextualAnomalyDetectorOSE(min_val, max_val,
                                                     restPeriod=1, baseThreshold=0.5, maxLeftSemiContextsLenght=contextLength,
                                                     maxActiveNeuronsNum=neurons, numNormValueBits=norm_bits)

            anomaly_scores = []
            for i, v in enumerate(train_dataset['x']):
                anomaly_score = self.algo.getAnomalyScore({"value": v})
                anomaly_scores.append(anomaly_score)
            self.thresh = np.percentile(anomaly_scores, percentile)

        def handleRecord(self, inputData):
            anomaly_score = self.algo.getAnomalyScore(inputData)
            return 1 if anomaly_score>self.thresh else 0

    min_val = np.min(x_NAB_dataset_train.data.values)
    max_val = np.max(x_NAB_dataset_train.data.values)
    hyerparameters={"min_val":min_val,"max_val":max_val,"context_length":128,"neurons":16,"norm_bits":3,"percentile":90}
    algo = OSE_detector(train_dataset, hyerparameters)


    return predict_with_strat(algo, test_dataset)


def CADKNN(train_dataset, test_dataset):
    x_NAB_dataset_train = NAB_dataset(train_dataset)
    from nab.detectors.knncad.knncad_detector import KnncadDetector
    algo = KnncadDetector(dataSet=x_NAB_dataset_train, probationaryPercent=1.) #train
    return predict_with_strat(algo, test_dataset)


def realtime_AD_strategy(strat_RT_AD):
    if strat_RT_AD == "RE":
        return RE
    elif strat_RT_AD == "CADKNN":
        return CADKNN
    elif strat_RT_AD == "ARTIME":
        return Numenta
    elif strat_RT_AD == "OSE":
        return OSE
    else:
        raise ValueError("Realtime strategy not understood")


"""
def relative_entropy_AD(train_dataset,test_dataset,strat_RT_AD)->np.ndarray:

    x_NAB_dataset_train = NAB_dataset(train_dataset)
    x_NAB_dataset_test = NAB_dataset(test_dataset)


    realtime_algo=realtime_AD_strategy(strat_RT_AD, x_NAB_dataset_train)

    rings = np.zeros((len(test_dataset['x']),),dtype=np.float32)
    for i,v in enumerate(test_dataset['x']):
        anomaly_score = realtime_algo.handleRecord({"value": v})
        print(anomaly_score)
        if isinstance(anomaly_score,list):#handle different technologies
            anomaly_score=anomaly_score[0]
        else:
            pass
        rings[i]= 1 if anomaly_score >= 0.5 else 0
    return rings
"""
