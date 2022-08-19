import pandas as pd
import numpy as np

class NAB_dataset:
    def __init__(self, d1,d2=None):
        if d2 is None:
            self.data = pd.DataFrame(d1['x'], columns=['value']) # this line crash if you want online detector with frame_size
        else:
            self.data=pd.DataFrame(np.concatenate((d1["x"],d2["x"])),columns=["value"])

def get_anomaly_scores(realtime_algo, test_dataset) -> np.ndarray:
    anomaly_scores = np.zeros((len(test_dataset['x']),), dtype=np.float32)
    for i, v in enumerate(test_dataset['x']):
        anomaly_score = realtime_algo.handleRecord({"value": v})
        if isinstance(anomaly_score, list) or isinstance(anomaly_score,tuple):  # handle different technologies
            anomaly_score = anomaly_score[0]
        else:
            pass
        anomaly_scores[i] = anomaly_score
    return anomaly_scores

def predict_with_strat(realtime_algo, test_dataset, thresh=0.5) -> np.ndarray:
    rings=get_anomaly_scores(realtime_algo,test_dataset)
    for i, v in enumerate(rings):
        rings[i] = 1 if v >= thresh else 0
    return rings


def RE(train_dataset, test_dataset,wanted_hyperparameters={"nb_bins":5,}):
    hyperparameters={"nb_bins":5,}
    hyperparameters.update(wanted_hyperparameters)

    x_NAB_dataset_train = NAB_dataset(train_dataset)
    from nab.detectors.relative_entropy.relative_entropy_detector import RelativeEntropyDetector
    realtime_algo = RelativeEntropyDetector(dataSet=x_NAB_dataset_train, probationaryPercent=1.)
    realtime_algo.W=len(train_dataset["x"])
    realtime_algo.N_bins=hyperparameters["nb_bins"]
    from scipy import stats



    #realtime_algo.T=stats.chi2.isf(0.01, realtime_algo.N_bins - 1)#stats.chi2.isf(0.01, realtime_algo.N_bins - 1)
    rings=predict_with_strat(realtime_algo, test_dataset)
    #anomaly_score=realtime_algo.P
    return rings


def Numenta(train_dataset, test_dataset,wanted_hyperparameters={}):
    hyperparameters={"scale":0.125}
    hyperparameters.update(wanted_hyperparameters)
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context


    x = "https://github.com/markNZed/ARTimeNAB.jl"
    jl_code = f'import Pkg;Pkg.add(url="{x}")'
    from juliacall import Main as jl
    jl.seval(jl_code)
    jl.seval("using ARTime")

    from nab.detectors.ARTime.ARTime_detector import ARTimeDetector

    class AR:
        def __init__(self,train_dataset,hyperparameters):
            self.scale = hyperparameters["scale"]
            self.max=max(np.max(train_dataset["x"]),np.max(test_dataset["x"]))
            self.min=min(np.min(train_dataset["x"]),np.min(test_dataset["x"]))
            x_NAB_dataset_train = NAB_dataset(train_dataset)

            # TODO: Those two lines are a uurn around. We need a  suitable normalization to avoid this or to update NAB algorithms!!!!
            # The code to update is made with julia
            """
            if np.min(test_dataset["x"])<self.min or np.max(test_dataset["x"])>self.max:
                x_NAB_dataset_train = NAB_dataset(train_dataset,test_dataset) # we concatenante the train and test to inform NAB of the min and max possible value
                probationaryPercent=float(len(train_dataset["x"]))/(len(train_dataset["x"])+len(test_dataset["x"]))
            else:
            """
            #x_NAB_dataset_train = NAB_dataset(train_dataset,
            #                                 test_dataset)  # we concatenante the train and test to inform NAB of the min and max possible value
            #probationaryPercent = float(len(train_dataset["x"])) / (len(train_dataset["x"]) + len(test_dataset["x"]))
            probationaryPercent=1.
            self.algo = ARTimeDetector(dataSet=x_NAB_dataset_train, probationaryPercent=probationaryPercent)
            self.algo.initialize()
            # WARNING: ARTime fails if it take a too large value as input compared to those seen during the training phase
            # To make it more robute y propose to reduce them


        def handleRecord(self,x):
            # value clipping
            #processed_x=min(x["value"],self.max*1.24)
            #processed_x=max(processed_x,self.min*1.24)
            """"
            def amplificator(x,s,dir="up"):
                if dir=="up":
                    if x>0:
                        return x*s #f(x=1.,s=1.2) -> 1.2
                    else:
                        return x/s #f(x=-1.,s=1.2) -> -0.8333...
                else:
                    if x>0:
                        return x/s # f(x=1.,s=1.2) -> 0.8333
                    else:
                        return x*s # f(x=-1,s=1.2) -> -1.2
            """
            #processed_x=amplificator(min(x["value"],self.max),1.01,dir="up")
            #processed_x=amplificator(max(processed_x,self.min),1.01,dir="down")
            record={"value":x}
            #wh=self.algo.jl.p
            #print(wh)
            p=self.algo.handleRecord(record) #WARNING: if x is too large it can fail!!!!!!!! In this case increase "scale"
            return p

    algo=AR(train_dataset,hyperparameters)
    rings=get_anomaly_scores(algo, train_dataset)
    thresh=np.quantile(rings,0.99)+0.0001
    return predict_with_strat(algo, test_dataset, thresh)


def OSE(train_dataset, test_dataset,hyperparameters={}):
    default_hyperparameters={"context_length":7,"neurons":15,"norm_bits":3,"percentile":0.90,"rest_period":1,"threshold":0.75}
    hyperparameters.update(default_hyperparameters)

    x_NAB_dataset_train = NAB_dataset(train_dataset)


    from nab.detectors.context_ose.cad_ose import ContextualAnomalyDetectorOSE

    class OSE_detector:
        def __init__(self,train_dataset, hyperparameters):
            min_val,max_val=hyperparameters["min_val"],hyperparameters["max_val"]
            contextLength=hyperparameters["context_length"]
            neurons=hyperparameters["neurons"]
            norm_bits=hyperparameters["norm_bits"]
            percentile=hyperparameters["percentile"]
            rest_period=hyperparameters["rest_period"]
            threshold=hyperparameters["threshold"]
            self.algo = ContextualAnomalyDetectorOSE(min_val, max_val,
                                                     restPeriod=rest_period, baseThreshold=threshold,
                                                     maxLeftSemiContextsLenght=contextLength,
                                                     maxActiveNeuronsNum=neurons, numNormValueBits=norm_bits)

            anomaly_scores = []
            for i, v in enumerate(train_dataset['x']):
                anomaly_score = self.algo.getAnomalyScore({"value": v})
                anomaly_scores.append(anomaly_score)
            self.thresh = np.quantile(anomaly_scores, percentile)
            self.algo.baseThreshold=self.thresh


        def handleRecord(self, inputData):
            anomaly_score = self.algo.getAnomalyScore(inputData)
            return anomaly_score

    min_val = np.min(x_NAB_dataset_train.data.values)
    max_val = np.max(x_NAB_dataset_train.data.values)
    hp={}
    hp.update(hyperparameters)
    hp.update({"min_val":min_val,"max_val":max_val})
    algo = OSE_detector(train_dataset, hp)


    return predict_with_strat(algo, test_dataset)


def CADKNN(train_dataset, test_dataset, hyperparameters={}):
    x_NAB_dataset_train = NAB_dataset(train_dataset)
    from nab.detectors.knncad.knncad_detector import KnncadDetector
    algo = KnncadDetector(dataSet=x_NAB_dataset_train, probationaryPercent=1.) #train
    algo.initialize()
    return predict_with_strat(algo, test_dataset)


def stump(train_dataset, test_dataset,hyperparameters={"w":128,"quantile":0.9}):
    import stumpy
    from stumpy import stumpi
    mp=stumpi(train_dataset["x"],hyperparameters["w"],egress=False)
    for x in test_dataset["x"]:
        mp.update(x)

    # extract distances and thresh
    distances=mp.left_P_


    # offset training distances
    training_distances=distances[:len(train_dataset["x"])]
    training_distances=training_distances[hyperparameters["w"]//2:]

    testing_distances=distances[-1*len(test_dataset["x"]):]

    from offline_strategies.autoencoder import _from_loss_to_proba
    min=np.min(training_distances)
    max=np.max(training_distances)
    thresh=np.quantile(training_distances,hyperparameters["quantile"])
    proba=_from_loss_to_proba(testing_distances,thresh,min,max)
    assert(len(proba)==len(test_dataset["x"]))


    return proba



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
