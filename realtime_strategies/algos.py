import pandas as pd
import numpy as np
# from nab.detectors.numenta.numentaTM_detector import NumentaTMDetector #Not used yet

class NAB_dataset:
    def __init__(self, d1,d2=None):
        if d2 is None:
            self.data = pd.DataFrame(d1['x'], columns=['value'])
        else:
            self.data=pd.DataFrame(np.concatenate((d1["x"],d2["x"])),columns=["value"])

def get_anomaly_scores(realtime_algo, test_dataset) -> np.ndarray:
    anomaly_scores = np.zeros((len(test_dataset['x']),), dtype=np.float32)
    for i, v in enumerate(test_dataset['x']):
        #try:
        anomaly_score = realtime_algo.handleRecord({"value": v})
        if isinstance(anomaly_score, list) or isinstance(anomaly_score,tuple):  # handle different technologies
            anomaly_score = anomaly_score[0]
        else:
            pass
        anomaly_scores[i] = anomaly_score
        #except:
        #    print(i,"->", v)
    return anomaly_scores

def predict_with_strat(realtime_algo, test_dataset, thresh=0.5) -> np.ndarray:
    rings=get_anomaly_scores(realtime_algo,test_dataset)
    for i, v in enumerate(rings):
        rings[i] = 1 if v >= thresh else 0
    return rings


def RE(train_dataset, test_dataset,hyperparameters={"nb_bins":5,}):
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


def Numenta(train_dataset, test_dataset,hyperparameters={"scale":0.125}):

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
            self.max=np.max(train_dataset["x"])
            self.min=np.min(train_dataset["x"])
            x_NAB_dataset_train = NAB_dataset(train_dataset)

            # TODO: Those two lines are a uurn around. We need a  suitable normalization to avoid this or to update NAB algorithms!!!!
            # The code to update is made with julia
            if np.min(test_dataset["x"])<self.min or np.max(test_dataset["x"])>self.max:
                x_NAB_dataset_train = NAB_dataset(train_dataset,test_dataset) # we concatenante the train and test to inform NAB of the min and max possible value

                probationaryPercent=float(len(train_dataset["x"]))/(len(train_dataset["x"])+len(test_dataset["x"]))
            else:
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
            #record={"value":processed_x}
            #wh=self.algo.jl.p
            #print(wh)
            p=self.algo.handleRecord(x) #WARNING: if x is too large it can fail!!!!!!!! In this case increase "scale"
            return p

    algo=AR(train_dataset,hyperparameters)
    rings=get_anomaly_scores(algo, train_dataset)
    thresh=np.quantile(rings,0.99)+0.0001
    return predict_with_strat(algo, test_dataset, thresh)


def OSE(train_dataset, test_dataset,hyperparameters={"context_length":7,"neurons":15,
                                                     "norm_bits":3,"percentile":0.90,
                                                     "rest_period":1,"threshold":0.75}):
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
    return predict_with_strat(algo, test_dataset)





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
