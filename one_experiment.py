import time
from typing import *
import numpy as np
from typing import Union,Tuple,Optional
from strat_map import detector_strat_map
from insight import confusion_matrix_and_F1, ROCAUC

def convert_to_callables(AD_strategy):
    if not isinstance(AD_strategy,list):
        AD_strategy=[AD_strategy]


    out=[]
    for a in AD_strategy:
        if isinstance(a, Callable):
            out.append(a)
        elif isinstance(a, str):
            f,_=detector_strat_map[a]
            out.append(f)
        else:
            raise ValueError("Not expectected type")
    return out

class experiment_ts_builder: #inspired from builder design pattern
    def __init__(self, AD_strategy:Union[Callable, List, str],
                        AD_hyperparameters:Union[Optional[Dict],List[Optional[Dict]]],
                 is_realtime:bool=False,
                 proba_thresh:float=0.5):

        # force it to be a list to be able to handle ensembles
        self.AD_strategies=convert_to_callables(AD_strategy)

        if isinstance(AD_hyperparameters, Dict) or AD_hyperparameters is None:
            self.AD_hyperparameters = [AD_hyperparameters]
        else:
            self.AD_hyperparameters=AD_hyperparameters


        self.is_realtime=is_realtime
        self.proba_thresh=proba_thresh
        self.experimental_result={}

        self.train_dataset=None
        self.test_dataset=None

    def __deduce_frame_size(self,x):
        if len(x.shape)==1:
            frame_size=1
        elif len(x.shape)==2:
            nb_frames=len(x)
            if nb_frames==0:
                raise ValueError("Unexpected x is empty")
            else:
                frame_size=len(x[0])
        else:
            raise ValueError("Unexpected x shape")
        return frame_size


    def fit_and_predict(self, train_dataset,test_dataset
                        )->np.ndarray:
        # can be used latter to compute insight in the evaluation phase
        self.train_dataset=train_dataset
        self.test_dataset=test_dataset

        # BUILD THE MODEL AND RETURNS ITS PREDICTIONS
        start_time = time.time()
        ensemble_size = len(self.AD_strategies)
        y_test_pred = None
        for detector_f,hp in zip(self.AD_strategies,self.AD_hyperparameters):
            x_test_pred_local = detector_f(train_dataset, test_dataset, hp)

            if y_test_pred is None:
                y_test_pred = x_test_pred_local / ensemble_size
            else:
                y_test_pred += x_test_pred_local / ensemble_size
        y_test_pred = y_test_pred > self.proba_thresh
        enlapsed_time = round(time.time() - start_time, 3)
        self.experimental_result["time"]=enlapsed_time
        return y_test_pred

    def evaluate(self,y_test_pred, test_dataset):
        # Warning: RT and Offline strategy do not exactly predict the same length of datasamples
        # Today, it is impossible to make an ensemble between those two strategies
        frame_size=self.__deduce_frame_size(test_dataset["x"])
        #y_test = test_dataset['y'][frame_size - 1:]

        self.experimental_result.update( confusion_matrix_and_F1(y_test_pred>=self.proba_thresh, test_dataset["y"]))
        self.experimental_result.update(ROCAUC(y_test_pred, test_dataset["y"]))

        details = {"train_dataset": self.train_dataset, "test_dataset": test_dataset, "frame_size": frame_size,
                   "y_test_pred": y_test_pred}

        return self.experimental_result, details











if __name__=="__main__":
    # GET DATA
    #from extract_data import extract_one_dataset
    #extract_datasets("./NAB-datasets/")
    #dataset=extract_one_dataset("./data/NAB/", "artificialWithAnomaly/art_daily_jumpsdown.csv")

    """
    for i in range(5):#test the reproducibility
        stats,details=OFFLINE_AD(dataset,
            train_test_split_rate=0.15,
            frame_size=128,
            normalize_strategy_name="STD",
            FE_frame_strategy_name="SIMPLE",
            AD_strategies_name=["AE" for i in range(1)] # we can make some ensembles! /!\ training time is multiplied
        )
        print(stats)
    """