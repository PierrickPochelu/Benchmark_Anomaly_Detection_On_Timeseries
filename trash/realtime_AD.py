from nab.detectors.knncad.knncad_detector import KnncadDetector
# https://github.com/numenta/NAB/tree/master/nab/detectors/knncad

from nab.detectors.relative_entropy.relative_entropy_detector import RelativeEntropyDetector


#from nab.detectors.context_ose.context_ose_detector import ContextOSEDetector


#from nab.detectors.ARTime.ARTime_detector import ARTimeDetector

#from nab.detectors.numenta.numentaTM_detector import NumentaTMDetector #HTM neural network



from typing import Union
import numpy as np


def REALTIME_AD(
        dataset:dict,
        train_test_split_rate:float,
        normalize_strategy_name:str,
        AD_strategies_name:Union[str,list],
               )->dict:



    # NORMALIZE
    from normalize_data import normalize_and_split
    train_dataset,test_dataset=normalize_and_split(dataset,train_test_split_rate,normalize_strategy_name=normalize_strategy_name)
    if np.sum(train_dataset["y"])!=0:
        raise ValueError("The begginning of the timeseries should be anomaly-free")

    #
    from realtime_strategies.algos import realtime_AD_strategy
    #_,x_test_pred=realtime_AD_strategy(train_dataset,test_dataset,AD_strategies_name)


    if isinstance(AD_strategies_name,str):
        AD_strategies_name=[AD_strategies_name]

    ensemble_size=len(AD_strategies_name)
    y_test_pred=None
    for AD_strategy_name in AD_strategies_name:
        strat=realtime_AD_strategy(AD_strategy_name)
        x_test_pred_local=strat(train_dataset,test_dataset)
        if y_test_pred is None:
            y_test_pred=x_test_pred_local/ensemble_size
        else:
            y_test_pred+=x_test_pred_local/ensemble_size


    from insight import confusion_matrix_and_F1
    y_test_frames=test_dataset['y']

    stats=confusion_matrix_and_F1(y_test_pred,y_test_frames)
    return stats

if __name__=="__main__":
    # GET DATA
    from extract_data import extract_one_dataset
    dataset=extract_one_dataset("./data/NAB/", "artificialWithAnomaly/art_daily_jumpsdown.csv")

    # run a workflow
    stats,details=REALTIME_AD(dataset,
        train_test_split_rate=0.15,
        normalize_strategy_name="STD",
        AD_strategies_name=["OSE" for i in range(1)]
    )
    print(stats)

