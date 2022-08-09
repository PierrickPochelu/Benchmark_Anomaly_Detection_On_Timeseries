import time

import numpy as np
from typing import Union,Tuple,Optional
from strat_map import detector_strat_map, feature_extractor_strat_map





def EXPERIMENT(
        train_dataset:dict,test_dataset:dict,data_prep_info:dict,
        AD_strategies_name:Union[str,list]="IFOREST",
        AD_hyperparameters:Optional[dict]=None,
        proba_thresh:float=0.5
               )->Tuple[Optional[dict],Optional[dict]]:
    experimental_result={}

    # DETECTOR STRATEGY extract info
    if isinstance(AD_strategies_name,str):
        AD_strategies_name=[AD_strategies_name] # force it as a list to handle ensembles
    detector_func_pointers=[]
    is_realtime_vec=[]
    for AD_strategy_name in AD_strategies_name:
        detector_f, is_realtime = detector_strat_map[AD_strategy_name]
        detector_func_pointers.append(detector_f)
        is_realtime_vec.append(is_realtime)

    # PREPARE TIMESTEPS FOR OFFLINE TRAINING: DATAAUG, AE, SIMPLE, ROCKET
    start_time=time.time()



    # BUILD THE MODEL AND RETURNS ITS PREDICTIONS
    ensemble_size=len(AD_strategies_name)
    y_test_pred=None
    for detector_f,is_realtime in zip(detector_func_pointers,is_realtime_vec):
        #TODO: investigate further a better architecture (how best handle optional and default arguments ?).
        #assert(is_realtime!=train_dataset["is_framed"] and is_realtime!=test_dataset["is_framed"])

        if AD_hyperparameters is None:
            x_test_pred_local = detector_f(train_dataset, test_dataset)
        else:
            x_test_pred_local = detector_f(train_dataset, test_dataset, AD_hyperparameters)

        if y_test_pred is None:
            y_test_pred=x_test_pred_local/ensemble_size
        else:
            y_test_pred+=x_test_pred_local/ensemble_size
    y_test_pred=y_test_pred>proba_thresh
    enlapsed_time=round(time.time()-start_time,3)
    experimental_result.update({"time":enlapsed_time})

    from insight import confusion_matrix_and_F1


    # Warning: RT and Offline strategy do not exactly predict the same length of datasamples
    # Today, it is impossible to make an ensemble between those two strategies
    if any(is_realtime_vec):
        y_test = test_dataset['y']
        AD_quality_preds = confusion_matrix_and_F1(y_test_pred, y_test)
        frame_size=1
    else:
        if data_prep_info["name"]=="NAB":
            frame_size=data_prep_info["frame_size"]
            y_test=test_dataset['y'][frame_size-1:]
            AD_quality_preds=confusion_matrix_and_F1(y_test_pred,y_test)
        elif data_prep_info["name"]=="DCASE":
            frame_size=data_prep_info["frame_size"]
            y_test=test_dataset['y']
            AD_quality_preds=confusion_matrix_and_F1(y_test_pred,y_test)

    experimental_result.update(AD_quality_preds)
    details={"train_dataset":train_dataset,"test_dataset":test_dataset,"frame_size":frame_size,"y_test_pred":y_test_pred}

    return experimental_result, details

if __name__=="__main__":
    # GET DATA
    from extract_data import extract_one_dataset

    #extract_datasets("./NAB-datasets/")
    dataset=extract_one_dataset("./data/NAB/", "artificialWithAnomaly/art_daily_jumpsdown.csv")

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