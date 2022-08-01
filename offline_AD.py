import numpy as np
from typing import Union,Tuple

def check_dataset(train_dataset,test_dataset): # May raise ValueError
    # Check the beginnin is anomaly-free
    if np.sum(train_dataset["y"])!=0:
        raise ValueError("The begginning of the timeseries should be anomaly-free")

    # Check nan values
    nb_nan_in_data=np.sum(np.isnan(train_dataset["x"])==1)+np.sum(np.isnan(test_dataset["x"])==1)
    if nb_nan_in_data>0:
        raise ValueError(f"Error {nb_nan_in_data} value(s) have been fond in x")
    nb_nan_in_labels=np.sum(np.isnan(train_dataset["y"])==1)+np.sum(np.isnan(test_dataset["y"])==1)
    if nb_nan_in_data>0:
        raise ValueError(f"Error {nb_nan_in_labels} value(s) have been fond in x")

def OFFLINE_AD(
        dataset:dict,
        train_test_split_rate:float,
        frame_size:int,
        normalize_strategy_name:str,
        FE_frame_strategy_name:str,
        AD_strategies_name:Union[str,list],
               )->Tuple[dict,dict]:

    # NORMALIZE: strategies: STD
    from normalize_data import normalize_and_split
    train_dataset,test_dataset=normalize_and_split(dataset,train_test_split_rate,normalize_strategy_name=normalize_strategy_name)
    #print(f"train x: {train_dataset['x'].shape}, train y: {train_dataset['y'].shape}")
    #print(f"test x: {test_dataset['x'].shape}, test y: {test_dataset['y'].shape}")

    # CHECK THE DATASET
    check_dataset(train_dataset, test_dataset)

    # PREPARE TIMESTEPS FOR OFFLINE TRAINING: DATAAUG, AE, SIMPLE, ROCKET
    from offline_strategies.from_signal_to_frames import from_signal_to_frame
    x_train_frames, x_test_frames = from_signal_to_frame(train_dataset, test_dataset, frame_size, frame_strategy_name=FE_frame_strategy_name)
    #print(f"Training frames: {x_train_frames.shape} min:{np.min(x_train_frames)} max:{np.max(x_train_frames)}")
    #print(f"Testining frames: {x_test_frames.shape} min:{np.min(x_test_frames)} max:{np.max(x_test_frames)}")

    # SELECT, BUILD THE MODEL AND RETURNS ITS PREDICTIONS
    from offline_strategies.algos import offline_AD_strategy
    if isinstance(AD_strategies_name,str):
        AD_strategies_name=[AD_strategies_name]

    ensemble_size=len(AD_strategies_name)
    y_test_pred=None
    for AD_strategy_name in AD_strategies_name:
        strat=offline_AD_strategy(AD_strategy_name)
        x_test_pred_local=strat(x_train_frames,x_test_frames)
        if y_test_pred is None:
            y_test_pred=x_test_pred_local/ensemble_size
        else:
            y_test_pred+=x_test_pred_local/ensemble_size
    y_test_pred=y_test_pred>0.5


    from insight import confusion_matrix_and_F1
    y_test_frames=test_dataset['y'][frame_size-1:]

    stats=confusion_matrix_and_F1(y_test_pred,y_test_frames)
    details={"train_dataset":train_dataset,"test_dataset":test_dataset,"frame_size":frame_size,
             "y_test_pred":y_test_pred}
    return stats, details

if __name__=="__main__":
    # GET DATA
    from extract_data import extract_one_dataset,extract_datasets
    #extract_datasets("./NAB-datasets/")
    dataset=extract_one_dataset("./data/NAB/", "artificialWithAnomaly/art_daily_jumpsdown.csv")

    for i in range(5):#test the reproducibility
        stats,details=OFFLINE_AD(dataset,
            train_test_split_rate=0.15,
            frame_size=128,
            normalize_strategy_name="STD",
            FE_frame_strategy_name="SIMPLE",
            AD_strategies_name=["AE" for i in range(1)] # we can make some ensembles! /!\ training time is multiplied
        )
        print(stats)
