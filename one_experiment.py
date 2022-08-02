import numpy as np
from typing import Union,Tuple,Optional
from strat_map import detector_strat_map, feature_extractor_strat_map
from normalize_data import normalize_and_split

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

def parametrized_feature_detector(**args):
    features_extractor=args["features_extractor"]
    if args["FE_hyperparameters"] is None:
        return features_extractor(args["train_dataset"], args["test_dataset"], args["frame_size"]) # default hyperparameters given
    else:
        return features_extractor(args["train_dataset"], args["test_dataset"], args["frame_size"],args["FE_hyperparameters"])

def EXPERIMENT(
        dataset:dict,
        dataset_name:str,
        train_test_split_rate:float,
        frame_size:int,
        normalize_strategy_name:str="STD",
        FE_frame_strategy_name:str="IDENTITY",
        FE_hyperparameters:Optional[dict]=None, #If none it means default ones
        AD_strategies_name:Union[str,list]="IFOREST",
        AD_hyperparameters:Optional[dict]=None
               )->Tuple[Optional[dict],Optional[dict]]:

    # NORMALIZE: strategies: STD
    try:
        train_dataset,test_dataset=normalize_and_split(dataset,train_test_split_rate,normalize_strategy_name=normalize_strategy_name)
        #print(f"train x: {train_dataset['x'].shape}, train y: {train_dataset['y'].shape}")
        #print(f"test x: {test_dataset['x'].shape}, test y: {test_dataset['y'].shape}")

        # CHECK THE DATASET
        check_dataset(train_dataset, test_dataset)
    except Exception as err:
        print(f"Exception with dataset {dataset_name} type:{type(err)} msg:{err}")
        return None, None

    # EXTRACTOR STRATEGIES
    if isinstance(AD_strategies_name,str):
        AD_strategies_name=[AD_strategies_name] # force it as a list to handle ensembles

    detector_func_pointers=[]
    is_realtime_vec=[]
    for AD_strategy_name in AD_strategies_name:
        detector_f, is_realtime = detector_strat_map[AD_strategy_name]
        detector_func_pointers.append(detector_f)
        is_realtime_vec.append(is_realtime)

    features_extractor=feature_extractor_strat_map[FE_frame_strategy_name]


    # PREPARE TIMESTEPS FOR OFFLINE TRAINING: DATAAUG, AE, SIMPLE, ROCKET
    # TODO: investigate further a better architecture
    if any(is_realtime_vec):
        x_train_frames,x_test_frames=parametrized_feature_detector(train_dataset=train_dataset,
                                                                   test_dataset=test_dataset,
                                                                   frame_size=frame_size,
                                                                   features_extractor=features_extractor,
                                                                   FE_hyperparameters=FE_hyperparameters)


    # BUILD THE MODEL AND RETURNS ITS PREDICTIONS
    ensemble_size=len(AD_strategies_name)
    y_test_pred=None
    for detector_f,is_realtime in zip(detector_func_pointers,is_realtime_vec):


        #TODO: investigate further a better architecture (how best handle optional and default arguments ?).
        if is_realtime:
            x_test_pred_local = detector_f(train_dataset, test_dataset) if AD_hyperparameters is None \
                else detector_f(x_train_frames, x_test_frames)

        else:
            x_test_pred_local=detector_f(x_train_frames,x_test_frames) if AD_hyperparameters is None \
                else detector_f(x_train_frames,x_test_frames)

        if y_test_pred is None:
            y_test_pred=x_test_pred_local/ensemble_size
        else:
            y_test_pred+=x_test_pred_local/ensemble_size
    y_test_pred=y_test_pred>0.5


    from insight import confusion_matrix_and_F1

    # TODO potential improvement:
    # RT and Offline strategy do not exactly predict the same length of datasamples
    # Today, it is impossible to make an ensemble between those two strategies
    if any(is_realtime_vec):
        y_test = test_dataset['y']
        stats = confusion_matrix_and_F1(y_test_pred, y_test)
        frame_size=1
    else:
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