import numpy as np
from typing import *
import os
from insight import plot_curves, mosaic
from one_experiment import EXPERIMENT

from read_data.read_dataset import read_and_prepare_dataset

MOSAIC=False

# LARGE SCALE INSIGHT
def LAUNCH_EXPERIMENTS_AT_SCALE(feature_extractor_name:str, detector_name:str, data_prep_info:dict):
    experimenta_result={}
    tmp_dir="tmp"
    os.makedirs("tmp", exist_ok=True)
    media_dir="media"
    os.makedirs("media",exist_ok=True)


    AD_hyperparameters=None
    proba_thresh=None

    # NORMALIZE: strategies: STD
    train_datasets_generator = read_and_prepare_dataset(data_prep_info) #/!\ MEMORY CONSUMPTION. TODO a generator would be more memory efficient

    datasets_name=data_prep_info["name"]

    paths_for_mosaic=[] # we will build a beautiful mosaic
    stats_for_mosaic=[]
    for train_dataset,test_dataset in train_datasets_generator:
        name = datasets_name.replace(os.sep, "_").split(".")[0] + "_"+detector_name
        path = os.path.join("tmp", name + ".png")
        stat,details=EXPERIMENT(train_dataset=train_dataset,
                                test_dataset=test_dataset,
                                data_prep_info=data_prep_info,
                                AD_strategies_name=detector_name,
                                AD_hyperparameters=AD_hyperparameters,
                                proba_thresh=0.5)

        print(datasets_name, " stats:", stat)
        stats_for_mosaic.append(stat)
        if stat is not None and MOSAIC:
            # Monitor
            paths_for_mosaic.append(path)

            txt = name + "\nF1-score:" + str(stat["f1"])
            plot_curves(x_train=details["train_dataset"]["x"],
                        x_test=details["test_dataset"]["x"],
                        y_test=details["test_dataset"]["y"],
                        y_pred=details["y_test_pred"],
                        frame_size=details["frame_size"],
                        path=path, txt=txt)

    # compute global results
    tp=sum([stat["tp"] for stat in stats_for_mosaic])
    tn=sum([stat["tn"] for stat in stats_for_mosaic])
    fp=sum([stat["fp"] for stat in stats_for_mosaic])
    fn=sum([stat["fn"] for stat in stats_for_mosaic])
    enlapsed_time=round(sum([stat["time"] for stat in stats_for_mosaic]),3)
    mean_f1_scores=round(np.mean([stat["f1"] for stat in stats_for_mosaic]),4)

    # display it
    experimental_resut={"time":enlapsed_time, "tp":tp, "tn":tn, "fp":fp, "fn":fn, "f1":mean_f1_scores}
    if MOSAIC:
        mosaic_name=f"{feature_extractor_name}_{detector_name}_mosaic.png"
        mosaic_path=os.path.join(media_dir,mosaic_name)
        mosaic(paths_for_mosaic, mosaic_path, experimental_resut)

        # delete cached files used to compute the mosaic
        for fname in os.listdir(tmp_dir):
            fpath=os.path.join(tmp_dir,fname)
            if os.path.exists(fpath):
                os.remove(fpath)
    print("ok")
    return experimental_resut