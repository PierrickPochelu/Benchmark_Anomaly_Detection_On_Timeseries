import numpy as np
from typing import *
import os
from insight import plot_curves, mosaic
from one_experiment import experiment_ts_builder
from strat_map import detector_strat_map
from read_data.read_dataset import read_and_prepare_dataset

MOSAIC=False

def aggregate_stats(stats:List[Dict])->Dict:
    tp=sum([stat["tp"] for stat in stats])
    tn=sum([stat["tn"] for stat in stats])
    fp=sum([stat["fp"] for stat in stats])
    fn=sum([stat["fn"] for stat in stats])
    enlapsed_time=round(sum([stat["time"] for stat in stats]),2)

    f1_scores=[]
    acc_scores=[]
    rocauc_scores=[]
    proposed_threshs,new_f1s,new_accs=[],[],[]
    for stat in stats:
        acc_scores.append(stat["acc"])
        if stat["f1"]==-1:
            f1_scores.append(stat["acc"])
        else:
            f1_scores.append(stat["f1"])
        if stat["rocauc"]!=-1:
            rocauc_scores.append(stat["rocauc"])
            proposed_threshs.append(stat["proposed_thresh"])
            new_f1s.append(stat["new_f1"])
            new_accs.append(stat["new_acc"])

    # mean and round
    mean_f1_scores=np.round(np.mean(f1_scores),4)
    mean_acc_scores = np.round(np.mean(acc_scores), 4)
    mean_rocauc_scores = np.round(np.mean(rocauc_scores), 4)
    prob_thresh=np.round(np.mean(proposed_threshs),4)
    new_f1 = np.round(np.mean(acc_scores), 4)
    new_acc = np.round(np.mean(rocauc_scores), 4)
    global_info={"time":enlapsed_time, "tp":tp, "tn":tn, "fp":fp, "fn":fn,
                        "f1":mean_f1_scores, "acc":mean_acc_scores, "rocauc":mean_rocauc_scores,
                        "prob_thresh":prob_thresh,"new_f1":new_f1,"new_acc":new_acc
                 }
    return global_info



# LARGE SCALE INSIGHT
def LAUNCH_EXPERIMENTS_AT_SCALE(data_prep_info:dict, detector_name:str, AD_hyperparameters:dict):
    tmp_dir="tmp"
    os.makedirs("tmp", exist_ok=True)
    media_dir="media"
    os.makedirs("media",exist_ok=True)

    # NORMALIZE: strategies: STD
    train_datasets_generator = read_and_prepare_dataset(data_prep_info) #/!\ MEMORY CONSUMPTION. A generator load and preprocess the timeserie(s)

    datasets_name=data_prep_info["name"]

    paths_for_mosaic=[] # we will build a beautiful mosaic
    stats=[]
    for train_dataset,test_dataset in train_datasets_generator:
        name = datasets_name.replace(os.sep, "_").split(".")[0] + "_"+detector_name
        path = os.path.join("tmp", name + ".png")

        # CREATE AND LAUNCH THE EXPERIMEN T
        exp=experiment_ts_builder(detector_name,AD_hyperparameters)
        preds=exp.fit_and_predict(train_dataset,test_dataset)
        stat,details=exp.evaluate(preds,test_dataset)


        print(detector_name, " stat:", stat)
        stats.append(stat)
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
    experimental_resut=aggregate_stats(stats)


    if MOSAIC:
        feature_extractor_name=data_prep_info["FE_name"]
        mosaic_name=f"{feature_extractor_name}_{detector_name}_mosaic.png"
        mosaic_path=os.path.join(media_dir,mosaic_name)
        mosaic(paths_for_mosaic, mosaic_path, experimental_resut)

        # delete cached files used to compute the mosaic
        for fname in os.listdir(tmp_dir):
            fpath=os.path.join(tmp_dir,fname)
            if os.path.exists(fpath):
                os.remove(fpath)

    return experimental_resut
