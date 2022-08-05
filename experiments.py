import numpy as np
import time
import os
from insight import plot_curves, mosaic
from one_experiment import EXPERIMENT



# LARGE SCALE INSIGHT
def LAUNCH_EXPERIMENTS_AT_SCALE(feature_extractor_name, detector_name, datasets):
    experimenta_result={}
    tmp_dir="tmp"
    os.makedirs("tmp", exist_ok=True)
    media_dir="media"
    os.makedirs("media",exist_ok=True)

    paths_for_mosaic=[] # we will build a beautiful mosaic
    stats_for_mosaic=[]
    for dataset_name, dataset in datasets.items():
            stat,details=EXPERIMENT(dataset,
                                     dataset_name,
                                     train_test_split_rate=0.15,
                                     frame_size=128,
                                     normalize_strategy_name="STD",
                                     FE_frame_strategy_name=feature_extractor_name,
                                     AD_strategies_name=detector_name,
                                     proba_thresh=0.5
                                     )
            if stat is not None:
                # Monitor
                #print(dataset_name, " stats:", stat)
                name = dataset_name.replace(os.sep, "_").split(".")[0] + "_isolation_forest"
                path = os.path.join("tmp", name + ".png")

                paths_for_mosaic.append(path)
                stats_for_mosaic.append(stat)
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