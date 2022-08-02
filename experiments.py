import numpy as np
import os
from insight import plot_curves, mosaic
from one_experiment import EXPERIMENT



# LARGE SCALE INSIGHT
def LAUNCH_EXPERIMENTS_AT_SCALE(feature_extractor_name, detector_name, datasets):

    paths=[] # we will build a beautiful mosaic
    stats=[]
    for dataset_name, dataset in datasets.items():
            stat,details=EXPERIMENT(dataset,
                                     dataset_name,
                                     train_test_split_rate=0.15,
                                     frame_size=128,
                                     normalize_strategy_name="STD",
                                     FE_frame_strategy_name=feature_extractor_name,
                                     AD_strategies_name=detector_name
                                     )
            if stat is not None:
                # Monitor
                print(dataset_name, " stats:", stat)
                name = dataset_name.replace(os.sep, "_").split(".")[0] + "_isolation_forest"
                path = os.path.join("media", name + ".png")
                paths.append(path)
                stats.append(stat)
                txt = name + "\nF1-score:" + str(stat["f1"])
                plot_curves(x_train=details["train_dataset"]["x"],
                            x_test=details["test_dataset"]["x"],
                            y_test=details["test_dataset"]["y"],
                            y_pred=details["y_test_pred"],
                            frame_size=details["frame_size"],
                            path=path, txt=txt)


    # compute global results
    tp=sum([stat["tp"] for stat in stats])
    tn=sum([stat["tn"] for stat in stats])
    fp=sum([stat["fp"] for stat in stats])
    fn=sum([stat["fn"] for stat in stats])
    mean_f1_scores=round(np.mean([stat["f1"] for stat in stats]),4)

    # display it
    txt_result=f"tp={tp} tn={tn} fp={fp} fn={fn} mean(F1_scores)={mean_f1_scores}"
    print(txt_result)
    mosaic(paths, f"{feature_extractor_name}_{detector_name}_mosaic.png", txt_result)

    # delete cache files used to compute the mosaic
    for p in paths:
        if os.path.exists(p):
            os.remove(p)
    print("ok")