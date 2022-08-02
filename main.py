import numpy as np
import os
from insight import plot_curves, mosaic
from one_experiment import EXPERIMENT

# LARGE SCALE INSIGHT
def LAUNCH_EXPERIMENTS_AT_SCALE(feature_extractor_name, detector_name, datasets):


    paths=[] # we will build a beautiful mosaic
    f1_scores=[]
    for dataset_name, dataset in datasets.items():
            stats,details=EXPERIMENT(dataset,
                                     dataset_name,
                                     train_test_split_rate=0.15,
                                     frame_size=128,
                                     normalize_strategy_name="STD",
                                     FE_frame_strategy_name=feature_extractor_name,
                                     AD_strategies_name=detector_name
                                     )
            if stats is not None:
                # Monitor
                print(dataset_name, " stats:", stats)
                name = dataset_name.replace(os.sep, "_").split(".")[0] + "_isolation_forest"
                path = os.path.join("media", name + ".png")
                paths.append(path)
                f1_scores.append(stats["f1"])
                txt = name + "\nF1-score:" + str(stats["f1"])
                plot_curves(x_train=details["train_dataset"]["x"],
                            x_test=details["test_dataset"]["x"],
                            y_test=details["test_dataset"]["y"],
                            y_pred=details["y_test_pred"],
                            frame_size=details["frame_size"],
                            path=path, txt=txt)

    mean_f1_scores=round(np.mean(f1_scores),4)
    print(mean_f1_scores)

    mosaic(paths, f"{feature_extractor_name}_{detector_name}_mosaic.png", f"Mean F1-score={mean_f1_scores}")

if __name__=="__main__":
    from extract_data import extract_datasets
    datasets=extract_datasets("./data/NAB/")
    feature_extractor="IDENTITY"
    detector="OSE"
    LAUNCH_EXPERIMENTS_AT_SCALE(feature_extractor,detector,datasets)