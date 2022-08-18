
from one_experiment import experiment_ts_builder
from read_data.read_dataset import read_and_prepare_dataset
from strat_map import detector_strat_map,feature_extractor_strat_map
AD_strat_name = "ONESVM"
data_prep_info = {"name": "NAB",
                  "path": "data/NAB_SMALL/",
                  "FE_name": "FRAMED",
                  "frame_size": 128}
AD_strategy, _ = detector_strat_map[AD_strat_name]

train_datasets_generator = read_and_prepare_dataset(
    data_prep_info)  # /!\ MEMORY CONSUMPTION. A generator load and preprocess the timeserie(s)

exp = experiment_ts_builder(AD_strategy=AD_strategy, AD_hyperparameters={})

(train_dataset, test_dataset) = next(train_datasets_generator)

preds=exp.fit_and_predict(train_dataset, test_dataset)

res,_=exp.evaluate(preds,test_dataset)

print(res)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # The CPU is about as fast as the GPU for TS applications and contains much more memory


from experiments import LAUNCH_EXPERIMENTS_AT_SCALE
if __name__=="__main__":

    fe_strat_name="FRAMED"
    detector_strat_name= "ONESVM"

    data_prep_info={"name":"DCASE", "path":"./data/DCASE/", "FE_name":fe_strat_name,"frame_size":4096*4}

    print("Compute the mosaic with the strategy: ", detector_strat_name)
    results=LAUNCH_EXPERIMENTS_AT_SCALE(fe_strat_name, detector_strat_name, data_prep_info)
    print(fe_strat_name, detector_strat_name, results)
    
"""