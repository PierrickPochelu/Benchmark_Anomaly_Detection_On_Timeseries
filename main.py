import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # The CPU is about as fast as the GPU for TS applications and contains much more memory


from experiments import LAUNCH_EXPERIMENTS_AT_SCALE
from strat_map import detector_strat_map,feature_extractor_strat_map
if __name__=="__main__":
    from extract_data import extract_datasets
    datasets=extract_datasets("./data/NAB/")
    fe_strat_name="LSTM_AE"
    detector_strat_name= "IFOREST" # Isolation Forest implemented by scikit-learn
    #for detector_strat_name in ["STUMP"]: # for each known detection strategy

    print("Compute the mosaic with the strategy: ", detector_strat_name)
    results=LAUNCH_EXPERIMENTS_AT_SCALE(fe_strat_name, detector_strat_name, datasets)
    print(fe_strat_name, detector_strat_name, results)