import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # The CPU is about as fast as the GPU for TS applications and contains much more memory

from read_data.read_dataset import read_and_prepare_dataset
from experiments import LAUNCH_EXPERIMENTS_AT_SCALE
from strat_map import detector_strat_map,feature_extractor_strat_map
if __name__=="__main__":

    fe_strat_name="FRAMED"
    detector_strat_name= "ONESVM" # Isolation Forest implemented by scikit-learn
    #for detector_strat_name in ["STUMP"]: # for each known detection strategy

    #data_prep_info={"name":"NAB","path":"./data/NAB/","FE_name":fe_strat_name,"frame_size":128}
    data_prep_info={"name":"DCASE", "path":"./data/DCASE/", "FE_name":"LOGMELSPECTR","frame_size":128}

    print("Compute the mosaic with the strategy: ", detector_strat_name)
    results=LAUNCH_EXPERIMENTS_AT_SCALE(fe_strat_name, detector_strat_name, data_prep_info)
    print(fe_strat_name, detector_strat_name, results)