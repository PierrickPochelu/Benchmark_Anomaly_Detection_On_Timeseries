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