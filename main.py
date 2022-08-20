import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # The CPU is about as fast as the GPU for TS applications and contains much more memory


from experiments import LAUNCH_EXPERIMENTS_AT_SCALE

if __name__=="__main__":
    for fe_strat_name,frame_size in [("FRAMED",128),("LOGMELSPECTR",4096)]:
        for detector_strat_name in ["IFOREST","ONESVM","ELLIPTIC","DENSE_AE","LSTM_AE","CONV_AE"]:
            hp={"batch_size":128}
            data_prep_info={"name":"DCASE", "path":"./data/DCASE/", "frame_size":frame_size, "FE_name":fe_strat_name,
                            "nb_frames_per_file":100, "max_files": 100} #used for DCASE dataset only

            data_prep_info.update(hp) # usefull for deep learning features extractor
            results=LAUNCH_EXPERIMENTS_AT_SCALE(data_prep_info, detector_strat_name, hp)
            print(fe_strat_name, detector_strat_name, results)

            with open("res.txt","a") as f:
                txt=fe_strat_name+" "+detector_strat_name+" "+str(results)+"\n"
                f.write(txt)
"""
if __name__=="__main__":

    fe_strat_name="DENSE_AE"
    detector_strat_name= "IFOREST"

    hp={"batch_size":50,"epochs":1}
    data_prep_info={"name":"DCASE", "path":"./data/DCASE_SMALL/", "frame_size":1024, "FE_name":fe_strat_name,
                    "nb_frames_per_file":100, "max_files": 100} #used for DCASE dataset only

    data_prep_info.update(hp) # usefull for deep learning features extractor
    results=LAUNCH_EXPERIMENTS_AT_SCALE(data_prep_info, detector_strat_name, hp)
    print(fe_strat_name, detector_strat_name, results)


    #FRAMED DENSE_AE {'time': 4883.93, 'tp': 2243, 'tn': 13500, 'fp': 2100, 'fn': 12457, 'f1': 0.1991, 'acc': 0.5196, 'rocauc': 0.5049}
"""
