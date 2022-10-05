import unittest
from one_experiment import experiment_ts_builder
from read_data.read_dataset import read_and_prepare_dataset
from strat_map import detector_strat_map,feature_extractor_strat_map
import os
import numpy as np

class Test(unittest.TestCase):
    def test_DCASE_SMALL(self):
        AD_strat_name="ONESVM"
        data_prep_info = {"name": "DCASE",
                          "path": "data/DCASE_SMALL/",
                          "FE_name": "FRAMED",
                          "frame_size": 4096 * 4}

        self.assertTrue(os.path.exists(data_prep_info["path"]),os.path.abspath("."))
        AD_strategy, _ = detector_strat_map[AD_strat_name]

        train_datasets_generator = read_and_prepare_dataset(
            data_prep_info)  # /!\ MEMORY CONSUMPTION. A generator load and preprocess the timeserie(s)

        exp = experiment_ts_builder(AD_strategy=AD_strategy, AD_hyperparameters={})

        (train_dataset, test_dataset) = next(train_datasets_generator)

        preds = exp.fit_and_predict(train_dataset, test_dataset)

        res, _ = exp.evaluate(preds, test_dataset)
        print(res)
        self.assertTrue("f1" in res)

    def test_NAB_SMALL(self):
        from one_experiment import experiment_ts_builder
        from read_data.read_dataset import read_and_prepare_dataset
        from strat_map import detector_strat_map, feature_extractor_strat_map
        AD_strat_name = "IFOREST"
        data_prep_info = {"name": "NAB",
                          "path": "data/NAB_SMALL/",
                          "FE_name": "FRAMED",
                          "frame_size": 1}
        AD_strategy, _ = detector_strat_map[AD_strat_name]

        train_datasets_generator = read_and_prepare_dataset(
            data_prep_info)  # /!\ MEMORY CONSUMPTION. A generator load and preprocess the timeserie(s)

        exp = experiment_ts_builder(AD_strategy=AD_strategy, AD_hyperparameters={})

        f1_scores = []
        acc_scores = []
        rocauc_scores = []
        for (train_dataset, test_dataset) in train_datasets_generator:
            preds = exp.fit_and_predict(train_dataset, test_dataset)
            res, _ = exp.evaluate(preds, test_dataset)
            acc_scores.append(res["acc"])

            if res["f1"]!=-1:
                f1_scores.append(res["f1"])
            if res["rocauc"]!=-1:
                rocauc_scores.append(res["rocauc"])
        self.assertTrue(1>np.mean(acc_scores)>0)
        self.assertTrue(1>np.mean(f1_scores)>0)
        self.assertTrue(1>np.mean(rocauc_scores)>0)

    def test_launch_experiments_at_scale(self):
        from experiments import LAUNCH_EXPERIMENTS_AT_SCALE
        fs = 128
        fe_strat_name = "FRAMED"
        for detector_strat_name, frame_size in [("RE", 1),
                                                ("IFOREST", fs),
                                                ("ONESVM", fs),
                                                ("ELLIPTIC", fs),
                                                ("DENSE_AE", fs),
                                                ("LSTM_AE", fs),
                                                ("CONV_AE", fs)]:
            hp = {"batch_size": 50, "epochs": 1}
            data_prep_info = {"name": "DCASE", "path": "./data/DCASE_SMALL/", "frame_size": frame_size,
                              "FE_name": fe_strat_name,
                              "nb_frames_per_file": 2, "max_files": 10}  # used for DCASE dataset only

            data_prep_info.update(hp)  # usefull for deep learning features extractor
            results = LAUNCH_EXPERIMENTS_AT_SCALE(data_prep_info, detector_strat_name, hp)
            print(fe_strat_name, detector_strat_name, results)





if __name__ == '__main__':
    unittest.main()