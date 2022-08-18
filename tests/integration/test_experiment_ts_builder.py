import unittest
from one_experiment import experiment_ts_builder
from read_data.read_dataset import read_and_prepare_dataset
from strat_map import detector_strat_map,feature_extractor_strat_map
import os

class Test(unittest.TestCase):
    def test_one_svm(self):
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

if __name__ == '__main__':
    unittest.main()