import pandas
import os
import numpy as np
from nab.corpus import Corpus
from nab.labeler import LabelCombiner, CorpusLabel
from read_data.normalize_data import normalize_and_split
from strat_map import feature_extractor_strat_map
WINDOW_SIZE = 0.1
TRAIN_TEST_RATE_SPLIT = 0.15
NAB_THRESHOLD = 0.5


def get_NAB_datasets_labels(NAB_path):

    NAB_raw_data_directory_path=os.path.join(NAB_path,"data")
    NAB_label_directory_path=os.path.join(os.path.join(NAB_path,"labels"), "raw")
    NAB_combined_windows_path=os.path.join(os.path.join(NAB_path,"labels"), "combined_windows.json")
    NAB_combined_labels_path=os.path.join(os.path.join(NAB_path,"labels"), "combined_labels.json")

    assert(os.path.exists(NAB_raw_data_directory_path))
    assert(os.path.exists(NAB_label_directory_path))


    #print("Getting corpus.")
    corpus = Corpus(NAB_raw_data_directory_path)#warning: it fails with relative path

    #print("Creating LabelCombiner.")
    labelCombiner = LabelCombiner(NAB_label_directory_path, corpus,
                                  threshold=NAB_THRESHOLD, windowSize=WINDOW_SIZE,probationaryPercent=TRAIN_TEST_RATE_SPLIT,
                                  verbosity=0)

    #print("Combining labels.")
    labelCombiner.combine()

    #print("Writing combined labels files.")
    labelCombiner.write(NAB_combined_labels_path, NAB_combined_windows_path)

    #print("Attempting to load objects as a test.")
    corpusLabel = CorpusLabel(NAB_combined_windows_path, corpus)
    corpusLabel.validateLabels()

    #print("Successfully combined labels!")
    #print("Resulting windows stored in:", NAB_combined_windows_path)

    #print("Convert labels into numpy")
    corpusLabel.getLabels()
    dataframe_labels=corpusLabel.labels

    y={}
    for timeserie_name,v in dataframe_labels.items():
        y[timeserie_name]=v.values[:,1].astype(np.float32)

    return y

def get_NAB_datasets_data(NAB_path):
    NAB_raw_data_directory_path = os.path.join(NAB_path, "data")
    corpus = Corpus(NAB_raw_data_directory_path)

    x={}
    for timeserie_name in corpus.getDataFiles():
        absolute_data_path=os.path.join(NAB_raw_data_directory_path,timeserie_name)
        dataframe=pandas.read_csv(absolute_data_path)
        x[timeserie_name]=np.array(dataframe["value"])
    return x

def join(x_info,y_info):
    if len(x_info)!=len(y_info):
        print(f"WARNING in join() function. Unexpected behaviour: nb x files: {len(x_info)} nb y info:{len(y_info)}")

    keys=x_info.keys()
    dataset={}
    for k in keys:
        x=x_info[k]
        y=y_info[k]
        dataset[k]={"x":x,"y":y}
    return dataset

def display_dataset(dataset):
    for k,v in dataset.items():
        x,y=v["x"],v["y"]
        print(k)
        print(f"x: {x.shape} mean(x)={np.mean(x)} std(x)={np.std(x)} , "
              f"y: {y.shape} mean(y)={np.mean(y)}")

        print("---------")

def extract_NAB_datasets(NAB_path):
    NAB_path=os.path.abspath(NAB_path)
    y_info=get_NAB_datasets_labels(NAB_path)
    x_info=get_NAB_datasets_data(NAB_path)
    dataset=join(x_info,y_info)
    return dataset




def extract_one_dataset(NAB_path,dataset_name):
    NAB_path=os.path.abspath(NAB_path)
    dataset_val=extract_NAB_datasets(NAB_path)[dataset_name]
    return dataset_val


def check_dataset(train_dataset,test_dataset): # May raise ValueError
    # Check the beginnin is anomaly-free
    if np.sum(train_dataset["y"])!=0:
        print("The begginning of the timeseries should be anomaly-free")
        return False
    # Check nan values
    nb_nan_in_data=np.sum(np.isnan(train_dataset["x"])==1)+np.sum(np.isnan(test_dataset["x"])==1)
    if nb_nan_in_data>0:
        print(f"Error {nb_nan_in_data} Nan value(s) have been fond in x")
        return False
    nb_nan_in_labels=np.sum(np.isnan(train_dataset["y"])==1)+np.sum(np.isnan(test_dataset["y"])==1)
    if nb_nan_in_data>0:
        print(f"Error {nb_nan_in_labels} Nan value(s) have been fond in x")
        return False
    return True




def NAB_datasets_generator(NAB_path:str, dataset_prep_info:dict):
    raw_NAB_datasets=extract_NAB_datasets(NAB_path)
    frame_size=dataset_prep_info["frame_size"]
    FE_name=dataset_prep_info["FE_name"]
    FE_hyperparameters=dataset_prep_info.get("FE_hp",None)
    features_extractor=feature_extractor_strat_map[FE_name]

    def gen():
        for dataset_name, dataset in raw_NAB_datasets.items():

            nonframed_train_dataset, nonframed_test_dataset = normalize_and_split(dataset,
                                                          train_rate=TRAIN_TEST_RATE_SPLIT,
                                                          normalize_strategy_name="STD")
            if FE_hyperparameters is None:
                train_dataset, test_dataset = features_extractor(train_dataset=nonframed_train_dataset,
                                                             test_dataset=nonframed_test_dataset,
                                                             frame_size=frame_size)
            else:
                train_dataset, test_dataset = features_extractor(train_dataset=nonframed_train_dataset,
                                                             test_dataset=nonframed_test_dataset,
                                                             frame_size=frame_size,
                                                             hyperparameters=FE_hyperparameters)
            # CHECK THE DATASET
            if not check_dataset(train_dataset, test_dataset):
                pass # we go to the next iteration
            else:
                yield train_dataset,test_dataset
    return gen()
if __name__=="__main__":
    dataset_generator=NAB_datasets_generator("../data/NAB/",{"frame_size":128,"FE_name":"FRAMED","FE_hp":None})
    for train_dataset,test_dataset in dataset_generator:
        print("train mediane:", np.median(train_dataset["x"]))
        print("test_mediane: ", np.median(test_dataset["x"]))
