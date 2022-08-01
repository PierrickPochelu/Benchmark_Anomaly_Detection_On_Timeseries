import pandas
import os
import numpy as np
from nab.corpus import Corpus
from nab.labeler import LabelCombiner, CorpusLabel


def get_NAB_datasets_labels(NAB_path):
    windowSize = 0.1
    probationaryPercent = 0.15
    threshold=0.5

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
                                  threshold=threshold, windowSize=windowSize,probationaryPercent=probationaryPercent,
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

def extract_datasets(NAB_path):
    NAB_path=os.path.abspath(NAB_path)
    y_info=get_NAB_datasets_labels(NAB_path)
    x_info=get_NAB_datasets_data(NAB_path)
    dataset=join(x_info,y_info)
    return dataset

def extract_one_dataset(NAB_path,dataset_name): #TODO: computing time can be widely improved
    NAB_path=os.path.abspath(NAB_path)
    dataset_val=extract_datasets(NAB_path)[dataset_name]
    return dataset_val


if __name__=="__main__":
    dataset=extract_datasets("data/NAB/")
    display_dataset(dataset)
