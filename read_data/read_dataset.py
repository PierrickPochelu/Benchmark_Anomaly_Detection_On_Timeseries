from read_data.read_NAB import NAB_datasets_generator
from read_data.read_DCASE import DCASE_datasets_generator
from typing import *
import numpy as np

#TODO: implement it with a generator if memory fails
def read_and_prepare_dataset(dataset_info:dict)->Dict[str,Dict]:

    dataset_name=dataset_info["name"]
    dataset_path=dataset_info["path"]


    if dataset_name=="NAB":
        datasets_generator=NAB_datasets_generator(dataset_path,dataset_info)
    elif dataset_name=="DCASE":
        if dataset_info["FE_name"]=="NONFRAMED":
            raise ValueError('error in read_and_prepare_dataset() DCASE dataset is not compatible with "NONFRAMED" features extractor strategy ')
        datasets_generator=DCASE_datasets_generator(dataset_path,dataset_info)
    else:
        raise ValueError(f"Error in read_dataset(), dataset_name={dataset_name} is not known")


    return datasets_generator

