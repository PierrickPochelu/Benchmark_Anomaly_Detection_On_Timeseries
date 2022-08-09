import numpy as np
import sys


def standardization(x, train_rate):
    x_train=x[:int(len(x)*train_rate)]
    return (x-np.mean(x_train))/(np.std(x_train)+1e-7)


def get_norm_strat(normalize_strategy_name):
    S={}
    S["STD"]=standardization
    # TODO: Proposed new technics later... Example: 1st derivate + standardization


    return S[normalize_strategy_name]

def normalize(dataset, train_rate, normalize_strategy_name):
    norm_func=get_norm_strat(normalize_strategy_name)
    raw_x=dataset["x"]
    normed_x=norm_func(raw_x, train_rate)
    return {"y":dataset["y"],"x":normed_x}

def normalize_and_split(nonframed_nonsplitted_dataset,
                        train_rate,
                        normalize_strategy_name):
    nonframed_nonsplitted_dataset=normalize(nonframed_nonsplitted_dataset, train_rate, normalize_strategy_name)

    # split
    n=len(nonframed_nonsplitted_dataset["x"])
    x_train,y_train= nonframed_nonsplitted_dataset["x"][:int(n * train_rate)], nonframed_nonsplitted_dataset["y"][:int(n * train_rate)]
    x_test,y_test= nonframed_nonsplitted_dataset["x"][int(n * train_rate):], nonframed_nonsplitted_dataset["y"][int(n * train_rate):]

    train_dataset={"x":x_train,"y":y_train}
    test_dataset={"x":x_test,"y":y_test}

    return train_dataset, test_dataset