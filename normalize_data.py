import numpy as np

def standardization(x, train_rate):
    x_train=x[:int(len(x)*train_rate)]

    return (x-np.mean(x_train))/np.std(x_train)


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

def normalize_and_split(dataset, train_rate, normalize_strategy_name):
    dataset=normalize(dataset, train_rate, normalize_strategy_name)

    # split
    n=len(dataset["x"])
    x_train,y_train=dataset["x"][:int(n*train_rate)], dataset["y"][:int(n*train_rate)]
    x_test,y_test=dataset["x"][int(n*train_rate):], dataset["y"][int(n*train_rate):]

    train_datasets={"x":x_train,"y":y_train}
    test_datasets={"x":x_test,"y":y_test}

    return train_datasets, test_datasets