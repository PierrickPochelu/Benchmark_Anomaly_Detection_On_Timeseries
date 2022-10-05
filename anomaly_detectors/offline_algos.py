import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import NearestNeighbors


def sklearn_post_process(model,train_dataset,test_dataset):
    y_test_sklearn=model.predict(test_dataset)
    test_rings=(-1*y_test_sklearn+1)/2
    return test_rings


def compute_hyperparameters(wanted_hp,default_hp):
    hyperparameters={}
    for k,v in default_hp.items():
        hyperparameters[k]=wanted_hp.get(k,v)
    return hyperparameters

def isolation_forest(train_dataset, test_dataset,wanted_hyperparameters={}):
    hyperparameters=compute_hyperparameters(wanted_hyperparameters,{"n_estimators":128,"bootstrap":False,"contamination":0.01})
    if hyperparameters["bootstrap"]:
        model=IsolationForest(**hyperparameters,max_features=0.5)
    else:
        model = IsolationForest(**hyperparameters, max_features=1.0)
    model.fit(train_dataset["x"])
    return sklearn_post_process(model,train_dataset["x"],test_dataset["x"])


def knn(train_dataset, test_dataset,wanted_hyperparameters={}):
    hyperparameters=compute_hyperparameters(wanted_hyperparameters,{"n_neighbors":5,"contamination":0.01})
    model=NearestNeighbors(n_neighbors=hyperparameters["n_neighbors"])
    model.fit(train_dataset["x"])

    # get stats
    distances_train, _ = model.kneighbors(train_dataset["x"])
    anomaly_score_train = np.sum(distances_train, axis=1)
    thresh=np.quantile(anomaly_score_train, 1.-hyperparameters["contamination"])
    min_train=np.min(anomaly_score_train)
    max_train=np.max(anomaly_score_train)

    # get test
    distances_test, _ = model.kneighbors(test_dataset["x"])
    anomaly_score_test = np.sum(distances_test, axis=1)



    from anomaly_detectors.autoencoder import _from_loss_to_proba
    out=_from_loss_to_proba(anomaly_score_test,thresh,min_train, max_train)
    return out

def oneclass_svm(train_dataset, test_dataset, wanted_hyperparameters={}):
    hyperparameters=compute_hyperparameters(wanted_hyperparameters,{"kernel": "linear"})
    model=OneClassSVM(kernel=hyperparameters["kernel"])
    model.fit(train_dataset["x"])
    return sklearn_post_process(model,train_dataset["x"],test_dataset["x"])

def elliptic_envelope(train_dataset, test_dataset,wanted_hyperparameters={}):
    hyperparameters=compute_hyperparameters(wanted_hyperparameters,{"contamination": 0.01,"assume_centered":True})
    model=EllipticEnvelope(**hyperparameters)
    model.fit(train_dataset["x"])
    return sklearn_post_process(model,train_dataset["x"],test_dataset["x"])

def lof(train_dataset, test_dataset,wanted_hyperparameters={}):
    hyperparameters=compute_hyperparameters(wanted_hyperparameters,{"novelty":True})
    model=LocalOutlierFactor(**hyperparameters)
    model.fit(train_dataset["x"])
    return sklearn_post_process(model,train_dataset["x"],test_dataset["x"])

def _AE_reconstruction(deeplearning_techno, train_dataset, test_dataset, wanted_hyperparameters={}):
    from anomaly_detectors.autoencoder import AE
    model = AE(deeplearning_techno, wanted_hyperparameters)
    model.fit(train_dataset["x"])
    x_test_pred = model.predict(test_dataset["x"])
    del model
    return x_test_pred  # return problities and not boolean values

def conv_AE_reconstruction(train_dataset, test_dataset, hyperparameters={}):
    return _AE_reconstruction("CONV_AE", train_dataset, test_dataset, hyperparameters)
def LSTM_AE_reconstruction(train_dataset, test_dataset, hyperparameters={}):
    return _AE_reconstruction("LSTM_AE", train_dataset, test_dataset, hyperparameters)
def dense_AE_reconstruction(train_dataset, test_dataset, hyperparameters={}):
    return _AE_reconstruction("DENSE_AE", train_dataset, test_dataset, hyperparameters)

def conv2D_AE_reconstruction(train_dataset, test_dataset, hyperparameters={}):
    return _AE_reconstruction("2DCONVAE", train_dataset, test_dataset, hyperparameters)


def alwaystrue(train_dataset, test_dataset, wanted_hyperparameters={}):
    ones=np.ones(len(test_dataset["x"]),dtype=np.float32)
    return ones


if __name__=="__main__":



    train_dataset={"x":np.array([[0, 1, 2],
                            [1, 2, 3],
                            [2, 3, 4],
                            [3, 4, 5],
                            [4, 5, 6],
                             ], dtype=np.float32)}
    test_dataset = {"x":np.array([[5, 6, 2],
                               [6, 2, 8],
                               [2, 8, 9]], dtype=np.float32
                              )}



    test_rings=LSTM_AE_reconstruction(train_dataset,test_dataset,{})
    #test_rings=elliptic_envelope(train_dataset,test_dataset)
    print(test_rings)
