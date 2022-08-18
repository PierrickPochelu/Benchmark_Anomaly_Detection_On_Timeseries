import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

def sklearn_post_process(model,train_dataset,test_dataset):
    y_test_sklearn=model.predict(test_dataset)
    test_rings=(-1*y_test_sklearn+1)/2
    return test_rings




def isolation_forest(train_dataset, test_dataset,hyperparameters={}):
    default_hyperparameters={"n_estimators":128,"bootstrap":False,"contamination":0.01}
    hyperparameters.update(default_hyperparameters)
    if hyperparameters["bootstrap"]:
        model=IsolationForest(**hyperparameters,max_features=0.5)
    else:
        model = IsolationForest(**hyperparameters, max_features=1.0)
    model.fit(train_dataset["x"])
    return sklearn_post_process(model,train_dataset["x"],test_dataset["x"])


def oneclass_svm(train_dataset, test_dataset, hyperparameters={}):
    default_hyperparameters = {"kernel": "linear"}
    hyperparameters.update(default_hyperparameters)
    model=OneClassSVM(kernel=hyperparameters["kernel"])
    model.fit(train_dataset["x"])
    return sklearn_post_process(model,train_dataset["x"],test_dataset["x"])

def elliptic_envelope(train_dataset, test_dataset,hyperparameters={}):
    default_hyperparameters = {"contamination": 0.01,"assume_centered":True}
    hyperparameters.update(default_hyperparameters)
    model=EllipticEnvelope(**hyperparameters)
    model.fit(train_dataset["x"])
    return sklearn_post_process(model,train_dataset["x"],test_dataset["x"])

def _AE_reconstruction(deeplearning_techno, train_dataset, test_dataset, hyperparameters={}):
    from offline_strategies.autoencoder import AE
    model = AE(deeplearning_techno, hyperparameters)
    model.fit(train_dataset["x"])
    x_test_pred = model.predict(test_dataset["x"])
    del model
    return x_test_pred  # return problities and not boolean values

def conv_AE_reconstruction(train_dataset, test_dataset, hyperparameters={}):
    return _AE_reconstruction("CONV_AE", train_dataset, test_dataset, hyperparameters)
def LSTM_AE_reconstruction(train_dataset, test_dataset,hyperparameters={}):
    return _AE_reconstruction("LSTM_AE", train_dataset, test_dataset, hyperparameters)
def dense_AE_reconstruction(train_dataset, test_dataset,hyperparameters={}):
    return _AE_reconstruction("DENSE_AE", train_dataset, test_dataset, hyperparameters)





if __name__=="__main__":
    timeseries=np.array([0,1,2,3,4,5,6,2,8,9])
    train_dataset=np.array([[0, 1, 2],
                            [1, 2, 3],
                            [2, 3, 4],
                            [3, 4, 5],
                            [4, 5, 6],
                             ], dtype=np.float32
                            )
    test_dataset = np.array([[5, 6, 2],
                               [6, 2, 8],
                               [2, 8, 9]], dtype=np.float32
                              )

    y=np.array([0,0,0,0,0,0,0,1,0,0])

    test_rings=LSTM_AE_reconstruction(train_dataset,test_dataset,{})
    #test_rings=elliptic_envelope(train_dataset,test_dataset)
    print(test_rings)