import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

def sklearn_post_process(model,x_train_frames,x_test_frames):
    y_test_sklearn=model.predict(x_test_frames)
    test_rings=(-1*y_test_sklearn+1)/2
    return test_rings




def isolation_forest(x_train_frames, x_test_frames):
    hyperparameters={"n_estimators":128,"bootstrap":False,"contamination":0.01}
    if hyperparameters["bootstrap"]:
        model=IsolationForest(**hyperparameters,max_features=0.5)
    else:
        model = IsolationForest(**hyperparameters, max_features=1.0)
    model.fit(x_train_frames)
    return sklearn_post_process(model,x_train_frames,x_test_frames)


def oneclass_svm(x_train_frames, x_test_frames):
    hyperparameters = {"kernel": "linear"}
    model=OneClassSVM(kernel=hyperparameters["kernel"])
    model.fit(x_train_frames)
    return sklearn_post_process(model,x_train_frames, x_test_frames)

def elliptic_envelope(x_train_frames, x_test_frames):
    hyperparameters = {"contamination": 0.01,"assume_centered":True}
    model=EllipticEnvelope(**hyperparameters)
    model.fit(x_train_frames)
    return sklearn_post_process(model,x_train_frames, x_test_frames)

def _AE_reconstruction(deeplearning_techno, x_train_frames, x_test_frames, hyperparameters={}):
    from offline_strategies.autoencoder import AE
    model = AE(deeplearning_techno, hyperparameters)
    model.fit(x_train_frames)
    x_test_pred = model.predict(x_test_frames)
    del model
    return x_test_pred  # return problities and not boolean values

def conv_AE_reconstruction(x_train_frames, x_test_frames, hyperparameters={}):
    return _AE_reconstruction("CONV_AE", x_train_frames, x_test_frames, hyperparameters)
def LSTM_AE_reconstruction(x_train_frames, x_test_frames,hyperparameters={}):
    return _AE_reconstruction("LSTM_AE", x_train_frames, x_test_frames, hyperparameters)
def dense_AE_reconstruction(x_train_frames, x_test_frames,hyperparameters={}):
    return _AE_reconstruction("DENSE_AE", x_train_frames, x_test_frames, hyperparameters)





if __name__=="__main__":
    timeseries=np.array([0,1,2,3,4,5,6,2,8,9])
    x_train_frames=np.array([[0, 1, 2],
                            [1, 2, 3],
                            [2, 3, 4],
                            [3, 4, 5],
                            [4, 5, 6],
                             ], dtype=np.float32
                            )
    x_test_frames = np.array([[5, 6, 2],
                               [6, 2, 8],
                               [2, 8, 9]], dtype=np.float32
                              )

    y=np.array([0,0,0,0,0,0,0,1,0,0])

    test_rings=LSTM_AE_reconstruction(x_train_frames,x_test_frames,{})
    #test_rings=elliptic_envelope(x_train_frames,x_test_frames)
    print(test_rings)