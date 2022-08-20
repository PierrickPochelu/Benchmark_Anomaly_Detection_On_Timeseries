VERBOSITY=1
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow import keras
from keras.backend import clear_session
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np


def default_LSTM_hyperparameters():
    return {"init_filters": 32, "dropout_rate":0.2, "lr":0.001, "nb_layers":1,
            "batch_size":32,"l2_reg":0.001,"percentile":0.999,"epochs":100} # warning it takes longtime so 100 epochs is enough

def default_dense_hyperparameters():
    return {"init_filters": 32, "dropout_rate":0.0, "lr":0.001, "nb_layers":1,
            "batch_size":128,"l2_reg":0.001,"percentile":0.995,"epochs":1000}

def default_conv_hyperparameters():
    return {"K": 7, "K_reduc":0,
            "init_filters": 32, "dropout_rate":0.2, "lr":0.001, "nb_layers":4,
            "batch_size":128,"l2_reg":0.001,"percentile":0.999,"epochs":1000}

def _from_standardized_loss_to_proba(rescaled_reconstruction_loss, rescaled_thresh):
    try:
        # check
        assert (1. >= rescaled_reconstruction_loss >= 0.)
        assert (1. >= rescaled_thresh >= 0.)
    except AssertionError as e:
        print(f"WARNING in _from_loss_to_proba: {e} "
              f"value={rescaled_reconstruction_loss} thresh={rescaled_thresh}")

    if rescaled_reconstruction_loss > rescaled_thresh:
        reconstruction_proba = ((1. - 0.5) / (1 - rescaled_thresh)) * (
                    rescaled_reconstruction_loss - rescaled_thresh)+0.5
    elif rescaled_reconstruction_loss < rescaled_thresh:
        reconstruction_proba = ((0.5 - 0.) / (rescaled_thresh - 0.)) * (rescaled_reconstruction_loss - 0)
    else:
        reconstruction_proba = 0.5

    reconstruction_proba=np.clip(reconstruction_proba,0,1)
    return reconstruction_proba

def _from_loss_to_proba(raw_reconstruction_loss,raw_threshold, min_train, max_train):
    rescaled_thresh = (raw_threshold - min_train) / (max_train - min_train)
    rescaled_reconstruction_loss = ((raw_reconstruction_loss - min_train) / (max_train - min_train))
    rescaled_reconstruction_loss = np.clip(rescaled_reconstruction_loss, 0, 1)
    probabilities = np.zeros((len(rescaled_reconstruction_loss),))
    for i, l in enumerate(rescaled_reconstruction_loss):
        probabilities[i] = _from_standardized_loss_to_proba(l, rescaled_thresh)
    return probabilities

def LSTM_AE_keras(x,X_train,hyperparameters):
    # Dimensions are write before each line break. The batch_size dims is ignored for visibility purpose

    nb_layers=hyperparameters["nb_layers"]
    init_units=hyperparameters["init_filters"]
    dropout=hyperparameters["dropout_rate"]
    a='relu'

    # input: timestep,channels
    x=layers.LSTM(init_units, input_shape=(X_train.shape[1], X_train.shape[2]),
                  activation=a,dropout=dropout,return_sequences=nb_layers>1)(x)# output: timesteps,last_hidden_features
    for i in range(nb_layers-1):
        x=layers.LSTM(init_units,activation=a,dropout=dropout,return_sequences=nb_layers - 2 != 0)(x) #output: hidden_features (due to return_sequences==False)

    features_extractor=x

    x=layers.RepeatVector(X_train.shape[1])(x) #reshape technics to produce 2D array: timesteps, features
    for i in range(nb_layers):
        x=layers.LSTM(init_units,activation=a, dropout=dropout,return_sequences=True)(x)  # output: timestep,features
    #x=  # Next value channels, input: channels,hidden_state ; output: timesteps,channels
    x=layers.TimeDistributed(layers.Dense(X_train.shape[2]))(x)
    return x,features_extractor




def dense_AE_keras(x,X_train,hyperparameters):
    nb_layers=hyperparameters["nb_layers"]
    init_units=hyperparameters["init_filters"]
    dropout=hyperparameters["dropout_rate"]
    a='relu'
    reg=regularizers.L2(hyperparameters["l2_reg"])


    # input: timestep,channels
    x=layers.Dense(init_units, activation=a,kernel_regularizer=reg)(x)
    for i in range(nb_layers-1):
        x=layers.Dropout(dropout)(x)
        x=layers.Dense(init_units,activation=a,kernel_regularizer=reg)(x)

    features_extractor=x

    for i in range(nb_layers):
        x=layers.Dropout(dropout)(x)
        x=layers.Dense(init_units,activation=a,kernel_regularizer=reg)(x)  # output: timestep,features
    x = layers.Dense(1)(x)
    return x,features_extractor


def conv_AE_keras(x, X_train, hyperparameters):
    NB_LAYERS=hyperparameters["nb_layers"]
    f=hyperparameters["init_filters"]
    k=hyperparameters["K"]
    DROPOUT_RATE=hyperparameters["dropout_rate"]
    K_reduc=hyperparameters["K_reduc"] #substractive
    reg=regularizers.L2(hyperparameters["l2_reg"])

    a="relu"

    for b in range(NB_LAYERS - 1):
        x = layers.Conv1D(
            filters=f, kernel_size=k, padding="same", strides=2, activation=a, kernel_regularizer=reg
        )(x)
        x = layers.Dropout(rate=DROPOUT_RATE)(x)
        f /= 2
        k -= K_reduc
    x = layers.Conv1D(filters=f, kernel_size=k, padding="same", strides=2, activation=a, kernel_regularizer=reg)(x)

    features_tensor = x

    for b in range(NB_LAYERS - 1):
        x = layers.Conv1DTranspose(
            filters=f, kernel_size=k, padding="same", strides=2, activation=a, kernel_regularizer=reg
        )(x)
        x = layers.Dropout(rate=DROPOUT_RATE)(x)
        f *= 2
        k += K_reduc
    x = layers.Conv1DTranspose(
        filters=f, kernel_size=k, padding="same", strides=2, activation=a, kernel_regularizer=reg
    )(x)
    x = layers.Conv1DTranspose(filters=1, kernel_size=k, padding="same", kernel_regularizer=reg)(x)
    return x,features_tensor


class AE:
    def __init__(self, deeplearning_techno, user_defined_hyperparameters):
        self.deeplearning_techno=deeplearning_techno

        # hyperparameters_to_use is the combination of exhaustive default hyperparameters and some updated by the user
        if self.deeplearning_techno=="LSTM_AE":
            self.hp=default_LSTM_hyperparameters()
        elif self.deeplearning_techno=="DENSE_AE":
            self.hp=default_dense_hyperparameters()
        elif self.deeplearning_techno=="CONV_AE":
            self.hp=default_conv_hyperparameters()
        else:
            raise ValueError(f"Error in AE class, deeplearning_techno not expected in the constructor: {self.deeplearning_techno}")
        self.hp.update(user_defined_hyperparameters)

        # common important hyperparameter are set now
        self.LR = self.hp["lr"]
        self.BATCH_SIZE = self.hp["batch_size"]
        self.PERCENTILE = self.hp["percentile"]
        self.NB_EPOCH = self.hp["epochs"]

        # Optimizer settings are common to all autoencoders technologies (LSTM, CONV, DENS)
        self.PATIENCE_RATE=0.05 #commonly used rule of thumb.
        self.input_tensor=None
        self.output_tensor=None
        self.features_tensor=None
        self.model=None
        self.threshold=None

    def _from_2Darray_to_3D_array(self, x):
        return np.reshape(x,(x.shape[0], x.shape[1], 1))
    def _from_3Darray_to_2Darray(self, x):
        return np.reshape(x,(x.shape[0],x.shape[1]))

    def fit(self,x_train_frames):
        x_train_frames=self._from_2Darray_to_3D_array(x_train_frames)

        # Designing of the model according self.deeplearning_techno and hyperparameters
        x = layers.Input(shape=(x_train_frames.shape[1], x_train_frames.shape[2]))
        self.input_tensor = x
        if self.deeplearning_techno=="LSTM_AE": #hyperparameters_to_use is the combination of exhaustive default hyperparameters and some updated by the user
            hyperparameters_to_use=default_LSTM_hyperparameters()
            hyperparameters_to_use.update(self.hp)
            x,self.features_tensor=LSTM_AE_keras(x, x_train_frames, hyperparameters_to_use)
        elif self.deeplearning_techno=="DENSE_AE":
            hyperparameters_to_use=default_dense_hyperparameters()
            hyperparameters_to_use.update(self.hp)
            x,self.features_tensor=dense_AE_keras(x,x_train_frames, hyperparameters_to_use)
        elif self.deeplearning_techno=="CONV_AE":
            hyperparameters_to_use=default_conv_hyperparameters()
            hyperparameters_to_use.update(self.hp)
            x,self.features_tensor=conv_AE_keras(x,x_train_frames, hyperparameters_to_use)
        else:
            raise ValueError(f"Error in AE class, deeplearning_technology not understood {self.deeplearning_techno}")
        self.output_tensor = x

        # Building of the model in memory
        patience=int(self.NB_EPOCH*self.PATIENCE_RATE)
        self.model = keras.Model(self.input_tensor, self.output_tensor)
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.LR,clipvalue=4.), loss="mse")

        # traning
        history = self.model.fit(
            x_train_frames,
            x_train_frames,
            epochs=self.NB_EPOCH,
            batch_size=self.BATCH_SIZE,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="loss", patience=patience, mode="min")
            ],
            verbose=VERBOSITY
        )

        # compute anomal threshold
        x_train_pred = self.model.predict(x_train_frames,verbose=VERBOSITY)
        train_reconstruction_loss = np.mean(np.abs(x_train_pred - x_train_frames), axis=1)
        self.threshold = np.quantile(train_reconstruction_loss,self.PERCENTILE)+1e-7 #1e-7 increase the airthmetic robustness
        self.max=max(np.max(train_reconstruction_loss),self.threshold*2) # I propose to handle it like this
        self.min=np.min(train_reconstruction_loss)




    def predict(self,x_frames):
        assert(self.model is not None)
        assert(self.threshold is not None)
        x_frames3D = self._from_2Darray_to_3D_array(x_frames)
        x_frames_pred3D = self.model.predict(x_frames3D,verbose=VERBOSITY)
        x_frames_pred=self._from_3Darray_to_2Darray(x_frames_pred3D)

        reconstruction_loss = np.mean(np.abs(x_frames_pred - x_frames), axis=1)

        probabilities=_from_loss_to_proba(reconstruction_loss,self.threshold,self.min,self.max)
        return probabilities

    def features_extractor(self,x_frames):
        assert(self.model is not None)
        assert(self.threshold is not None)
        fe=keras.Model(self.input_tensor, self.features_tensor)
        features=fe.predict(x_frames,verbose=VERBOSITY)

        if len(features.shape)==3: #the nominal case
            newshape=(features.shape[0],features.shape[1]*features.shape[2])
        elif len(features.shape)==2:
            newshape=(features.shape[0],features.shape[1])
        else:
            raise ValueError("Error in features_extractor(). Unexpected features vector shape")

        features=np.reshape(features,newshape=newshape)

        # handle nan values to be more robust
        features[np.isnan(features)]=0.

        return features

    def __del__(self):
        try:
            clear_session()
        except:
            pass
