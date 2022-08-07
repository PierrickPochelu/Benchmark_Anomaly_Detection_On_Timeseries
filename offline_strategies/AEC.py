
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow import keras
from keras.backend import clear_session
from tensorflow.keras import layers
import numpy as np

def default_hyperparameters():
    return {"nb_layers":4, "K": 7,"K_reduc":0, "init_filters": 32,"dropout_rate":0.2, "lr":0.001,
            "batch_size":128,"l2_reg":0.001,"percentile":0.999}

def _from_loss_to_proba(rescaled_reconstruction_loss, rescaled_thresh):
    # check
    assert (1. >= rescaled_reconstruction_loss >= 0.)
    assert (1. >= rescaled_thresh >= 0.)


    if rescaled_reconstruction_loss > rescaled_thresh:
        reconstruction_proba = ((1. - 0.5) / (1 - rescaled_thresh)) * (
                    rescaled_reconstruction_loss - rescaled_thresh)+0.5
    elif rescaled_reconstruction_loss < rescaled_thresh:
        reconstruction_proba = ((0.5 - 0.) / (rescaled_thresh - 0.)) * (rescaled_reconstruction_loss - 0)
    else:
        reconstruction_proba = 0.5

    reconstruction_proba=np.clip(reconstruction_proba,0,1)
    return reconstruction_proba

class AE:
    def __init__(self, hyperparameters):
        self.h=hyperparameters


        self.NB_LAYERS = self.h["nb_layers"]
        self.K = self.h["K"]
        self.K_reduc=self.h["K_reduc"]
        self.INIT_FILTERS=self.h["init_filters"]
        self.DROPOUT_RATE=self.h["dropout_rate"]
        self.LR=self.h["lr"]
        self.BATCH_SIZE=self.h["batch_size"]
        self.L2_REG=self.h["l2_reg"]
        self.PERCENTILE=self.h["percentile"]
        self.NB_EPOCH=1000
        self.PATIENCE_RATE=0.1
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

        reg=keras.regularizers.l2(self.L2_REG)
        a="relu" #keras.layers.LeakyReLU(alpha=0.01)
        x = layers.Input(shape=(x_train_frames.shape[1], x_train_frames.shape[2]))
        self.input_tensor = x

        f = self.INIT_FILTERS
        k=self.K
        for b in range(self.NB_LAYERS - 1):
            x = layers.Conv1D(
                filters=f, kernel_size=k, padding="same", strides=2, activation=a, kernel_regularizer=reg
            )(x)
            #x=layers.AveragePooling1D(pool_size=2,padding="same")(x)
            x = layers.Dropout(rate=self.DROPOUT_RATE)(x)
            f /= 2
            k -= self.K_reduc
        x = layers.Conv1D(filters=f, kernel_size=k, padding="same", strides=2, activation=a, kernel_regularizer=reg)(x)
        #x = layers.AveragePooling1D(pool_size=2, padding="same")(x)
        self.features_tensor=x


        #f = self.INIT_FILTERS
        for b in range(self.NB_LAYERS - 1):
            x = layers.Conv1DTranspose(
                filters=f, kernel_size=k, padding="same", strides=2, activation=a, kernel_regularizer=reg
            )(x)
            x = layers.Dropout(rate=self.DROPOUT_RATE)(x)
            f *= 2
            k += self.K_reduc
        x = layers.Conv1DTranspose(
            filters=f, kernel_size=k, padding="same", strides=2, activation=a, kernel_regularizer=reg
        )(x)
        x = layers.Conv1DTranspose(filters=1, kernel_size=k, padding="same", kernel_regularizer=reg)(x)
        self.output_tensor = x

        self.model = keras.Model(self.input_tensor, self.output_tensor)
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.LR), loss="mse")
        #self.model.summary()

        patience=int(self.NB_EPOCH*self.PATIENCE_RATE)
        history = self.model.fit(
            x_train_frames,
            x_train_frames,
            epochs=self.NB_EPOCH,
            batch_size=self.BATCH_SIZE,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="loss", patience=patience, mode="min")
            ],
            verbose=0
        )

        # compute anomal threshold
        x_train_pred = self.model.predict(x_train_frames,verbose=0)
        train_reconstruction_loss = np.mean(np.abs(x_train_pred - x_train_frames), axis=1)
        self.threshold = np.quantile(train_reconstruction_loss,self.PERCENTILE)
        self.max=max(np.max(train_reconstruction_loss),self.threshold*2) # I propose to handle it like this
        self.min=np.min(train_reconstruction_loss)




    def predict(self,x_frames):
        assert(self.model is not None)
        assert(self.threshold is not None)
        x_frames3D = self._from_2Darray_to_3D_array(x_frames)
        x_frames_pred3D = self.model.predict(x_frames3D,verbose=0)
        x_frames_pred=self._from_3Darray_to_2Darray(x_frames_pred3D)

        reconstruction_loss = np.mean(np.abs(x_frames_pred - x_frames), axis=1)

        rescaled_thresh=(self.threshold-self.min)/(self.max-self.min)
        rescaled_reconstruction_loss=((reconstruction_loss-self.min)/(self.max-self.min))
        rescaled_reconstruction_loss=np.clip(rescaled_reconstruction_loss,0,1)

        probabilities=np.zeros((len(rescaled_reconstruction_loss),))
        for i,l in enumerate(rescaled_reconstruction_loss):
            probabilities[i]=_from_loss_to_proba(l, rescaled_thresh)
        return probabilities


    def features_extractor(self,x_frames):
        assert(self.model is not None)
        assert(self.threshold is not None)
        fe=keras.Model(self.input_tensor, self.features_tensor)
        features=fe.predict(x_frames,verbose=0)
        newshape=(features.shape[0],features.shape[1]*features.shape[2])
        features=np.reshape(features,newshape=newshape)
        return features

    def __del__(self):
        try:
            clear_session()
        except:
            pass

