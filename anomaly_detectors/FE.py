import numpy as np


def create_sequences(values, frame_size):
    output = []
    nb_frames=get_nb_frames(values,frame_size)
    for i in range(nb_frames):
        output.append(values[i : (i + frame_size)])
    stacked_frames=np.stack(output)
    stacked_frames=stacked_frames.reshape((stacked_frames.shape[0],stacked_frames.shape[1],1))
    return stacked_frames

def frames(x,frame_size):
    #x=create_sequences(x,frame_size).squeeze()
    return x

def get_nb_frames(x,frame_size):
    return len(x) - frame_size + 1
def _data_augment_frames(x, y, frame_size, nb_frames):
    from tsaug import TimeWarp, Crop, Quantize, Drift, AddNoise

    """
    lahf=0.1 # low amplitude high frequency
    hilf=0.05 # high amplitude low frequency
    my_augmenter = (Crop(size=frame_size)*nb_frames +  # random crop subsequences
                    TimeWarp(max_speed_ratio=2) @ lahf +
                    TimeWarp(max_speed_ratio=4) @ hilf +
                    Quantize(n_levels=[16]) @ lahf +  # random quantization
                    Quantize(n_levels=[8]) @ hilf +  # random quantization
                    Drift(max_drift=(0, 0.05),n_drift_points=[1,2],kind="multiplicative") @ lahf +
                    Drift(max_drift=(0, 0.1), n_drift_points=[2,4], kind="multiplicative") @ hilf +
                    AddNoise(scale=(0., 0.05)) @ lahf +
                    AddNoise(scale=(0., 0.1)) @ hilf
                    )
    """
    lahf=0.2
    my_augmenter = (Crop(size=frame_size)*nb_frames +  # random crop subsequences
                    TimeWarp(max_speed_ratio=2) @ lahf +
                    Quantize(n_levels=[16]) @ lahf +  # random quantization
                    Drift(max_drift=(0, 0.05),n_drift_points=[1,2],kind="multiplicative") @ lahf +
                    AddNoise(scale=(0., 0.05)) @ lahf
                    )
    X_aug, Y_aug = my_augmenter.augment(x, y)
    X_aug=X_aug.reshape((X_aug.shape[0],X_aug.shape[1]))
    return X_aug,Y_aug


def NONFRAMED(train_dataset,test_dataset,frame_size=None,hyperparameters=0):
    return train_dataset,test_dataset



def FRAMED(train_dataset, test_dataset, frame_size, hyperparameters={}):
    x_frames_train = frames(train_dataset["x"], frame_size)
    x_frames_test = frames(test_dataset["x"], frame_size)
    train_dataset["x"]=x_frames_train
    test_dataset["x"]=x_frames_test
    return train_dataset, test_dataset
def ROCKET(train_dataset,test_dataset,frame_size,hyperparameters = {"n_kernels": 128, "kernel_sizes": [7, 9, 11]}):
    # https://pyts.readthedocs.io/en/stable/modules/transformation.html
    from pyts.transformation import ROCKET as pyts_ROCKET

    def rocket_pre_shape(x):
        return x.reshape((1, len(x)))

    def rocket_post_shape(x):
        x = x.T
        return x.squeeze()

    def ROCKET_transform(hyperparameters: dict, frame: np.ndarray) -> np.ndarray:
        model = pyts_ROCKET(**hyperparameters)
        pre_frame = rocket_pre_shape(frame)
        features_extracted_frame = model.fit_transform(pre_frame)
        post_frame = rocket_post_shape(features_extracted_frame)
        return post_frame

    # "SIMPLE" STRATEGY on PREPROCESSED SIGNALS
    #x_frames_train = frames(train_dataset["x"], frame_size)
    #x_frames_test = frames(test_dataset["x"], frame_size)
    x_frames_train=train_dataset["x"]
    x_frames_test=test_dataset["x"]

    rocket_x_frames_train = np.array([ROCKET_transform(hyperparameters, frame) for frame in x_frames_train])
    rocket_x_frames_test = np.array([ROCKET_transform(hyperparameters, frame) for frame in x_frames_test])

    train_dataset["x"]=rocket_x_frames_train
    test_dataset["x"]=rocket_x_frames_test
    return train_dataset, test_dataset
def _AE_FE(deeplearning_techno, train_dataset,test_dataset,frame_size,hyperparameters={}):

    x_frames_train, x_frames_test = train_dataset["x"],test_dataset["x"]#FRAMED(train_dataset, test_dataset, frame_size)

    from anomaly_detectors.autoencoder import AE
    model = AE(deeplearning_techno,hyperparameters)
    model.fit(x_frames_train)
    x_frames_train = model.features_extractor(x_frames_train).squeeze()
    x_frames_test = model.features_extractor(x_frames_test).squeeze()

    # re-centering
    mu = np.mean(x_frames_train)
    std = np.std(x_frames_train)
    x_frames_train = (x_frames_train - mu) / (std+1e-7)
    x_frames_test = (x_frames_test - mu) / (std+1e-7)

    # replace train and test data
    train_dataset["x"]=x_frames_train
    test_dataset["x"]=x_frames_test
    return train_dataset,test_dataset

def conv_AE_FE(train_dataset,test_dataset,frame_size,hyperparameters={"nb_layers":2}):
    return _AE_FE("CONV_AE", train_dataset,test_dataset,frame_size,hyperparameters)
def LSTM_AE_FE(train_dataset,test_dataset,frame_size,hyperparameters={}):
    return _AE_FE("LSTM_AE", train_dataset,test_dataset,frame_size,hyperparameters)
def dense_AE_FE(train_dataset,test_dataset,frame_size,hyperparameters={}):
    return _AE_FE("DENSE_AE", train_dataset,test_dataset,frame_size,hyperparameters)

def DATAAUG (train_dataset,test_dataset,frame_size,hyperparameters={"multiplier":10}):
    x_train=train_dataset["x"]
    y_train=train_dataset["y"]
    x_test=test_dataset["x"]
    y_test=test_dataset["y"]
    nb_wanted_frames=get_nb_frames(x_train,frame_size)*hyperparameters["multiplier"]
    train_dataset["x"], train_dataset["y"] = _data_augment_frames(x_train, y_train, frame_size, nb_wanted_frames)
    test_dataset["x"] = frames(x_test, frame_size)
    return train_dataset, test_dataset

def _from_signal_to_logmelspectrogram(signal:np.ndarray, hyperparameters:dict):
    import librosa
    n_mels = hyperparameters["n_mels"]
    hop_length = hyperparameters["hop_length"]
    sampling_rate = hyperparameters["sampling_rate"]
    n_fft = 1024
    power = 2.
    # generate melspectrogram using librosa
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sampling_rate,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,  # 1hop=1column in the spectrogram
                                                     n_mels=n_mels,
                                                     power=power)

    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram.T
def LOGMELSPECTR(train_dataset,test_dataset,frame_size,wanted_hyperparameters={}): #"sampling_rate": 16000, "n_mels": 64, "hop_length": 512}
    hyperparameters={"sampling_rate": 16000, "n_mels": 64, "hop_length": 512}
    wanted_hyperparameters.update(wanted_hyperparameters)
    new_train_frames=[]
    for i,xi in enumerate(train_dataset["x"]):
        new_train_frames.append(_from_signal_to_logmelspectrogram(xi,hyperparameters))
    train_dataset["x"] = np.array(new_train_frames).astype(np.float)

    new_test_frames = []
    for i,xi in enumerate(test_dataset["x"]):
        new_test_frames.append(_from_signal_to_logmelspectrogram(xi,hyperparameters))
    test_dataset["x"] = np.array(new_test_frames).astype(np.float)

    # reshape
    nbtrainframes,t,s=train_dataset["x"].shape
    train_dataset["x"]=train_dataset["x"].reshape(len(train_dataset["x"]),t*s)
    test_dataset["x"]=test_dataset["x"].reshape(len(test_dataset["x"]),t*s)

    # renorm
    mu,std=np.mean(train_dataset["x"]),np.std(train_dataset["x"])
    train_dataset["x"]=(train_dataset["x"]-mu)/(std+1e-7)
    test_dataset["x"] =(test_dataset["x"]-mu) / (std + 1e-7)

    return train_dataset, test_dataset

def SPECTR(train_dataset,test_dataset,frame_size,hyperparameters={"sampling_rate": 16000, "n_mels": 64, "hop_length": 512}):
    raise NotImplementedError("no spectrogram")
    train_dataset["x"] = _from_signal_to_logmelspectrogram(train_dataset["x"],hyperparameters)
    test_dataset["x"] = _from_signal_to_logmelspectrogram(test_dataset["x"],hyperparameters)
    train_dataset, test_dataset=FRAMED(train_dataset,test_dataset,frame_size,hyperparameters)
    return train_dataset, test_dataset



if __name__=="__main__":

    import numpy as np
    x=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.2,0.8,0.9])
    y=np.array([0,0,0,0,0,0,0,1,0,0])
    train_dataset={"x":x[:5],"y":y[:5]}
    test_dataset={"x":x[5:],"y":y[5:]}
    xi,yi=from_signal_to_frame(train_dataset, test_dataset, 3, "SIMPLE")

    from pyts.transformation import ROCKET
    model=ROCKET(n_kernels=8192,kernel_sizes=[3])
    raw_train_x=train_dataset["x"]

    def rocket_pre_shape(x):
        return x.reshape((1,len(x)))
    def rocket_post_shape(x):
        return x.T.reshape((len(x),))
    raw_train_x=rocket_pre_shape(raw_train_x)
    model.fit(raw_train_x)
    rocket_features_extracted_train=model.transform(rocket_pre_shape(train_dataset["x"]))
    rocket_features_extracted_test=model.transform(rocket_pre_shape(test_dataset["x"]))

    train_dataset["x"]=rocket_post_shape(rocket_features_extracted_train)
    train_dataset["x"]=rocket_post_shape(rocket_features_extracted_test)

