import numpy as np
from anomaly_detectors.offline_algos import compute_hyperparameters

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
def ROCKET(train_dataset,test_dataset,frame_size,hyperparameters = {}):
    # https://pyts.readthedocs.io/en/stable/modules/transformation.html
    hyperparameters=compute_hyperparameters(hyperparameters,{"n_kernels": 128, "kernel_sizes": [7, 9, 11]})
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

def conv_AE_FE(train_dataset,test_dataset,frame_size,hyperparameters={}):
    return _AE_FE("CONV_AE", train_dataset,test_dataset,frame_size,hyperparameters)
def LSTM_AE_FE(train_dataset,test_dataset,frame_size,hyperparameters={}):
    return _AE_FE("LSTM_AE", train_dataset,test_dataset,frame_size,hyperparameters)
def dense_AE_FE(train_dataset,test_dataset,frame_size,hyperparameters={}):
    return _AE_FE("DENSE_AE", train_dataset,test_dataset,frame_size,hyperparameters)
def conv2D_AE_FE(train_dataset,test_dataset,frame_size,hyperparameters={}):
    return _AE_FE("2DCONVAE", train_dataset,test_dataset,frame_size,hyperparameters)

def DATAAUG (train_dataset,test_dataset,frame_size,wanted_hyperparameters={}):
    hyperparameters=compute_hyperparameters(wanted_hyperparameters,{"multiplier":10})
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

import scipy


def descriptiv(x):
    _get=lambda v: v[0] if isinstance(v,list) else v

    mean=np.mean(x)
    std=np.std(x)
    q99=np.quantile(x,0.99)
    q90=np.quantile(x,0.9)
    q75=np.quantile(x,0.75)
    q50=np.quantile(x,0.5)
    q25=np.quantile(x,0.25)
    q10=np.quantile(x,0.1)
    q01=np.quantile(x,0.01)

    skew = scipy.stats.skew(x)
    skew=_get(skew)

    kurt = scipy.stats.kurtosis(x) #distance between gaussian and real distribution tai
    kurt=_get(kurt)

    mode = scipy.stats.mode(x)[0][0]
    mode=_get(mode)

    iqr = scipy.stats.iqr(x)#distance between 25th and 75th
    iqr=_get(iqr)
    return np.array([mean,std,skew,kurt,mode,iqr,
                     q99,q90,q75,q50,q25,q10,q01])
def spec(x):
    import librosa
    spec_centroid = librosa.feature.spectral_centroid(x)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(x)[0]
    spectral_contrast = librosa.feature.spectral_contrast(x)[0]
    spectral_flatness = librosa.feature.spectral_flatness(x)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(x)[0]
    return np.concatenate([spec_centroid,spectral_bandwidth,spectral_contrast,spectral_flatness,spectral_rolloff])

def fesound(x):
    import librosa
    sr=16000
    #hop_length = int(len(x)*1.1)

    #https://maelfabien.github.io/machinelearning/Speech9/#2-energy
    #https://medium.com/@alexandro.ramr777/audio-files-to-dataset-by-feature-extraction-with-librosa-d87adafe5b64


    # cent = librosa.feature.spectral_centroid(y=x, sr=sr)
    # cent=cent[0,0]
    # zero_crossing=librosa.feature.zero_crossing_rate(x + 0.0001)
    # zero_crossing=zero_crossing[0,0]
    # chromagram=librosa.feature.chroma_stft(y=x, sr=sr, hop_length=hop_length)
    # chromagram=np.mean(chromagram)
    mel_spectrogram = librosa.feature.melspectrogram(y=x, sr=sr,
                                                     n_fft=1024,
                                                     hop_length=512,
                                                     n_mels=64,
                                                     power=2.)
    S_dB = librosa.power_to_db(mel_spectrogram, ref=np.max).flatten()

    mfccs = librosa.feature.mfcc(x, sr=sr)#mel
    mfccs=mfccs.flatten()

    #poly_features = librosa.feature.poly_features(x)#generate boring warnings
    #descpoly = descriptiv(poly_features)
    #mel=librosa.feature.

    # spectr=spec(x)
    vector=[mfccs,S_dB]
    #vector=[desc,spectr,cent,zero_crossing,rmse]
    #vector=[desc, mfccs, cent, zero_crossing, rmse, tempogram, spectr]
    #vector=[mfccs]

    for i,v in enumerate(vector):
        v=np.array(v) # force conversion
        if len(v.shape)>2:
            raise ValueError("more than 1D it makes the concatenation impossible")
        elif len(v.shape)==1:
            pass #ok
        elif len(v.shape)==0:
            vector[i]=v.reshape((1,))
    vec = np.concatenate(vector)
    return vec


def FESOUND(train_dataset,test_dataset,frame_size,hyperparameters={"sampling_rate": 16000, "hop_length": 512}):
    fex=[]
    for xi in train_dataset["x"]:
        fex.append( fesound(xi) )
    train_dataset["x"]=np.array(fex)

    fex=[]
    for xi in test_dataset["x"]:
        fex.append( fesound(xi) )
    test_dataset["x"]=np.array(fex)

    mu=np.mean(train_dataset["x"],axis=0)
    std=np.std(train_dataset["x"],axis=0)
    train_dataset["x"]=(train_dataset["x"]-mu)/(std+1e-7)
    test_dataset["x"]=(test_dataset["x"]-mu)/(std+1e-7)
    return train_dataset, test_dataset



def SPECTRO(train_dataset,test_dataset,frame_size,hyperparameters={}):
    sr=16000
    # https://dcase.community/challenge2021/task-unsupervised-detection-of-anomalous-sounds-results
    # https://dcase.community/documents/challenge2021/technical_reports/DCASE2021_Morita_59_t2.pdf
    # https://dcase.community/documents/challenge2021/technical_reports/DCASE2021_Lopez_6_t2.pdf
    # https://dcase.community/documents/challenge2021/technical_reports/DCASE2021_Wilkinghoff_31_t2.pdf


    default_hp={"sr": 16000, "n_fft":1024, "hop_length":512, "n_mels":128}
    hyperparameters=compute_hyperparameters(hyperparameters,default_hp=default_hp)

    if frame_size<=hyperparameters["hop_length"]:
        raise ValueError("frame_size should be superior of hop_length")
    import librosa

    def one_frame(signal,hp):
        mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=hp["sr"],
                                                         n_fft=hp["n_fft"],
                                                         hop_length=hp["hop_length"],
                                                         n_mels=hp["n_mels"],
                                                         power=2.)
        S_dB = librosa.power_to_db(mel_spectrogram, ref=np.max)





        return S_dB.flatten()

    # convert frames into images
    new_db=[]
    for i in range(len(train_dataset["x"])):
        fe=one_frame(train_dataset["x"][i],hyperparameters)
        new_db.append(fe)
    train_dataset["x"]=np.array(new_db)

    new_db=[]
    for i in range(len(test_dataset["x"])):
        fe=one_frame(test_dataset["x"][i],hyperparameters)
        new_db.append(fe)
    test_dataset["x"]=np.array(new_db)

    """
    import matplotlib.pyplot as plt
    import librosa.display  # mandatory to use librosa.display
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sampling_rate, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    label_name = "normal" if label == 0 else "anomaly"
    img_name = "./jpeg/" + str(db_id) + "/" + str(frame_id) + "_" + str(label_name) + ".jpg"
    plt.savefig(img_name)
    """

    M=np.max(train_dataset["x"])
    m=np.min(train_dataset["x"])
    train_dataset["x"]=(train_dataset["x"]-m)/(M-m)
    test_dataset["x"]=(test_dataset["x"]-m)/(M-m)

    return train_dataset, test_dataset

def MFCCS(train_dataset,test_dataset,frame_size,hyperparameters={}):
    import librosa
    fex=[]
    for xi in train_dataset["x"]:
        mfccs = librosa.feature.mfcc(xi, sr=16000)  # mel
        mfccs = mfccs.flatten()
        fex.append( mfccs )
    train_dataset["x"]=np.array(fex)

    fex=[]
    for xi in test_dataset["x"]:
        mfccs = librosa.feature.mfcc(xi, sr=16000)  # mel
        mfccs = mfccs.flatten()
        fex.append( mfccs )
    test_dataset["x"]=np.array(fex)

    mu=np.mean(train_dataset["x"],axis=0)
    std=np.std(train_dataset["x"],axis=0)
    train_dataset["x"]=(train_dataset["x"]-mu)/(std+1e-7)
    test_dataset["x"]=(test_dataset["x"]-mu)/(std+1e-7)

    return train_dataset,test_dataset

def MFCCS2D(train_dataset,test_dataset,frame_size,hyperparameters={}):
    import librosa
    hop_length=512
    n_mfcc=32
    fex=[]
    for xi in train_dataset["x"]:
        mfccs = librosa.feature.mfcc(xi, sr=16000,n_mfcc=n_mfcc,hop_length=hop_length)  # mel
        mfccs=mfccs[:,0:len(xi)//hop_length]
        fex.append( mfccs )
    train_dataset["x"]=np.array(fex)

    fex=[]
    for xi in test_dataset["x"]:
        mfccs = librosa.feature.mfcc(xi, sr=16000,n_mfcc=n_mfcc,hop_length=hop_length)  # mel
        mfccs=mfccs[:,0:len(xi)//hop_length]
        fex.append( mfccs )
    test_dataset["x"]=np.array(fex)

    mu=np.mean(train_dataset["x"])
    std=np.mean(np.std(train_dataset["x"],axis=(1,2)))
    train_dataset["x"]=(train_dataset["x"]-mu)/(std+1e-7)
    test_dataset["x"]=(test_dataset["x"]-mu)/(std+1e-7)

    return train_dataset,test_dataset