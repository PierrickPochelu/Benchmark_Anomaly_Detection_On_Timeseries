# When a new algorithm is implemented do not forget to update this file
from anomaly_detectors.realtime_algos import CADKNN,RE,Numenta,OSE,stump
from anomaly_detectors.offline_algos import oneclass_svm,isolation_forest,elliptic_envelope,lof,knn
from anomaly_detectors.offline_algos import conv_AE_reconstruction,LSTM_AE_reconstruction,dense_AE_reconstruction,conv2D_AE_reconstruction
#from anomaly_detectors.offline_algos import 2Dconv_AE_reconstruction
from anomaly_detectors.offline_algos import alwaystrue
from anomaly_detectors.FE import ROCKET, DATAAUG, FRAMED, LOGMELSPECTR, SPECTRO, NONFRAMED, FESOUND, MFCCS, MFCCS2D
from anomaly_detectors.FE import conv_AE_FE,LSTM_AE_FE,dense_AE_FE,conv2D_AE_FE


detector_strat_map={"RE":(RE,True),
           "CADKNN":(CADKNN,True),
           "ARTIME":(Numenta,True),
           "OSE":(OSE,True),
            "STUMP":(stump,True),
            "ALWAYSTRUE":(alwaystrue,True),

           "ELLIPTIC": (elliptic_envelope,False),
           "ONESVM": (oneclass_svm,False),
           "IFOREST": (isolation_forest,False),
            "LOF":(lof,False),
            "KNN":(knn,False),

            "LSTM_AE":(LSTM_AE_reconstruction,False),
            "CONV_AE": (conv_AE_reconstruction, False),
            "CONV2D_AE": (conv2D_AE_reconstruction, False),
            "DENSE_AE": (dense_AE_reconstruction, False)
           }

feature_extractor_strat_map={
    "CONVAE":conv_AE_FE,
    "LSTMAE": LSTM_AE_FE,
    "DENSEAE":dense_AE_FE,
    "ROCKET":ROCKET,
    "DATAAUG":DATAAUG,
    "FRAMED":FRAMED,
    "NONFRAMED":NONFRAMED,
    "LOGMELSPECTR":LOGMELSPECTR,
    # 1D SOUND
    "FESOUND":FESOUND,
    "SPECTRO":SPECTRO,
    "MFCCS":MFCCS,
    #2D SOUND
    "2DCONVAE": conv2D_AE_FE,
    "MFCCS2D": MFCCS2D
}