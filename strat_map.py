# When a new algorithm is implemented do not forget to update this file
from realtime_strategies.algos import CADKNN,RE,Numenta,OSE,stump
from offline_strategies.algos import oneclass_svm,isolation_forest,elliptic_envelope
from offline_strategies.algos import conv_AE_reconstruction,LSTM_AE_reconstruction,dense_AE_reconstruction
from offline_strategies.FE import ROCKET, DATAAUG, FRAMED, SPECTR, LOGMELSPECTR, NONFRAMED
from offline_strategies.FE import conv_AE_FE,LSTM_AE_FE,dense_AE_FE

detector_strat_map={"RE":(RE,True),
           "CADKNN":(CADKNN,True),
           "ARTIME":(Numenta,True),
           "OSE":(OSE,True),
            "STUMP":(stump,True),

           "ELLIPTIC": (elliptic_envelope,False),
           "ONESVM": (oneclass_svm,False),
           "IFOREST": (isolation_forest,False),

            "LSTM_AE":(LSTM_AE_reconstruction,False),
            "CONV_AE": (conv_AE_reconstruction, False),
            "DENSE_AE": (dense_AE_reconstruction, False)
           }

feature_extractor_strat_map={
    "CONV_AE":conv_AE_FE,
    "LSTM_AE": LSTM_AE_FE,
    "DENSE_AE":dense_AE_FE,
    "ROCKET":ROCKET,
    "DATAAUG":DATAAUG,
    "FRAMED":FRAMED,
    "NONFRAMED":NONFRAMED,
    "LOGMELSPECTR":LOGMELSPECTR,
    "SPECTR":SPECTR
}