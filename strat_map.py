# When a new algorithm is implemented do not forget to update this file
from realtime_strategies.algos import CADKNN,RE,Numenta,OSE
from offline_strategies.algos import oneclass_svm,isolation_forest,elliptic_envelope,AE_reconstruction
from offline_strategies.from_signal_to_frames import AE_features_extractor, ROCKET, DATAAUG, IDENTITY

detector_strat_map={"RE":(RE,True),
           "CADKNN":(CADKNN,True),
           "ARTIME":(Numenta,True),
           "OSE":(OSE,True),
           "AE": (AE_reconstruction,False),
           "ELLIPTIC": (elliptic_envelope,False),
           "ONESVM": (oneclass_svm,False),
           "IFOREST": (isolation_forest,False)
           }

feature_extractor_strat_map={
    "AE":AE_features_extractor,
    "ROCKET":ROCKET,
    "DATAAUG":DATAAUG,
    "IDENTITY":IDENTITY
}