from experiments import LAUNCH_EXPERIMENTS_AT_SCALE
from strat_map import detector_strat_map,feature_extractor_strat_map
if __name__=="__main__":
    from extract_data import extract_datasets
    datasets=extract_datasets("./data/NAB/")
    feature_extractor="IDENTITY"

    for detector in detector_strat_map.keys(): # for each known detection strategy
        print("Compute the mosaic with the strategy: ", detector)
        results=LAUNCH_EXPERIMENTS_AT_SCALE(feature_extractor,detector,datasets)
        print(feature_extractor, detector, results)
