from experiments import LAUNCH_EXPERIMENTS_AT_SCALE
if __name__=="__main__":
    from extract_data import extract_datasets
    datasets=extract_datasets("./data/NAB/")
    feature_extractor="IDENTITY"
    detector="OSE"
    LAUNCH_EXPERIMENTS_AT_SCALE(feature_extractor,detector,datasets)