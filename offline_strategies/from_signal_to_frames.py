import numpy as np


def create_sequences(values, frame_size):
    output = []
    for i in range(len(values) - frame_size + 1):
        output.append(values[i : (i + frame_size)])
    stacked_frames=np.stack(output)
    stacked_frames=stacked_frames.reshape((stacked_frames.shape[0],stacked_frames.shape[1],1))
    return stacked_frames

def frames(x,y,frame_size, nb_frames):
    x=create_sequences(x,frame_size).squeeze()
    return x,y

def data_augment_frames(x, y, frame_size, nb_frames):
    from tsaug import TimeWarp, Crop, Quantize, Drift, AddNoise

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
    X_aug, Y_aug = my_augmenter.augment(x, y)
    X_aug=X_aug.reshape((X_aug.shape[0],X_aug.shape[1]))
    return X_aug,Y_aug

def from_signal_to_frame(train_dataset, test_dataset, frame_size, frame_strategy_name):
    x_train = train_dataset["x"]
    y_train = train_dataset["y"]
    nb_train_frames = len(x_train) - frame_size+1
    x_test = test_dataset["x"]
    y_test = test_dataset["y"]
    nb_test_frames = len(x_test) - frame_size+1

    #frame_construction_function=S[frame_strategy]
    if frame_strategy_name== "IDENTITY":
        x_frames_train, _ = frames(x_train, y_train, frame_size,nb_train_frames)
        x_frames_test, _ = frames(x_test, y_test, frame_size, nb_test_frames)
    elif frame_strategy_name== "DATAAUG":
         x_frames_train, train_dataset["y"] = data_augment_frames(x_train, y_train, frame_size, nb_train_frames*10)
         x_frames_test, test_dataset["y"]  = frames(x_test, y_test, frame_size, nb_test_frames)
         # we ensure ourself those datastruc are not used anymore
         del train_dataset["x"]
         del test_dataset["x"]
    elif frame_strategy_name== "AE":
        x_frames_train, _ = frames(x_train, y_train, frame_size,nb_train_frames)
        x_frames_test, _ = frames(x_test, y_test, frame_size, nb_test_frames)

        from offline_strategies.AEC import default_hyperparameters,AE
        hp=default_hyperparameters()
        hp["nb_layers"]=2
        model=AE(hp)
        model.fit(x_frames_train)
        x_frames_train=model.features_extractor(x_frames_train).squeeze()
        x_frames_test=model.features_extractor(x_frames_test).squeeze()

        # re-centering
        mu=np.mean(x_frames_train)
        std=np.std(x_frames_train)
        x_frames_train=(x_frames_train-mu)/std
        x_frames_test=(x_frames_test-mu)/std
    elif frame_strategy_name== "ROCKET":
        # "SIMPLE" STRATEGY on PREPROCESSED SIGNALS
        x_frames_train, _ = frames(train_dataset["x"], train_dataset["y"], frame_size,nb_train_frames)
        x_frames_test, _ = frames(test_dataset["x"], test_dataset["y"], frame_size, nb_test_frames)

        def rocket_pre_shape(x):
            return x.reshape((1, len(x)))

        def rocket_post_shape(x):
            x=x.T
            return x.squeeze()

        hyperparameters = {"n_kernels": 128, "kernel_sizes": [7,9,11]}
        # https://pyts.readthedocs.io/en/stable/modules/transformation.html
        from pyts.transformation import ROCKET
        def ROCKET_transform(hyperparameters: dict, frame: np.ndarray) -> np.ndarray:
            model = ROCKET(**hyperparameters)
            pre_frame=rocket_pre_shape(frame)
            features_extracted_frame=model.fit_transform(pre_frame)
            post_frame=rocket_post_shape(features_extracted_frame)
            return post_frame

        rocket_x_frames_train=np.array([ROCKET_transform(hyperparameters,frame) for frame in x_frames_train])
        rocket_x_frames_test=np.array([ROCKET_transform(hyperparameters,frame) for frame in x_frames_test])
        x_frames_train, x_frames_test = rocket_x_frames_train, rocket_x_frames_test
    else:
        raise ValueError(f"ERROR in from_signal_to_frame() strategy unknown {frame_strategy_name} ")

    return x_frames_train, x_frames_test

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

