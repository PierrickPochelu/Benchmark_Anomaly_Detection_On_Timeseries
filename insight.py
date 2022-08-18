from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import numpy as np
from typing import Optional
import os

def confusion_matrix_and_F1(y_pred,y_true):
    # Some timeseries does not contains abnormaly. We avoid division per zero by yielding accuracy instead of f1.
    # The returned score is always named "F1 score", for simplicity reasons.
    both_values = 0 in y_true and 1 in y_true
    accuracy = round(np.mean(y_pred == y_true),4)
    if both_values:
        cm=confusion_matrix(y_true, y_pred)
        vals = cm.ravel()

        tn, fp, fn, tp = vals
        f1 = f1_score(y_true, y_pred)
        stats={"tn":tn, "fp":fp, "fn":fn, "tp":tp,"acc":accuracy, "f1":round(f1,4)}
    else:
        ones=np.ones(len(y_true))
        twos=ones*2
        zeros=np.zeros(len(y_true))
        tp=np.sum((y_pred+y_true)==twos)
        tn=np.sum((y_pred+y_true)==zeros)
        fp=np.sum((2*y_pred+y_true)==twos)
        fn=np.sum((y_pred+2*y_true)==twos)
        stats={"tn":tn, "fp":fp, "fn":fn, "tp":tp,"acc":accuracy, "f1":-1}
    return stats

def ROCAUC(y_pred,y_true):
    both_values = 0 in y_true and 1 in y_true
    if not both_values:
        return {"rocauc":-1, "proposed_thresh":-1, "new_f1":-1}

    fpr, tpr, thresholds = roc_curve(y_true, y_pred,pos_label=1.,drop_intermediate=False)
    score=roc_auc_score(y_true, y_pred)
    thresh_proposed=thresholds[np.argmax((tpr-fpr))]

    infos=confusion_matrix_and_F1(y_pred>=thresh_proposed, y_true)

    return {"rocauc":score,
            "proposed_thresh":thresh_proposed,
            "new_f1":infos["f1"]}

def plot_curves(x_train:np.ndarray, x_test:np.ndarray, y_test:np.ndarray, y_pred:np.ndarray, frame_size:int,
                path:Optional[str]=None, txt:str= ""):

    x_test=x_test[frame_size-1:]
    if len(x_train.shape)==2:
        x=np.concatenate((
            np.concatenate((x_train[:,0],x_train[1,1:])),
            np.concatenate((x_test[:,0],x_test[1,1:])),))
    elif len(x_train.shape)==1:
        x=np.concatenate((x_train,x_test))
    else:
        raise ValueError("plot_curves() error in the number of received dimensions")

    non_prediction_area=frame_size-1

    y_train_offset=np.zeros(len(x_train))
    y_non_pred_area=np.zeros(len(x_train)+non_prediction_area)

    y=np.concatenate((y_train_offset,y_test))
    y_pred=np.concatenate((y_non_pred_area,y_pred))

    try:
        assert(len(x)==len(y)==len(y_pred))
    except AssertionError as e:
        print(len(x))
        print(len(y))
        print(len(y_pred))
    # plot curve
    plt.plot(np.arange(0,len(x)),x)

    # plot the anomalies and detections
    max_amplitude = np.max(x)
    min_amplitude = np.min(x)
    def color_fill(start_t,end_t,color):
        plt.fill_between(
            x=[start_t, end_t],
            y1=min_amplitude,
            y2=max_amplitude,
            color=color,
            alpha=0.5,
            linewidth=0,
            zorder=0,
        )
    color_fill(0, len(y_non_pred_area), "grey")


    for i in range(len(y_non_pred_area), len(y_pred)):
        if y[i]==y_pred[i]==0:
            # True Negative -> we keep it white
            pass
        elif  y[i]==y_pred[i]==1:
            # True positif
            color_fill(i-1, i, "green")
        elif y[i]==1 and y_pred[i]==0:
            # False negative
            color_fill(i-1,i,"red")
        elif  y[i]==0 and y_pred[i]==1:
            # False positive
            color_fill(i-1,i,"orange")
        else:
            raise ValueError(f"Unexpected case: y[i]={y[i]} y_pred[i]={y_pred[i]}")

    plt.title(txt)
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
    plt.close() #mandatory if you call multiple time to draw multiple figures

def mosaic(paths, fname, total_experimental_result, ncols=6, nrows=9):
    from matplotlib import pyplot as plt
    from PIL import Image

    valid_paths=[]
    for p in paths:
        if os.path.exists(p):
            valid_paths.append(p)
        else:
            print(f"warning: the plot {p} is not found")

    # I need to put 51 images (6*9=54 places) with ratio 4:3 (640x480 pixels)
    assert(len(valid_paths)<=ncols*nrows)
    plt.figure(0) # new figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3))

    for path_id, path in enumerate(valid_paths):
        raw_image = Image.open(path)
        i = path_id % ncols
        j = path_id // ncols
        axes[j, i].set_axis_off()
        axes[j, i].imshow(raw_image)
    plt.tight_layout()
    # write global scores in the last tile of the mosaic
    txt=f"mean(F1_scores):{total_experimental_result['f1']}\n" \
        f"tp:{total_experimental_result['tp']} tn:{total_experimental_result['tn']}\n" \
        f"fp:{total_experimental_result['fp']} fn:{total_experimental_result['fn']}\n" \
        f"enlapsed_time:{total_experimental_result['time']}"
    plt.text(0.01, 0.3, txt, fontdict={"fontsize": "x-large"})
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(fname)
    plt.close()

