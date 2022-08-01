from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np
from typing import Optional


def confusion_matrix_and_F1(y_pred,y_true):
    # Some timeseries does not contains abnormaly. We avoid division per zero by yielding accuracy instead of f1.
    # The returned score is always named "F1 score", for simplicity reasons.
    both_values = 0 in y_true and 1 in y_true
    if both_values:
        cm=confusion_matrix(y_true, y_pred)
        vals = cm.ravel()

        tn, fp, fn, tp = vals
        f1 = f1_score(y_true, y_pred)
        stats={"tn":tn, "fp":fp, "fn":fn, "tp":tp, "f1":round(f1,4)}
    else:
        accuracy=np.mean(y_pred==y_true)
        ones=np.ones(len(y_true))
        twos=ones*2
        zeros=np.zeros(len(y_true))
        tp=np.sum((y_pred+y_true)==twos)
        tn=np.sum((y_pred+y_true)==zeros)
        fp=np.sum((2*y_pred+y_true)==twos)
        fn=np.sum((y_pred+2*y_true)==twos)
        stats={"tn":tn, "fp":fp, "fn":fn, "tp":tp, "f1":round(accuracy,4)}
    return stats

def plot_curves(x_train:np.ndarray, x_test:np.ndarray, y_test:np.ndarray, y_pred:np.ndarray, frame_size:int,
                path:Optional[str]=None, txt:str= ""):
    x=np.concatenate((x_train,x_test))

    non_prediction_area=frame_size-1

    y_train_offset=np.zeros(len(x_train))
    y_non_pred_area=np.zeros(len(x_train)+non_prediction_area)

    y=np.concatenate((y_train_offset,y_test))
    y_pred=np.concatenate((y_non_pred_area,y_pred))

    assert(len(x)==len(y)==len(y_pred))

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

def mosaic(paths,fname):
    from matplotlib import pyplot as plt
    from PIL import Image
    # I need to put 51 images (6*9=54 places) with ratio 4:3 (640x480 pixels)
    ncols = 6
    nrows = 9
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3))
    for path_id, path in enumerate(paths):
        raw_image = Image.open(path)
        i = path_id % ncols
        j = path_id // ncols
        axes[j, i].set_axis_off()
        axes[j, i].imshow(raw_image)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(fname)

if __name__=="__main__":
    x=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.2,0.2,0.8,0.9])
    y=    np.array([0,0,0,0,0,0,0,1,1,0,0])
    y_pred=np.array([0,0,0,0,0,0,0,0,1,1,0])
    plot_curves(x[0:5], x[5:], y[5:],y_pred[5:], 1, "./media/unit_test_of_plot_curves.png","unit_test_of_plot_curves\nF1-score:0.3333")


    #res=confusion_matrix_and_F1(np.array([1,0,0]), np.array([1,1,1]))
    #print(res)