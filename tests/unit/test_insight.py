import unittest
import numpy as np
import os
from ...insight import plot_curves,confusion_matrix_and_F1,ROCAUC

class Test(unittest.TestCase):
    def test_rocauc(self):
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],dtype=float)
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0.7, 0.2, 0.9, 0.3, 0.4],dtype=float)
        infos=ROCAUC(y_pred,y_true)
        self.assertTrue(0.999>infos["rocauc"]>0.001)
        self.assertTrue(0.999>infos["proposed_thresh"]>0.001)
        self.assertTrue(0.999>infos["new_f1"]>0.001)
    def test_f1(self):
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],dtype=float)
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0.7, 0.2, 0.9, 0.3, 0.4],dtype=float)
        infos=confusion_matrix_and_F1(y_pred>0.5,y_true)
        s=infos["tn"]+infos["tp"]+infos["fn"]+infos["fp"]
        self.assertTrue(s==len(y_true))
        self.assertTrue(0.999>infos["f1"]>0.001)
    def test_f1_normal(self):
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0.8, 0.9, 0.3, 0.4])
        infos=confusion_matrix_and_F1(y_pred>0.5,y_true)
        s=infos["tn"]+infos["tp"]+infos["fn"]+infos["fp"]
        self.assertTrue(s==len(y_true))
        self.assertTrue(0.999>infos["f1"]>0.5)

    def test_curve(self):
        x = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.2, 0.2, 0.8, 0.9])
        y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0])
        path="unit_test_of_plot_curves.png"
        plot_curves(x[0:5], x[5:], y[5:], y_pred[5:], 1,
                    path,
                    "unit_test_of_plot_curves\nF1-score:0.3333")
        self.assertTrue(os.path.exists(path))
        os.remove(path)


