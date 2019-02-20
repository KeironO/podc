'''
Copyright (c) 2019 Keiron O'Shea

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public
License as published by the Free Software Foundation; either
version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public
License along with this program; if not, write to the
Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
Boston, MA 02110-1301 USA
'''

from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    auc,
    roc_curve,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
import numpy as np


class Inference:
    def __init__(self, y_true=None, y_pred=None, y_predC=None):

        self._multi_model = False
        if type(y_pred[0]) == list:
            self._multi_model = True

        self.y_true = y_true
        self.y_pred = y_pred
        self.y_predC = y_predC

        if self.y_predC is None:
            self.y_predC = self._to_class()

    def _to_class(self):
        # predict in binary_crossentropy is equal to the probability
        # of it belonging to the FIRST class (0).
        if self._multi_model:
            return [
                (np.array(x) > 0.5).astype(np.int) for x in self.y_pred
            ]
        else:
            return (np.array(self.y_pred) < 0.5).astype(np.int)

    def confusion_matrix(self):
        if self._multi_model:
            pass
        else:
            return confusion_matrix(self.y_true, self.y_predC)

    def classification_report(self):
        if self._multi_model:
            pass
        else:
            return classification_report(self.y_true, self.y_predC)

    def accuracy_score(self):
        if self._multi_model:
            pass
        else:
            return accuracy_score(self.y_true, self.y_predC)

    def precision(self):
        if self._multi_model:
            pass
        else:
            return precision_score(self.y_true, self.y_predC)

    def recall(self):
        if self._multi_model:
            pass
        else:
            return recall_score(self.y_true, self.y_predC)

    def ppv(self):
        if self._multi_model:
            pass
        else:
            cm = self.confusion_matrix()
            tp = cm[0][0]
            fp = cm[0][1]

            return tp / (tp + fp)

    def npv(self):
        if self._multi_model:
            pass
        else:
            cm = self.confusion_matrix()
            fn = cm[1][0]
            tn = cm[1][1]

            return tn / (tn + fn)

    def roc_curve(self, fp=None):
        tpr, fpr, thresholds = roc_curve(self.y_true, self.y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color="black", lw=2, label="AUC: %0.2f" % roc_auc)
        plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        if fp != None:
            plt.savefig(fp, dpi=400)
        else:
            plt.show()

    def cohens_kappa(self):
        '''
        We use Cohen's kappa to measure the reliability of the diagnosis by
        measuring the agreement between the classifiers, subtracting out
        agreement due to chance.
        '''
        if self._multi_model:
            return cohen_kappa_score(*self.y_predC)
        else:
            return Exception("This is only available for comparing multiple models")

    def to_dict(self):
        results = {}
        results["y_pred"] = self.y_pred
        results["y_true"] = self.y_true
        results["y_predC"] = self.y_predC.tolist()
        results["cm"] = self.confusion_matrix().tolist()
        results["precision"] = self.precision()
        results["recall"] = self.recall()
        return results

