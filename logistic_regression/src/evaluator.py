# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 17:29:29 2025

@author: rezi
"""

import numpy as np


class Evaluator:

    # -------- Confusion Matrix (General) --------
    def confusion_matrix(self, y_true, y_pred, num_classes):

        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm

    # -------- Metrics (General) --------
    def accuracy(self, cm):
        return np.trace(cm) / np.sum(cm)

    def precision_per_class(self, cm):
        precisions = []
        for k in range(len(cm)):
            TP = cm[k, k]
            FP = cm[:, k].sum() - TP
            precisions.append(TP / (TP + FP + 1e-9))
        return precisions

    def recall_per_class(self, cm):
        recalls = []
        for k in range(len(cm)):
            TP = cm[k, k]
            FN = cm[k, :].sum() - TP
            recalls.append(TP / (TP + FN + 1e-9))
        return recalls

    def f1_per_class(self, cm):
        p = self.precision_per_class(cm)
        r = self.recall_per_class(cm)
        return [2*p[i]*r[i]/(p[i]+r[i]+1e-9) for i in range(len(p))]

    def macro_f1(self, cm):
        return np.mean(self.f1_per_class(cm))

    # -------- Binary Shortcut  --------
    def evaluate_binary(self, y_true, y_pred):
        cm = self.confusion_matrix(y_true, y_pred, num_classes=2)

        precision = self.precision_per_class(cm)[1]
        recall = self.recall_per_class(cm)[1]
        f1 = self.f1_per_class(cm)[1]

        return {
            "accuracy": self.accuracy(cm),
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
