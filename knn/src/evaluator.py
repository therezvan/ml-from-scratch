# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 17:29:29 2025

@author: rezi
"""
from .preprocessor import Preprocessor
from .knn import KNN
from .visualizer import Visualizer
import numpy as np
from collections import Counter


class Evaluator:

    def __init__(self, model, data_set, k_folds=5):

        self.model = model
        self.data_set = data_set
        self.k_folds = k_folds

    def cross_validation(self):

        X, y = self.data_set.normalize()
        y = y.to_numpy()
        n = len(X)
        fold_size = n // self.k_folds

        indices = np.arange(n)
        np.random.shuffle(indices)

        accuracies = []

        for i in range(self.k_folds):
            start = i*fold_size
            end = (i+1)*fold_size if i != self.k_folds - 1 else n
            test_idx = indices[start:end]
            train_idx = np.concatenate((indices[:start], indices[end:]))

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)

            acc = np.mean(y_pred == y_test)
            accuracies.append(acc)

            print(f"Fold {i+1} Accuracy: {acc:.3f}")

        avg_acc = np.mean(accuracies)
        print(f"\nAverage Accuracy: {avg_acc:.3f}")

        return avg_acc
