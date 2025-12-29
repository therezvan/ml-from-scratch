# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 13:22:12 2025

@author: rezi
"""
import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):

        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))

    def _get_neighbor(self, test_sample):
        distances = []
        for i in range(len(self.X_train)):
            dist = self.euclidean_distance(test_sample, self.X_train[i])
            distances.append((dist, self.y_train[i]))

        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.k]
        return neighbors

# predict all test samples
    def predict(self, X_test):
        predictions = []
        for sample in X_test:
            neighbors = self._get_neighbor(sample)
            labels = []
            for n in neighbors:
                labels.append(n[1])

            most_common = max(set(labels), key=labels.count)
            predictions.append(most_common)
        return np.array(predictions)

# prefict a single instance
    def predict_for_one(self, test_sample):
        neighbors = self._get_neighbor(test_sample)
        labels = [n[1] for n in neighbors]
        vote_result = Counter(labels).most_common(1)[0][0]
        return vote_result

    def error_knn_ordinary(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return 1 - np.mean(y_pred == y_test)

    def error_rate(self, X_test, y_test):
        y_pred = self.predict(X_test)

        right_predictions = 0
        for i in range(len(y_test)):
            if y_pred[i] == y_test[i]:
                right_predictions += 1

        accuracy = right_predictions / len(y_test)
        error = 1 - accuracy
        return error
