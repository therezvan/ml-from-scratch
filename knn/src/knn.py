# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 13:22:12 2025

@author: rezi
"""
import numpy as np
from collections import Counter
import heapq
from .kdtree import KDtree


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.kdtree = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.kdtree = KDtree(X_train, y_train)

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

    def predict_for_one(self, test_sample):
        neighbors = self._get_neighbor(test_sample)
        labels = [n[1] for n in neighbors]
        vote_result = Counter(labels).most_common(1)[0][0]
        return vote_result

    def predict_with_heap_for_one(self, test_sample, k=None):
        if k is None:
            k = self.k

        heap = []
        for i in range(len(self.X_train)):
            dist = self.euclidean_distance(test_sample, self.X_train[i])
            if len(heap) < k:
                heapq.heappush(heap, (-dist, self.y_train[i]))
            else:
                if -dist > heap[0][0]:
                    heapq.heappushpop(heap, (-dist, self.y_train[i]))
        k_labels = [label for (_, label) in heap]
        return Counter(k_labels).most_common(1)[0][0]

    def predict_with_heap(self, X_test, k=None):
        if k is None:
            k = self.k

        predictions = []
        for sample in X_test:
            pred = self.predict_with_heap_for_one(sample, k)
            predictions.append(pred)
        return np.array(predictions)

    def predict_with_KDtree(self, X_test, k=None):
        if k is None:
            k = self.k

        if self.kdtree is None:
            raise ValueError("You must call fit() before using KD-tree.")

        predictions = []
        for x in X_test:
            neighbors = self.kdtree.query(x, k)
            labels = [label for (label, dist) in neighbors]
            pred = Counter(labels).most_common(1)[0][0]
            predictions.append(pred)

        return np.array(predictions)

    def error_knn_KDtree(self, X_test, y_test):
        y_pred = self.predict_with_KDtree(X_test, k=self.k)
        return 1 - np.mean(y_pred == y_test)

    def error_knn_heap(self, X_test, y_test):
        y_pred = self.predict_with_heap(X_test)
        return 1 - np.mean(y_pred == y_test)

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
