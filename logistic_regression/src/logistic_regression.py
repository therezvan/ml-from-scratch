# -*- coding: utf-8 -*-
"""
Logistic Regression (Binary & Multiclass)
@author: rezi
"""

import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01, epochs=100, multi_class=False):
        self.lr = lr
        self.epochs = epochs
        self.multi_class = multi_class
        self.W = None
        self.loss_history = []

    # ---------- utils ----------
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def one_hot(self, y, C):
        oh = np.zeros((len(y), C))
        oh[np.arange(len(y)), y] = 1
        return oh

    # ---------- loss ----------
    def binary_cross_entropy(self, y, y_hat):
        eps = 1e-9
        return -np.mean(y * np.log(y_hat + eps) +
                        (1 - y) * np.log(1 - y_hat + eps))

    def categorical_cross_entropy(self, y_oh, probs):
        eps = 1e-9
        return -np.mean(np.sum(y_oh * np.log(probs + eps), axis=1))

    # ---------- train ----------
    def fit(self, X, y):
        n, d = X.shape

        # bias
        X = np.hstack([np.ones((n, 1)), X])

        if not self.multi_class:
            self.W = np.zeros(d + 1)

            for _ in range(self.epochs):
                z = X @ self.W
                y_hat = self.sigmoid(z)

                # loss
                loss = self.binary_cross_entropy(y, y_hat)
                self.loss_history.append(loss)

                # gradient
                grad = X.T @ (y_hat - y) / n
                self.W -= self.lr * grad

        else:
            C = len(np.unique(y))
            y_oh = self.one_hot(y, C)

            self.W = np.zeros((d + 1, C))

            for _ in range(self.epochs):
                scores = X @ self.W
                probs = self.softmax(scores)

                # loss
                loss = self.categorical_cross_entropy(y_oh, probs)
                self.loss_history.append(loss)

                # gradient
                grad = X.T @ (probs - y_oh) / n
                self.W -= self.lr * grad

    # ---------- predict ----------
    def predict(self, X):
        n = X.shape[0]
        X = np.hstack([np.ones((n, 1)), X])

        if not self.multi_class:
            probs = self.sigmoid(X @ self.W)
            return (probs >= 0.5).astype(int)
        else:
            scores = X @ self.W
            return np.argmax(scores, axis=1)
