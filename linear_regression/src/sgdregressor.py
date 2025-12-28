# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 16:15:41 2025

@author: rezi
"""
import numpy as np
from .metrics import Metrics
from sklearn.preprocessing import StandardScaler


# -*- coding: utf-8 -*-
"""
SGD, AdaGrad, RMSProp, Adam for Linear Regression (no y scaling)
@author: rezi
"""


# =========================================
# Base SGD (no y scaling)
# =========================================

class SGDRegressor:
    def __init__(self, lr=1e-6, epochs=50, batch_size=32):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        for epoch in range(self.epochs):
            idx = np.random.permutation(n_samples)
            X_shuf = X[idx]
            y_shuf = y[idx]

            for i in range(0, n_samples, self.batch_size):
                xb = X_shuf[i:i+self.batch_size]
                yb = y_shuf[i:i+self.batch_size]

                pred = xb @ self.w
                grad = xb.T @ (pred - yb) / len(xb)
                self.w -= self.lr * grad

    def predict(self, X):
        return X @ self.w


class AdaptiveSGD:
    def __init__(self, lr=0.01, epochs=50, batch_size=32):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        epsilon = 1e-8
        G = np.zeros(d)   # مجموع مربعات گرادیان‌ها (AdaGrad)

        for epoch in range(self.epochs):

            idx = np.random.permutation(n)
            Xs = X[idx]
            ys = y[idx]

            for i in range(0, n, self.batch_size):
                xb = Xs[i:i+self.batch_size]
                yb = ys[i:i+self.batch_size]

                pred = xb.dot(self.w)
                grad = (-2/len(xb)) * xb.T.dot(yb - pred)

                # AdaGrad update
                G += grad**2
                adjusted_lr = self.lr / (np.sqrt(G) + epsilon)

                self.w -= adjusted_lr * grad

    def predict(self, X):
        return X.dot(self.w)
# =========================================
# AdaGrad (no y scaling)
# =========================================


class AdaGrad(SGDRegressor):
    def fit(self, X, y, record_loss=None):
        n, d = X.shape
        self.w = np.zeros(d)
        G = np.zeros(d)
        eps = 1e-8

        for epoch in range(self.epochs):
            idx = np.random.permutation(n)
            Xs, ys = X[idx], y[idx]

            for i in range(0, n, self.batch_size):
                xb = Xs[i:i+self.batch_size]
                yb = ys[i:i+self.batch_size]

                pred = xb @ self.w
                grad = xb.T @ (pred - yb) / len(xb)

                G += grad**2
                self.w -= self.lr * grad / (np.sqrt(G) + eps)

            if record_loss is not None:
                record_loss.append(Metrics.mse(ys, Xs @ self.w))


# =========================================
# RMSProp (no y scaling)
# =========================================
class RMSProp(SGDRegressor):
    def fit(self, X, y, record_loss=None, beta=0.9):
        n, d = X.shape
        self.w = np.zeros(d)
        S = np.zeros(d)
        eps = 1e-8

        for epoch in range(self.epochs):
            idx = np.random.permutation(n)
            Xs, ys = X[idx], y[idx]

            for i in range(0, n, self.batch_size):
                xb = Xs[i:i+self.batch_size]
                yb = ys[i:i+self.batch_size]

                pred = xb @ self.w
                grad = xb.T @ (pred - yb) / len(xb)

                S = beta * S + (1 - beta) * grad**2
                self.w -= self.lr * grad / (np.sqrt(S) + eps)

            if record_loss is not None:
                record_loss.append(Metrics.mse(ys, Xs @ self.w))


# =========================================
# Adam (no y scaling)
# =========================================
class Adam(SGDRegressor):
    def fit(self, X, y, record_loss=None, beta1=0.9, beta2=0.999):
        n, d = X.shape
        self.w = np.zeros(d)
        m = np.zeros(d)
        v = np.zeros(d)
        eps = 1e-8
        t = 0

        for epoch in range(self.epochs):
            idx = np.random.permutation(n)
            Xs, ys = X[idx], y[idx]

            for i in range(0, n, self.batch_size):
                t += 1
                xb = Xs[i:i+self.batch_size]
                yb = ys[i:i+self.batch_size]

                pred = xb @ self.w
                grad = xb.T @ (pred - yb) / len(xb)

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)

                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)

                self.w -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

            if record_loss is not None:
                record_loss.append(Metrics.mse(ys, Xs @ self.w))
