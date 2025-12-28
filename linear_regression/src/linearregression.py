# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 16:06:25 2025

@author: rezi
"""
import numpy as np


class LinearRegressionClosedForm:

    def fit(self, X, y):
        self.w = np.linalg.pinv(X, rcond=1e-6).dot(y)

    def predict(self, X):
        return X.dot(self.w)


class PolynomialRegressionClosedForm:
    def fit(self, X, y):
        self.w = np.linalg.pinv(X.T @ X) @ (X.T @ y)

    def predict(self, X):
        return X @ self.w
