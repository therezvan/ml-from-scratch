
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 11:08:26 2025

@author: rezi
"""

import pandas as pd
import numpy as np


class Preprocessor:

    def __init__(self, df):
        self.df = df

    def fill_missing(self, df, cols):
        for c in cols:
            df[c] = df[c].fillna(df[c].median())
        return df

    def split(self, X, y, test_size=0.1, random_state=None):

        if random_state != None:
            np.random.seed(random_state)

        index = np.arange(len(X))
        np.random.shuffle(index)

        split_index = int(len(X)*(1-test_size))
        train_index = index[:split_index]
        test_index = index[split_index:]

        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index].to_numpy()
        y_test = y[test_index].to_numpy()

        return X_train, X_test, y_train, y_test

    def scale(self, X_train, X_test):

        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)

        # اگر خیلی خیلی کوچک بود → 1
        std[std < 1e-8] = 1

        X_train_scaled = (X_train - mean) / std
        X_test_scaled = (X_test - mean) / std

        return X_train_scaled, X_test_scaled

    def add_bias(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def remove_zero_variance(self, X_train, X_test):
        variances = X_train.var(axis=0)
        keep = variances > 1e-12
        return X_train[:, keep], X_test[:, keep]

    def polynomial_features(self, X):
        n, d = X.shape
        features = [X]          # degree 1
        features.append(X ** 2)  # squared terms

        # cross terms
        cross = []
        for i in range(d):
            for j in range(i+1, d):
                cross.append((X[:, i] * X[:, j]).reshape(-1, 1))

        if cross:
            features.append(np.hstack(cross))

        return np.hstack(features)


class Encoder:

    def frequency_encode(self, series):
        freq = series.value_counts(normalize=True)
        return series.map(freq)

    def one_hot(self, df, col):
        dummies = pd.get_dummies(df[col], prefix=col)
        return pd.concat([df.drop(columns=[col]), dummies], axis=1)

    def encode(self, df, low_card_cols, high_card_cols):
        # Frequency Encoding
        for c in high_card_cols:
            df[c] = self.frequency_encode(df[c])

        # One-hot Encoding
        for c in low_card_cols:
            df = self.one_hot(df, c)

        return df
