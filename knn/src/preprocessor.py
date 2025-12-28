
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 11:08:26 2025

@author: rezi
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


class Preprocessor:

    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

    def clean_data(self):
        self.df = self.df.drop(columns=['id', 'Unnamed: 32'])
        self.df["diagnosis"] = self.df["diagnosis"].map({"M": 1, "B": 0})

        return self.df

    def normalize(self):
        scaler = StandardScaler()
        X = self.df.drop("diagnosis", axis=1)
        y = self.df["diagnosis"]
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y

    def split(self, X, y, test_size=0.2, random_state=None):

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
