# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 16:17:11 2025

@author: rezi
"""
import numpy as np


class Metrics:
    @staticmethod
    def mse(y, y_pred):
        return np.mean((y - y_pred) ** 2)

    @staticmethod
    def rmse(y, y_pred):
        return np.sqrt(Metrics.mse(y, y_pred))

    @staticmethod
    def mae(y, y_pred):
        return np.mean(np.abs(y - y_pred))

    @staticmethod
    def r2(y, y_pred):
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        return 1 - ss_res/ss_tot
