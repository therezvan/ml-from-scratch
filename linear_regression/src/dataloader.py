# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 15:39:56 2025

@author: rezi
"""
import pandas as pd
import numpy as np
import re


class DataLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        df = pd.read_csv(self.path)
        return df

    def extract_number(self, x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).replace(",", "")
        m = re.search(r"(-?\d+(\.\d+)?)", s)
        return float(m.group(1)) if m else np.nan

    def clean_numeric_columns(self, df, columns):
        for c in columns:
            df[c] = df[c].apply(self.extract_number)
        return df
