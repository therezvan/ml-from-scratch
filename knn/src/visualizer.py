# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 17:31:04 2025

@author: rezi
"""

import numpy as np
import matplotlib.pyplot as plt


class Visualizer:

    def __init__(self, data_set):
        self.data_set = data_set

    def plot_feature_histograms(self):

        self.data_set.hist(bins=20, figsize=(15, 10),
                           color='lightblue', edgecolor='black')
        plt.suptitle("Feature Distributions", fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_label_distribution(self, column_name="diagnosis"):

        counts = self.data_set[column_name].value_counts()
        plt.bar(counts.index.astype(str), counts.values, color=[
                'lightcoral', 'lightblue'], edgecolor='black')
        plt.title('Diagnosis Class Distribution')
        plt.xlabel('Diagnosis (0 = Benign, 1 = Malignant)')
        plt.ylabel('Count')
        plt.show()
