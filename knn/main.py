# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 11:07:26 2025

@author: rezi
"""
from src.preprocessor import Preprocessor
from src.knn import KNN
from src.visualizer import Visualizer
from src.evaluator import Evaluator
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def main():

    prep = Preprocessor("data/Cancer_Data.csv")
    df = prep.clean_data()
    X, y = prep.normalize()
    X_train, X_test, y_train, y_test = prep.split(X, y, random_state=42)

    vis = Visualizer(df)
    vis.plot_feature_histograms()
    vis.plot_label_distribution()
    # lookin for the best k
    k_values = [1, 3, 5, 7, 9, 11, 13]
    cv_errors = []

    for k in k_values:
        print(f"for k = {k}")
        knn = KNN(k=k)
        evaluator = Evaluator(model=knn, data_set=prep, k_folds=5)
        cv_acc = evaluator.cross_validation()
        cv_error = 1 - cv_acc
        cv_errors.append(cv_error)

    best_k = k_values[np.argmin(cv_errors)]
    print("Optimal k from cross-validation:", best_k)

    plt.figure(figsize=(6, 4))
    plt.plot(k_values, cv_errors, marker='o')
    plt.xlabel("k")
    plt.ylabel("Cross-Validation Error")
    plt.title("CV Error vs k")
    plt.show()

    knn = KNN(k=best_k)
    knn.fit(X_train, y_train)

    methods = ["Ordinary", "Heap", "KD-tree"]
    times = []
    test_errors = []
    # --- Ordinary KNN ---
    start = time.time()
    error_ordinary = knn.error_knn_ordinary(X_test, y_test)
    end = time.time()
    times.append(end - start)
    test_errors.append(error_ordinary)

    # --- Heap-based KNN ---
    start = time.time()
    error_heap = knn.error_knn_heap(X_test, y_test)
    end = time.time()
    times.append(end - start)
    test_errors.append(error_heap)

    # --- KD-tree KNN ---
    start = time.time()
    error_kdtree = knn.error_knn_KDtree(X_test, y_test)
    end = time.time()
    times.append(end - start)
    test_errors.append(error_kdtree)

    for method, t, err in zip(methods, times, test_errors):
        print(f"{method}: Test Error={err:.4f}, Time={t:.4f} sec")

    plt.bar(methods, times)
    plt.ylabel("Time (sec)")
    plt.title("Comparison of Execution Time")
    plt.show()

    y_pred = knn.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Ordinary KNN")
    plt.show()


if __name__ == "__main__":
    main()
