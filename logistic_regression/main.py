# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 14:03:35 2025

@author: rezi
"""

import numpy as np
import matplotlib.pyplot as plt
from src.preprocessor import Preprocessor
from src.knn import KNN
from src.evaluator import Evaluator
from src.logistic_regression import LogisticRegression


def main():
    binary_classification()
    multi_classification()


def binary_classification():
    prep = Preprocessor()

    df = prep.load_data("data/wdbc.csv")
    # clean & encode labels
    df = prep.clean_data()

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    # split
    X_train, X_test, y_train, y_test = prep.split(
        X, y, test_size=0.2, random_state=42
    )

    # normalization
    X_train_scaled, X_test_scaled = prep.scale(X_train, X_test)

    X_train_scaled, X_test_scaled = X_train_scaled.to_numpy(), X_test_scaled.to_numpy()

    # ------------------ Logistic Reression ----------
    model = LogisticRegression(lr=0.01, epochs=3000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    evaluator1 = Evaluator()
    res = evaluator1.evaluate_binary(y_test, y_pred)

    print("\n--- Logistic Regression (Binary) ---")
    for k, v in res.items():
        print(f"{k}: {v}")

    # plot loss
    plt.plot(model.loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("Loss Curve for Binary Logistic Regression")
    plt.grid(True)
    plt.show()

    print("Final Training Loss:", model.loss_history[-1])

    # -------------------KNN --------------------
    evaluator2 = Evaluator()

    k_values = [1, 3, 5, 7, 9, 11]
    f1_scores = []
    print("\n--- Knn ---")
    for k in k_values:
        knn = KNN(k=k)
        knn.fit(X_train_scaled, y_train)

        y_pred = knn.predict(X_test_scaled)

        # binary evaluation (class 1 is positive)
        results = evaluator2.evaluate_binary(y_test, y_pred)

        f1_scores.append(results["f1"])

        print(f"\nFor k = {k}")
        print("Accuracy :", results["accuracy"])
        print("Precision:", results["precision"])
        print("Recall   :", results["recall"])
        print("F1-score :", results["f1"])

    # plot
    plt.plot(k_values, f1_scores, marker='o')
    plt.xlabel("k")
    plt.ylabel("F1-score")
    plt.title("k vs F1-score (Breast Cancer)")
    plt.show()

# ===============================
# Multiclass Classification
# ===============================


def multi_classification():

    prep = Preprocessor()
    df = prep.load_data("data/winequality-white.csv", sep=';')

    X = df.drop("quality", axis=1)
    y = df["quality"] - 3   # classes: 0..5

    X_train, X_test, y_train, y_test = prep.split(X, y, test_size=0.2)
    X_train_scaled, X_test_scaled = prep.scale(X_train, X_test)
    X_train_scaled, X_test_scaled = X_train_scaled.to_numpy(), X_test_scaled.to_numpy()

    model = LogisticRegression(
        lr=0.05,
        epochs=4000,
        multi_class=True
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    evaluator = Evaluator()
    num_classes = max(y_test.max(), y_pred.max()) + 1
    cm = evaluator.confusion_matrix(y_test, y_pred, num_classes)

    print("\n--- Logistic Regression (Multiclass) ---")
    print("Accuracy :", evaluator.accuracy(cm))
    print("Macro F1 :", evaluator.macro_f1(cm))

    plt.figure()
    plt.plot(range(1, len(model.loss_history) + 1), model.loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Softmax Cross-Entropy Loss")
    plt.title("Training Loss vs Epoch (Softmax Logistic Regression)")
    plt.grid(True)
    plt.show()
    print("Final training loss:", model.loss_history[-1])


if __name__ == "__main__":
    main()
