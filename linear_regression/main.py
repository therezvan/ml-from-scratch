# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 13:40:32 2025

@author: rezi
"""
from src.dataloader import DataLoader
from src.preprocessor import Preprocessor, Encoder
from src.metrics import Metrics
from src.linearregression import LinearRegressionClosedForm, PolynomialRegressionClosedForm
from src.sgdregressor import SGDRegressor, AdaptiveSGD, Adam, RMSProp, AdaGrad

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Pipeline:

    def __init__(self, path):
        self.path = path
        self.loader = DataLoader(path)
        self.encoder = Encoder()

    def exploratory_data_analysis(self, df):
        print("\n===== DESCRIPTIVE STATISTICS =====")
        print(df.describe())

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Histogram
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) == 0:
                continue
            plt.hist(data, bins=20)
            plt.title(f"Histogram of {col}")
            plt.show()

        # Boxplots
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) == 0:
                continue
            plt.boxplot(data)
            plt.title(f"Boxplot of {col}")
            plt.show()

        # Correlation heatmap
        df_corr = df[numeric_cols].dropna(axis=1, how='all')
        if df_corr.shape[1] > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(df_corr.corr(), annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Correlation Matrix")
            plt.show()

    def run_sgd_experiments(self, X_train, y_train, X_test, y_test):

        batch_sizes = [4, 8, 16, 32, 64, 128]
        results = []

        for b in batch_sizes:
            print(f"\n=== Running SGD with batch size {b} ===")

            model = AdaptiveSGD(lr=0.01, epochs=50, batch_size=b)

            start = time.time()
            model.fit(X_train, y_train)
            elapsed = time.time() - start

            preds = model.predict(X_test)

            mse = Metrics.mse(y_test, preds)
            rmse = Metrics.rmse(y_test, preds)
            mae = Metrics.mae(y_test, preds)
            r2 = Metrics.r2(y_test, preds)

            print(f"Time: {elapsed:.4f}s | MSE={mse:.4f} | RMSE={
                  rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f}")

            results.append({
                "batch_size": b,
                "time": elapsed,
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2
            })

        return pd.DataFrame(results)

    # -----------------------------
    # Polynomial Regression Section
    # -----------------------------
    def run_polynomial_regression(self):

        pre = Preprocessor(None)

        # Generate polynomial features
        X_train_poly = pre.polynomial_features(self.X_train)
        X_test_poly = pre.polynomial_features(self.X_test)

        results = {}

        # Closed-form polynomial regression
        model_cf = PolynomialRegressionClosedForm()
        start = time.time()
        model_cf.fit(X_train_poly, self.y_train)
        t_cf = time.time() - start
        pred_cf = model_cf.predict(X_test_poly)

        results["Closed-Form Polynomial"] = {
            "MSE": Metrics.mse(self.y_test, pred_cf),
            "RMSE": Metrics.rmse(self.y_test, pred_cf),
            "MAE": Metrics.mae(self.y_test, pred_cf),
            "R2": Metrics.r2(self.y_test, pred_cf),
            "Time": t_cf
        }

        # SGD polynomial regression
        model_sgd = SGDRegressor(lr=0.01, epochs=80, batch_size=32)
        start = time.time()
        model_sgd.fit(X_train_poly, self.y_train)
        t_sgd = time.time() - start
        pred_sgd = model_sgd.predict(X_test_poly)

        results["SGD Polynomial"] = {
            "MSE": Metrics.mse(self.y_test, pred_sgd),
            "RMSE": Metrics.rmse(self.y_test, pred_sgd),
            "MAE": Metrics.mae(self.y_test, pred_sgd),
            "R2": Metrics.r2(self.y_test, pred_sgd),
            "Time": t_sgd
        }

        return results

    # -----------------------------
    # Optimization : AdaGrad , RMSProp , Adam
    # -----------------------------
    def run_optimization_methods(self):
        methods = {
            "AdaGrad": AdaGrad(lr=0.01, epochs=50, batch_size=32),
            "RMSProp": RMSProp(lr=0.01, epochs=50, batch_size=32),
            "Adam": Adam(lr=0.01, epochs=50, batch_size=32)
        }

        results = {}
        plt.figure(figsize=(8, 5))

        for name, model in methods.items():
            print(f"\n=== Running {name} ===")
            losses = []
            model.fit(self.X_train, self.y_train, record_loss=losses)
            preds = model.predict(self.X_test)

            mse = Metrics.mse(self.y_test, preds)
            rmse = Metrics.rmse(self.y_test, preds)
            mae = Metrics.mae(self.y_test, preds)
            r2 = Metrics.r2(self.y_test, preds)

            results[name] = {"MSE": mse, "RMSE": rmse,
                             "MAE": mae, "R2": r2, "LossCurve": losses}
            plt.plot(losses, label=name)

        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.show()

        return results

    def compare_with_libraries(self):

        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import statsmodels.api as sm

        X_train = self.X_train
        X_test = self.X_test

        y_train = self.y_train
        y_test = self.y_test

        # -----------------------------
        # scikit-learn Linear Regression
        # -----------------------------
        start = time.time()
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        elapsed_skl = time.time() - start

        y_pred_skl = lr.predict(X_test)

        mse_skl = mean_squared_error(y_test, y_pred_skl)
        rmse_skl = np.sqrt(mse_skl)
        mae_skl = mean_absolute_error(y_test, y_pred_skl)
        r2_skl = r2_score(y_test, y_pred_skl)

        print("\n===== scikit-learn Linear Regression =====")
        print(f"Time: {elapsed_skl:.4f}s")
        print(f"MSE:  {mse_skl:.3f}")
        print(f"RMSE: {rmse_skl:.3f}")
        print(f"MAE:  {mae_skl:.3f}")
        print(f"R2:   {r2_skl:.4f}")

        # -----------------------------
        # statsmodels OLS
        # -----------------------------
        start = time.time()
        X_train_sm = sm.add_constant(X_train)
        X_test_sm = sm.add_constant(X_test)
        model_sm = sm.OLS(y_train, X_train_sm).fit()
        elapsed_sm = time.time() - start

        y_pred_sm = model_sm.predict(X_test_sm)

        mse_sm = mean_squared_error(y_test, y_pred_sm)
        rmse_sm = np.sqrt(mse_sm)
        mae_sm = mean_absolute_error(y_test, y_pred_sm)
        r2_sm = r2_score(y_test, y_pred_sm)

        print("\n===== statsmodels OLS =====")
        print(f"Time: {elapsed_sm:.4f}s")
        print(f"MSE:  {mse_sm:.3f}")
        print(f"RMSE: {rmse_sm:.3f}")
        print(f"MAE:  {mae_sm:.3f}")
        print(f"R2:   {r2_sm:.4f}")

    # -----------------------------
    # MAIN RUN
    # -----------------------------

    def run(self):

        df = self.loader.load()

        print("\n===== DATA TYPES BEFORE PROCESSING =====")
        print(df.dtypes)

        # Convert textual numbers
        object_cols = df.select_dtypes(include=['object']).columns
        df = self.loader.clean_numeric_columns(df, object_cols)

        # Fill NaN (missing values)
        pre = Preprocessor(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = pre.fill_missing(df, numeric_cols)

        self.exploratory_data_analysis(df)

        # Encoding
        cat_cols = df.select_dtypes(include=['object']).columns
        low_card = [c for c in cat_cols if df[c].nunique() <= 10]
        high_card = [c for c in cat_cols if df[c].nunique() > 10]
        df = self.encoder.encode(df, low_card, high_card)

        print("\n===== DATA TYPES AFTER PROCESSING =====")
        print(df.dtypes)

        # Rebuild preprocessor
        pre = Preprocessor(df)

        # X / y
        y = df["Price"]
        X = df.drop(columns=["Price"]).to_numpy()

        X_train, X_test, y_train, y_test = pre.split(X, y, test_size=0.1)

        # Save inside class
        self.X_train = np.nan_to_num(X_train)
        self.X_test = np.nan_to_num(X_test)
        self.y_train = y_train
        self.y_test = y_test

        # Remove zero variance + scale
        self.X_train, self.X_test = pre.remove_zero_variance(
            self.X_train, self.X_test)
        self.X_train, self.X_test = pre.scale(self.X_train, self.X_test)
        self.X_train, self.X_test = pre.remove_zero_variance(
            self.X_train, self.X_test)

        # Add bias
        self.X_train = pre.add_bias(self.X_train)
        self.X_test = pre.add_bias(self.X_test)

        # Linear regression (closed-form)
        model_cf = LinearRegressionClosedForm()
        model_cf.fit(self.X_train, self.y_train)
        pred_cf = model_cf.predict(self.X_test)

        print("\n===== CLOSED-FORM LINEAR REGRESSION RESULTS =====")
        print("MSE :", Metrics.mse(self.y_test, pred_cf))
        print("RMSE:", Metrics.rmse(self.y_test, pred_cf))
        print("MAE :", Metrics.mae(self.y_test, pred_cf))
        print("R²  :", Metrics.r2(self.y_test, pred_cf))

        # SGD experiments
        sgd_results = self.run_sgd_experiments(
            self.X_train, self.y_train, self.X_test, self.y_test
        )

        # Polynomial Regression Experiments
        poly_results = self.run_polynomial_regression()

        # Optimization methods
        opt_results = self.run_optimization_methods()
        pre.polynomial_features

        print("\n===== POLYNOMIAL REGRESSION RESULTS =====")
        for name, res in poly_results.items():
            print(f"\n--- {name} ---")
            for k, v in res.items():
                print(f"{k}: {v}")

        self.compare_with_libraries()

        return {
            "LinearClosedForm": pred_cf,
            "SGD_Results": sgd_results,
            "Polynomial_Results": poly_results
        }


# RUN
if __name__ == "__main__":
    Pipeline("data/data.csv").run()
