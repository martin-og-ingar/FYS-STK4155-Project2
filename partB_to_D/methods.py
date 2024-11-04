import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def ols_regression(n=100, noise=0.1, max_degree=5):
    np.random.seed(42)
    x = np.random.rand(n)
    y = np.random.rand(n)
    noise = noise * np.random.randn(n)
    z = franke_function(x, y) + noise
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_x.fit_transform(x.reshape(-1, 1))
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    x_combined = np.hstack((X_scaled, y_scaled))
    X_train, X_test, z_train, z_test = train_test_split(x_combined, z, test_size=0.2)

    def prep_poly_features(x, degree):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(x)
        return X_poly

    mse_train_results = {}
    mse_test_results = {}
    r2_train_results = {}
    r2_test_results = {}
    beta_values = []
    res = []

    for degree in range(1, max_degree + 1):
        X_poly_train = prep_poly_features(X_train, degree)
        model = LinearRegression()
        model.fit(X_poly_train, z_train)

        z_pred_train = model.predict(X_poly_train)

        X_poly_test = prep_poly_features(X_test, degree)
        z_pred_test = model.predict(X_poly_test)

        betas = model.coef_

        mse_train = mean_squared_error(z_train, z_pred_train)
        mse_test = mean_squared_error(z_test, z_pred_test)
        r2_train = r2_score(z_train, z_pred_train)
        r2_test = r2_score(z_test, z_pred_test)

        mse_train_results[degree] = mse_train
        mse_test_results[degree] = mse_test
        beta_values.append(betas)
        res.append((degree, mse_train, mse_test, r2_train, r2_test))

    results = pd.DataFrame(
        res, columns=["Degree", "Train MSE", "Test MSE", "Train R²", "Test R²"]
    )

    return results


def ridge_regression(n=100, noise=0.1, max_degree=5):
    np.random.seed(123)
    n = 100
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    noise = 0.1 * np.random.normal(0, 1, n)
    Z = franke_function(x, y) + noise

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    x_scaled = scaler_x.fit_transform(x.reshape(-1, 1))
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    x_combined = np.hstack((x_scaled, y_scaled))

    # Split the data into training and test data
    x_train, x_test, z_train, z_test = train_test_split(x_combined, Z, test_size=0.2)

    lambda_values = [0.001, 0.01, 0.1, 1, 10]
    mse_values = {lmb: [] for lmb in lambda_values}
    mse_values_test = {lmb: [] for lmb in lambda_values}
    r2_values_train = {lmb: [] for lmb in lambda_values}
    r2_values_test = {lmb: [] for lmb in lambda_values}
    res = {lmb: [] for lmb in lambda_values}

    degrees = range(1, 6)

    for degree in degrees:
        poly = PolynomialFeatures(degree=degree)
        x_train_poly = poly.fit_transform(x_train)
        x_test_poly = poly.transform(x_test)
        for lmb in lambda_values:
            beta_ridge = (
                np.linalg.inv(
                    x_train_poly.T @ x_train_poly + lmb * np.eye(x_train_poly.shape[1])
                )
                @ x_train_poly.T
                @ z_train
            )

            # pred of train and test
            z_pred_train = x_train_poly @ beta_ridge
            z_pred_test = x_test_poly @ beta_ridge
            mse_train = mean_squared_error(z_train, z_pred_train)
            mse_test = mean_squared_error(z_test, z_pred_test)
            r2_train = r2_score(z_train, z_pred_train)
            r2_test = r2_score(z_test, z_pred_test)
            mse_values_test[lmb].append(mse_test)
            mse_values[lmb].append(mse_train)
            res[lmb].append((degree, mse_train, mse_test, r2_train, r2_test))

    ridge_results_df = {
        lmb: pd.DataFrame(
            res, columns=["Degree", "Train MSE", "Test MSE", "Train R²", "Test R²"]
        )
        for lmb, res in res.items()
    }
    return ridge_results_df


import os
import matplotlib.pyplot as plt
import numpy as np


def franke_function(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


def save_plot(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "Figures")
    os.makedirs(output_dir, exist_ok=True)

    output_file_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(output_file_path)
    plt.close()


def add_to_compare_log_training_score_csv(
    function_name, lr, lmb, mb_size, score, epochs
):
    """
    Adds the current gradient descent execution to a csv file named in partA.py
    """
    filepath = f"partE/results/compare_log_training_scores.csv"

    if not os.path.exists(filepath):
        header = [
            "function name",
            "learning rate",
            "lambda",
            "mini batch size",
            "epochs",
            "score",
        ]
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)

    with open(filepath, mode="a", newline="") as file:
        writer = csv.writer(file)
        row = []
        row.append(function_name)
        row.append(lr or "N/A")
        row.append(lmb or "N/A")
        row.append(mb_size or "N/A")
        row.append(epochs or "N/A")
        row.append(score or "N/A")
        writer.writerow(row)
