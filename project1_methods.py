import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from methods import franke_function, save_plot


def ols_regression(n=1000, noise=0.1, max_degree=5):
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

        mse_train_results[degree] = mse_train
        mse_test_results[degree] = mse_test
        beta_values.append(betas)
        res.append((degree, mse_train, mse_test))

    return res


def ridge_regression(n=1000, noise=0.1, max_degree=5):
    np.random.seed(42)
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
            mse_values_test[lmb].append(mse_test)
            mse_values[lmb].append(mse_train)
            res[lmb].append((degree, mse_train, mse_test))
    return res


ridge_regression()
