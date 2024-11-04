import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from partB_to_D.neural_network import generate_data
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor


def test_regression_with_sklearn():
    X_train, X_test, z_train, z_test, scaler_Z = generate_data()

    param_grid = {
        "learning_rate_init": [0.0001, 0.001, 0.01, 0.1],
        "alpha": [1e-6, 1e-4, 1e-2, 1e-1],
    }
    # comparing on the sci-kit learn implementation.
    mlp = MLPRegressor(
        hidden_layer_sizes=(5, 5),
        activation="logistic",
        max_iter=1000,
        random_state=42,
    )
    # Perform Grid Search
    grid_search = GridSearchCV(mlp, param_grid, scoring="neg_mean_squared_error", cv=5)
    grid_search.fit(X_train, z_train)

    # Get the best parameters
    print("Best parameters found: ", grid_search.best_params_)

    y_pred_test = grid_search.best_estimator_.predict(X_test)
    y_pred_train = grid_search.best_estimator_.predict(X_train)

    mse_train = mean_squared_error(z_train, y_pred_train)
    r2_train = r2_score(z_train, y_pred_train)

    mse_test = mean_squared_error(z_test, y_pred_test)
    r2_test = r2_score(z_test, y_pred_test)

    print(
        f"Test MSE: {mse_test}, Test R2: {r2_test}, Train MSE: {mse_train}, Train R2: {r2_train}"
    )


if __name__ == "__main__":
    test_regression_with_sklearn()
