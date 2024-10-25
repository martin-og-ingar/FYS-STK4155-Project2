"""
"""

import os, sys

from sklearn.neural_network import MLPRegressor


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
from methods import franke_function, save_plot
from project1_methods import ols_regression, ridge_regression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class FeedForwardNeuralNetwork:
    def __init__(
        self,
        X_train,
        Z_train,
        layer_sizes,
        epochs,
        learning_rate,
        hidden_activation,
        batch_size=5,
        lmb=0.0,
    ):
        """
        Initialising the neural network.
        :param layer_sizes: list of integers, where each integer represents the number of nodes in the layer.
        """
        self.X_train = X_train
        self.Z_train = Z_train
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.hidden_activation = hidden_activation
        self.batch_size = batch_size
        self.lmb = lmb

        self.weights = []
        self.biases = []

        # create W and b for each layer.
        for i in range(1, self.num_layers):
            # weight is initialized to a small random value (normal distribution) with mean 0 and std 1.

            # USING XAVIER.
            W = np.random.randn(layer_sizes[i], layer_sizes[i - 1]) * np.sqrt(
                2 / (layer_sizes[i - 1] + layer_sizes[i])
            )

            # Can be initialized to a small random value(normal distribution) or zeros.
            # initializing to zeros in this case (common practice)
            b = np.zeros((layer_sizes[i], 1))
            self.weights.append(W)
            self.biases.append(b)

    def activate(self, z):
        if self.hidden_activation == "sigmoid":
            return self.sigmoid(z)
        elif self.hidden_activation == "relu":
            return self.relu(z)
        elif self.hidden_activation == "leaky_relu":
            return self.leaky_relu(z)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def leaky_relu(self, z):
        return np.where(z > 0, z, 0.01 * z)

    def leaky_relu_derivative(self, z):
        return np.where(z > 0, 1, 0.01)

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)  # Clip z to avoid overflow in exp
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    # MSE as it is a function typically used for regression tasks.
    # in addition it is a convex function, which makes it easier to find the global minimum.
    # also it was used in project 1.
    # also it is easy to differentiate because of the square term.

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def mse_derivative(self, y_true, y_pred):
        # can drop 2/n since it does not change the direction of the gradient, only the scale.
        return y_pred - y_true

    def feed_forward(self, x):
        """
        Perform a foreward pass through the network.

        :param x: input data of shape (input_size, 1)
        :return: output of the network
        """
        a = x  # activation function.
        activations = [a]
        z_values = []

        # loop through all layers except the last one.

        # This is becuase, for regression tasks, the final output layer should typically use linear activation function.
        # since we are predicting real-valued numbers.
        #
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(W, a) + b
            z_values.append(z)
            a = self.activate(z)  # apply activation function.
            activations.append(a)

        W_final_layer, b_final_layer = self.weights[-1], self.biases[-1]
        final_output = W_final_layer @ a + b_final_layer
        z_values.append(final_output)
        activations.append(final_output)

        return final_output, activations, z_values

    def back_prop(self, x, y_true):
        """
        Perform back propagation through the network.

        :param x: input data of shape (input_size, 1)
        :param y_true: true labels of shape (output_size, 1)
        """
        y_pred, activations, zs = self.feed_forward(x)

        # Init gradients.
        dW = [np.zeros(W.shape) for W in self.weights]
        db = [np.zeros(b.shape) for b in self.biases]

        m = x.shape[0]

        # gradient for output layer
        delta = self.mse_derivative(y_true, y_pred)

        # gradient for hidden layers.
        dW[-1] = (
            np.dot(delta, activations[-2].T) + (self.lmb / m) * self.weights[-1]
        )  # Gradient final layer weights
        db[-1] = delta  # Gradient final layer biases

        for l in range(2, self.num_layers):
            z = zs[-l]

            if self.hidden_activation == "sigmoid":
                delta = np.dot(self.weights[-l + 1].T, delta) * self.sigmoid_derivative(
                    z
                )
            elif self.hidden_activation == "relu":
                delta = np.dot(self.weights[-l + 1].T, delta) * self.relu_derivative(z)
            elif self.hidden_activation == "leaky_relu":
                delta = np.dot(
                    self.weights[-l + 1].T, delta
                ) * self.leaky_relu_derivative(z)
            # delta = np.dot(self.weights[-l + 1].T, delta) * self.sigmoid_derivative(z)
            dW[-l] = (
                np.dot(delta, activations[-l - 1].T) + (self.lmb / m) * self.weights[-l]
            )
            db[-l] = delta
        return dW, db

    def update_params(self, dW, db):
        """
        Update the weights and biases of the network.
        :param dW: gradients for the weights
        :param db: gradients for the biases
        :param learning_rate: learning rate for the optimizer
        """
        self.weights = [
            W - self.learning_rate * dWi for W, dWi in zip(self.weights, dW)
        ]
        self.biases = [b - self.learning_rate * dbi for b, dbi in zip(self.biases, db)]

    def train_network(self):
        """
        Train the neural network.
        """
        losses = []
        num_samples = self.X_train.shape[0]
        for epoch in range(self.epochs):
            epoch_loss = 0

            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            X_shuffled = self.X_train[indices]
            Z_shuffled = self.Z_train[indices]

            for start_idx in range(0, num_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_samples)
                x_batch = X_shuffled[start_idx:end_idx].T
                z_batch = Z_shuffled[start_idx:end_idx].T

                # Perform back propagation
                dW, db = self.back_prop(x_batch, z_batch)

                self.update_params(dW, db)

                y_pred, _, _ = self.feed_forward(x_batch)
                epoch_loss += self.mse(z_batch, y_pred)

            epoch_loss /= num_samples / self.batch_size
            losses.append(epoch_loss)
            # print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.6f}")
        return losses

    def predict(self, X):
        predictions = []
        num_samples = X.shape[0]

        for start_idx in range(0, num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_samples)
            x_batch = X[start_idx:end_idx].T  # Shape (num_features, batch_size)

            # Get predictions for the mini-batch
            batch_predictions, _, _ = self.feed_forward(x_batch)
            predictions.append(
                batch_predictions.T
            )  # Transpose to match the original shape

        return np.vstack(predictions)


def eval_ffnn():
    # Data initialization.
    X_train, X_test, z_train, z_test, scaler_Z = generate_data()

    layer_sizes = [2, 50, 1]
    epochs = 1000
    mini_batch_size = 10
    learning_rates = [0.00001, 0.0001, 0.001, 0.01]

    lmbdas = [0.0, 0.001, 0.1, 1.0]
    best_mse_test = float("inf")
    best_r2_test = float("-inf")
    best_params = {}
    activations = ["sigmoid", "relu", "leaky_relu"]
    results = []
    for ac in activations:
        print(f"Activation function: {ac}")
        for lr in learning_rates:
            for lmb in lmbdas:
                ffnn = FeedForwardNeuralNetwork(
                    X_train,
                    z_train,
                    layer_sizes,
                    epochs,
                    lr,
                    ac,
                    mini_batch_size,
                    lmb,
                )
                losses = ffnn.train_network()
                z_pred_train = ffnn.predict(X_train)
                z_pred_test = ffnn.predict(X_test)

                mse_train = mean_squared_error(z_train, z_pred_train)
                mse_test = mean_squared_error(z_test, z_pred_test)
                r2_train = r2_score(z_train, z_pred_train)
                r2_test = r2_score(z_test, z_pred_test)

                results.append(
                    {
                        "learning_rate": lr,
                        "lmbda": lmb,
                        "mse_train": mse_train,
                        "mse_test": mse_test,
                        "r2_train": r2_train,
                        "r2_test": r2_test,
                        "activation": ac,
                    }
                )

                if mse_test < best_mse_test:
                    best_mse_test = mse_test
                    best_r2_test = r2_test
                    best_params = {"learning_rate": lr, "lmbda": lmb, "activation": ac}

                print(
                    f"lr: {lr}, lambda: {lmb}, mse_train: {mse_train}, mse_test: {mse_test}"
                )

    print(f"Best MSE test: {best_mse_test}, Best R2 test: {best_r2_test}")
    print(
        f"Best parameters: {best_params['learning_rate']}, {best_params['lmbda']}, {best_params['activation']}"
    )

    results_df = pd.DataFrame(results)

    # 1. Heatmap of MSE Test Scores
    plt.figure(figsize=(16, 8))
    for ac in activations:
        plt.subplot(1, len(activations), activations.index(ac) + 1)  # Create subplots
        subset = results_df[results_df["activation"] == ac]
        pivot_mse = subset.pivot_table(
            values="mse_test", index="lmbda", columns="learning_rate"
        )

        sns.heatmap(
            pivot_mse, annot=True, cmap="viridis", fmt=".3f", cbar_kws={"label": "MSE"}
        )
        plt.title(f"MSE Test Scores - Activation: {ac}")
        plt.xlabel("Learning Rate")
        plt.ylabel("Lambda (Regularization)")
    plt.tight_layout()
    save_plot("ffnn_mse_heatmap_various_activations")
    plt.show()

    return results, best_params


def generate_data():
    np.random.seed(42)
    n = 100
    x = np.random.rand(n)
    y = np.random.rand(n)
    noise = np.random.randn(n)
    Z = franke_function(x, y) + noise

    X = np.column_stack((x, y))  # Combine x and y into a feature matrix
    X_train, X_test, z_train, z_test = train_test_split(
        X, Z, test_size=0.2, random_state=42
    )

    # scale.
    scaler_X = StandardScaler()
    scaler_Z = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    z_train = scaler_Z.fit_transform(z_train.reshape(-1, 1)).ravel()
    z_test = scaler_Z.transform(z_test.reshape(-1, 1)).ravel()

    return X_train, X_test, z_train, z_test, scaler_Z


if __name__ == "__main__":

    # compare
    X_train, X_test, z_train, z_test, scaler_Z = generate_data()
    ols_res = ols_regression()
    ridge_res = ridge_regression()
    ffnn_res = eval_ffnn()

    print(f"OLS: {ols_res}")
    print(f"Ridge: {ridge_res}")
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
