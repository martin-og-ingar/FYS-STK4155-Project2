"""
This code is developed for project 2 in FYS-STK4155 at the University of Oslo.
It contain subtask b, c and d of the project.
"""

import os, sys

from sklearn.neural_network import MLPRegressor


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
from partB_to_D.methods import (
    ols_regression,
    ridge_regression,
    franke_function,
    save_plot,
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score


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
        mode="regression",
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
        self.mode = mode

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

    def activate_output(self, z):
        if self.mode == "classification":
            return self.sigmoid(z)
        elif self.mode == "regression":
            return z

    def activate_hidden(self, z):
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

    def cost_classification(self, y_true, y_pred):
        epsilon = 1e-12  # A small value to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

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

        :param x: input data of shape (num_features, batch_size)
        :return: output of the network
        """
        a = x  # activation function.
        activations = []
        z_values = []

        # loop through all layers except the last one.

        # This is becuase, for regression tasks, the final output layer should typically use linear activation function.
        # since we are predicting real-valued numbers.
        #
        for i in range(self.num_layers - 1):
            W, b = self.weights[i], self.biases[i]
            z = np.dot(W, a) + b
            z_values.append(z)

            if i < len(self.weights) - 2:
                a = self.activate_hidden(z)
            else:
                a = self.activate_output(z)

            activations.append(a)

        return a, activations, z_values

    def back_prop(self, x, y_true, current_batch_size):
        """
        Perform back propagation through the network.

        :param x: input data of shape (input_size, batch_size)
        :param y_true: true labels of shape (output_size, batch_size)
        """
        y_pred, activations, zs = self.feed_forward(x)

        # Init gradients.
        dW = [np.zeros(W.shape) for W in self.weights]
        db = [np.zeros(b.shape) for b in self.biases]
        m = current_batch_size  # number of samples.

        if self.mode == "classification":
            delta = (y_pred - y_true) / m
        else:
            delta = y_pred - y_true

        dW_final = np.dot(delta, activations[-2].T)
        db_final = np.sum(delta, axis=1, keepdims=True)

        dW[-1] = dW_final + (self.lmb / m) * self.weights[-1]
        db[-1] = db_final

        for l in range(self.num_layers - 3, -1, -1):
            delta = np.dot(self.weights[l + 1].T, delta) * self.sigmoid_derivative(
                zs[l]
            )

            dW[l] = np.dot(delta, activations[l - 1].T)
            db[l] = np.sum(delta, axis=1, keepdims=True)
        return dW, db

    def update_params(self, dW, db):
        """
        Update the weights and biases of the network.
        :param dW: gradients for the weights
        :param db: gradients for the biases
        :param learning_rate: learning rate for the optimizer
        """
        m = len(dW[0])
        self.weights = [
            W - self.learning_rate * (dWi / m) for W, dWi in zip(self.weights, dW)
        ]
        self.biases = [
            b - self.learning_rate * (dbi / m) for b, dbi in zip(self.biases, db)
        ]

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

                if self.mode == "classification":
                    m = z_batch.shape[1]  # actual batch size.
                else:
                    m = x_batch.shape[1]
                y_pred, _, _ = self.feed_forward(x_batch)

                # Perform back propagation

                if self.mode == "classification":
                    epoch_loss += self.cost_classification(z_batch, y_pred)
                else:
                    epoch_loss += self.mse(z_batch, y_pred)

                dW, db = self.back_prop(x_batch, z_batch, m)
                self.update_params(dW, db)

            epoch_loss /= num_samples / self.batch_size
            losses.append(epoch_loss)
            # print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.6f}")
        return losses

    def predict(self, X):
        # Transpose the input to match the expected shape
        x_batch = X.T  # Shape (n_features, num_samples)

        output, _, _ = self.feed_forward(x_batch)

        if self.mode == "classification":
            # For binary classification, output will be a single value
            # Apply thresholding to convert probabilities to class labels
            class_labels = (output > 0.5).astype(
                int
            )  # Convert probabilities to class labels (0 or 1)
            return class_labels.flatten()  # Flatten to return a 1D array

        elif self.mode == "regression":
            # For regression, output can be returned directly
            return output.flatten()  # Flatten to return a 1D array of predictions

        else:
            raise ValueError(
                "Unsupported mode. Choose either 'classification' or 'regression'."
            )


def eval_ffnn():
    # Data initialization.
    X_train, X_test, z_train, z_test, scaler_Z = generate_data()

    layer_sizes = [2, 50, 1]
    epochs = 100
    mini_batch_size = 10
    mode = "regression"
    learning_rates = [0.0001, 0.001, 0.01, 0.1]

    lmbdas = [0.0, 0.001, 0.01, 0.1]
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
                    mode,
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


def eval_classification_ffnn():
    from sklearn.metrics import accuracy_score, confusion_matrix

    # malignant[0], benign[1], positive indicates cancer.

    data = load_breast_cancer()

    X = data.data
    y = data.target.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    layer_sizes = [30, 32, 16, 1]
    learning_rates = [0.0001, 0.001, 0.01, 0.1]

    epochs = [100, 200, 500, 1000]

    mini_batch_size = 10
    accuracy_matrix = np.zeros((len(epochs), len(learning_rates)))

    for i, epoch in enumerate(epochs):

        for j, lr in enumerate(learning_rates):

            nn = FeedForwardNeuralNetwork(
                X_train,
                y_train,
                layer_sizes,
                epoch,
                lr,
                "sigmoid",
                mini_batch_size,
                0.1,
                mode="classification",
            )
            nn.train_network()

            pred = nn.predict(X_test)

            accuracy = accuracy_score(y_test, pred)
            accuracy_matrix[i, j] = accuracy
            cm = confusion_matrix(y_test, pred)

            print(f"Accuracy: {accuracy:.4f}, Learning rate: {lr}, Epochs: {epoch}")
            print("Confusion Matrix:")
            print(cm)
            print("-------------------------------------------------")
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        accuracy_matrix,
        annot=True,
        xticklabels=learning_rates,
        yticklabels=epochs,
        cmap="YlGnBu",
        cbar_kws={"label": "Accuracy"},
    )
    plt.xlabel("Learning Rate")
    plt.ylabel("Epochs")
    plt.title("Accuracy Heatmap for Different Learning Rates and Epochs")
    save_plot("classification_accuracy_ffnn_heatmap")
    plt.show()


def test_classification_ffnn(X_train, y_train, X_test, y_test, lmb):
    layer_sizes = [30, 32, 16, 1]

    epochs = 1000
    mini_batch_size = 10
    nn = FeedForwardNeuralNetwork(
        X_train,
        y_train,
        layer_sizes,
        epochs,
        lmb,
        "sigmoid",
        mini_batch_size,
        mode="classification",
    )
    nn.train_network()

    pred = nn.predict(X_test)
    y_true = y_test

    return accuracy_score(y_true, pred)


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


def plot_ols(res):
    plt.figure(figsize=(12, 5))

    # Plot MSE
    plt.subplot(1, 2, 1)
    plt.plot(res["Degree"], res["Train MSE"], marker="o", label="Train MSE")
    plt.plot(res["Degree"], res["Test MSE"], marker="o", label="Test MSE")
    plt.title("MSE vs. Polynomial Degree")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid()

    # Plot R²
    plt.subplot(1, 2, 2)
    plt.plot(res["Degree"], res["Train R²"], marker="o", label="Train R²")
    plt.plot(res["Degree"], res["Test R²"], marker="o", label="Test R²")
    plt.title("R² vs. Polynomial Degree")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("R² Score")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    save_plot("ols_mse_r2")
    plt.show()


def plot_ridge(result):
    lambda_to_plot = 0.1  # Choose the lambda value you want to visualize

    plt.figure(figsize=(12, 5))

    # Plot MSE
    plt.subplot(1, 2, 1)
    plt.plot(
        ridge_res[lambda_to_plot]["Degree"],
        ridge_res[lambda_to_plot]["Train MSE"],
        marker="o",
        label="Train MSE",
    )
    plt.plot(
        ridge_res[lambda_to_plot]["Degree"],
        ridge_res[lambda_to_plot]["Test MSE"],
        marker="o",
        label="Test MSE",
    )
    plt.title(f"MSE vs. Polynomial Degree (Lambda = {lambda_to_plot})")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid()

    # Plot R²
    plt.subplot(1, 2, 2)
    plt.plot(
        ridge_res[lambda_to_plot]["Degree"],
        ridge_res[lambda_to_plot]["Train R²"],
        marker="o",
        label="Train R²",
    )
    plt.plot(
        ridge_res[lambda_to_plot]["Degree"],
        ridge_res[lambda_to_plot]["Test R²"],
        marker="o",
        label="Test R²",
    )
    plt.title(f"R² vs. Polynomial Degree (Lambda = {lambda_to_plot})")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("R² Score")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    save_plot("ridge_mse_r2")
    plt.show()


if __name__ == "__main__":

    input = sys.argv[1]
    print(input)
    if input == "reg":
        ols_res = ols_regression()
        plot_ols(ols_res)

        ridge_res = ridge_regression()
        plot_ridge(ridge_res)

        ffnn_res = eval_ffnn()

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
        grid_search = GridSearchCV(
            mlp, param_grid, scoring="neg_mean_squared_error", cv=5
        )
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
    elif input == "class":
        eval_classification_ffnn()
        print("Classification task completed.")
