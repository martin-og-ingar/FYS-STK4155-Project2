"""
Your aim now, and this is the central part of this project, is to write your own Feed Forward Neural Network code implementing the back propagation algorithm discussed in the 
lecture slides from week 41 and week 42.

We will focus on a regression problem first and study either the simple second-order polynomial from part a) or the Franke function or terrain data (or both or other data sets) 
from project 1.

Discuss again your choice of cost function.

Write an FFNN code for regression with a flexible number of hidden layers and nodes using the Sigmoid function as activation function for the hidden layers. 
Initialize the weights using a normal distribution. 
How would you initialize the biases? And which activation function would you select for the final output layer?

Train your network and compare the results with those from your OLS and Ridge Regression codes from project 1 if you use the Franke function or the terrain data. 
You should test your results against a similar code using Scikit-Learn (see the examples in the above lecture notes from weeks 41 and 42) or 
tensorflow/keras or Pytorch (for Pytorch, see Raschka et al.’s text chapters 12 and 13).

Comment your results and give a critical discussion of the results obtained with the Linear Regression code and your own Neural Network code.
Make an analysis of the regularization parameters and the learning rates employed to find the optimal MSE and 
 scores.

A useful reference on the back progagation algorithm is Nielsen’s book at http://neuralnetworksanddeeplearning.com/. It is an excellent read.
"""

import os, sys

from sklearn.neural_network import MLPRegressor


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from methods import franke_function, save_plot
from project1_methods import ols_regression, ridge_regression
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class FeedForwardNeuralNetwork:
    def __init__(self, layer_sizes):
        """
        Initialising the neural network.
        :param layer_sizes: list of integers, where each integer represents the number of nodes in the layer.
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        self.weights = []
        self.biases = []

        # create W and b for each layer.
        for i in range(1, self.num_layers):
            W = np.random.rand(layer_sizes[i], layer_sizes[i - 1])

            # Can be initialized to a small random value(normal distribution) or zeros.
            # initializing to zeros in this case (common practice)
            b = np.zeros((layer_sizes[i], 1))

            self.weights.append(W)
            self.biases.append(b)

    def sigmoid(self, z):
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
            a = self.sigmoid(z)
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

        # gradient for output layer
        delta = self.mse_derivative(y_true, y_pred)

        # gradient for hidden layers.
        dW[-1] = np.dot(delta, activations[-2].T)  # Gradient final layer weights
        db[-1] = delta  # Gradient final layer biases

        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l + 1].T, delta) * self.sigmoid_derivative(z)
            dW[-l] = np.dot(delta, activations[-l - 1].T)
            db[-l] = delta
        return dW, db

    def update_params(self, dW, db, learning_rate):
        """
        Update the weights and biases of the network.
        :param dW: gradients for the weights
        :param db: gradients for the biases
        :param learning_rate: learning rate for the optimizer
        """
        self.weights = [W - learning_rate * dWi for W, dWi in zip(self.weights, dW)]
        self.biases = [b - learning_rate * dbi for b, dbi in zip(self.biases, db)]

    def train_network(self, X, Y, epochs, learning_rate):
        """
        Train the neural network.
        :param X: input data of shape (num_samples, input_size)
        :param Y: true labels of shape (num_samples, output_size)
        :param epochs: number of epochs to train the network
        :param learning_rate: learning rate for the optimizer
        """
        for epoch in range(epochs):
            epoch_loss = 0

            for x, y_true in zip(X, Y):
                x = x.reshape(-1, 1)
                y_true = y_true.reshape(-1, 1)

                # Perform back propagation
                dW, db = self.back_prop(x, y_true)

                self.update_params(dW, db, learning_rate)

                y_pred, _, _ = self.feed_forward(x)
                epoch_loss += self.mse(y_true, y_pred)

            print(f"Epoch {epoch}, Loss: {epoch_loss}")
        return y_pred.flatten()


def plot_ffnn(ffnn, X_test, Y_test, scaler_Y):
    Y_pred_scaled = np.array(
        [ffnn.feed_forward(xi.reshape(-1, 1))[0] for xi in X_test]
    ).flatten()

    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled.reshape(-1, 1)).flatten()

    mse_test = ffnn.mse(Y_test, Y_pred)
    print(f"Test MSE: {mse_test}")
    plt.figure(figsize=(10, 6))
    plt.scatter(Y_test, Y_pred, label="Predicted Data")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True vs Predicted Values")
    plt.plot(
        [Y_test.min(), Y_test.max()],
        [Y_test.min(), Y_test.max()],
        "k--",
        lw=2,
        label="Perfect Prediction",
    )
    plt.legend()
    save_plot("true_vs_predicted.png")
    plt.show()


def eval_ffnn(X_train, X_test, z_train, z_test, epochs, learning_rate):
    layer_sizes = [2, 20, 1]
    ffnn = FeedForwardNeuralNetwork(layer_sizes)
    ffnn.train_network(X_train, z_train.reshape(-1, 1), epochs, learning_rate)

    y_pred_train, _, _ = ffnn.feed_forward(X_train.T)

    y_pred_test, _, _ = ffnn.feed_forward(X_test.T)

    mse_train = mean_squared_error(z_train.reshape(-1, 1), y_pred_train.T)
    mse_test = mean_squared_error(z_test.reshape(-1, 1), y_pred_test.T)
    return mse_train, mse_test


if __name__ == "__main__":
    np.random.seed(42)
    n = 100
    x = np.random.rand(n)
    y = np.random.rand(n)
    X = np.c_[x, y]
    Z = franke_function(x, y).astype(np.float64)

    # Split the data.
    X_train, X_test, z_train, z_test = train_test_split(X, Z, test_size=0.2)

    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    Z_train_scaled = scaler_Y.fit_transform(z_train.reshape(-1, 1))
    Z_test_scaled = scaler_Y.transform(z_test.reshape(-1, 1))

    epochs = 100
    learning_rate = 0.01

    # compare
    ols_res = ols_regression()
    ridge_res = ridge_regression()

    ffnn_res = eval_ffnn(
        X_train_scaled,
        X_test_scaled,
        Z_train_scaled,
        Z_test_scaled,
        epochs,
        learning_rate,
    )

    print(f"OLS: {ols_res}")
    print(f"Ridge: {ridge_res}")
    print(f"FFNN: {ffnn_res}")

    # Plot the fit of the model on the Franke function
    # x_grid = np.linspace(0, 1, 100)
    # y_grid = np.linspace(0, 1, 100)
    # x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    # z_mesh_true = franke_function(x_mesh, y_mesh)

    # X_mesh = np.c_[x_mesh.ravel(), y_mesh.ravel()]
    # X_mesh_scaled = scaler_X.transform(X_mesh)
    # z_mesh_pred_scaled = np.array(
    #     [ffnn.feed_forward(xi.reshape(-1, 1))[0] for xi in X_mesh_scaled]
    # ).flatten()
    # z_mesh_pred = scaler_Y.inverse_transform(
    #     z_mesh_pred_scaled.reshape(-1, 1)
    # ).flatten()
    # z_mesh_pred = z_mesh_pred.reshape(x_mesh.shape)

    # fig = plt.figure(figsize=(14, 6))

    # ax1 = fig.add_subplot(121, projection="3d")
    # ax1.plot_surface(x_mesh, y_mesh, z_mesh_true, cmap="viridis")
    # ax1.set_title("True Franke Function")

    # ax2 = fig.add_subplot(122, projection="3d")
    # ax2.plot_surface(x_mesh, y_mesh, z_mesh_pred, cmap="viridis")
    # ax2.set_title("Predicted Franke Function")

    # save_plot("ffnn_franke_function")
    # plt.show()

    # testing
    mlp = MLPRegressor(
        hidden_layer_sizes=(10,), activation="logistic", max_iter=1000, solver="adam"
    )

    mlp.fit(X_train_scaled, Z_train_scaled)
    y_pred_train_sklearn = mlp.predict(X_train)
    y_pred_test_sklearn = mlp.predict(X_test)

    # MSE for Scikit-Learn FFNN
    mse_train_sklearn = mean_squared_error(z_train, y_pred_train_sklearn)
    mse_test_sklearn = mean_squared_error(z_test, y_pred_test_sklearn)
    print(
        f"Scikit-Learn FFNN: Train MSE = {mse_train_sklearn}, Test MSE = {mse_test_sklearn}"
    )
