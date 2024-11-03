import autograd.numpy as np
from sklearn.model_selection import train_test_split
from autograd import grad
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from partA.gradient_methods import learning_schedule


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


class Logistic:

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def predict(self, X):
        probabilities = sigmoid(X @ self.beta)
        return (probabilities >= 0.5).astype(int)

    def cost_function(self, beta, X, y):
        y_pred = sigmoid(X @ beta)
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def sgd(self, X_train, y_train, epochs, mb_size, tolerance=1e-6):
        X_train, X_validate, y_train, y_validate = train_test_split(
            X_train, y_train, test_size=0.2
        )
        training_gradient = grad(self.cost_function)
        m, n = X_train.shape
        iterations = int(m / mb_size)
        self.beta = np.zeros((n, 1))
        prev_score = float("inf")

        for epoch in range(epochs):
            for i in range(iterations):
                index = mb_size * np.random.randint(iterations)
                Xi = (
                    X_train[index : index + mb_size]
                    if index + mb_size <= m
                    else X_train[index:m]
                )
                yi = (
                    y_train[index : index + mb_size]
                    if index + mb_size <= m
                    else y_train[index:m]
                )
                gradient = (1 / mb_size) * training_gradient(self.beta, Xi, yi)
                learning_rate = learning_schedule(epoch * iterations + i)
                self.beta -= learning_rate * gradient

                self.training_score = self.cost_function(
                    self.beta, X_validate, y_validate
                )

                if abs(self.training_score - prev_score) < tolerance:
                    return
                prev_score = self.training_score

    def reset_beta(self):
        self.beta = np.zeros_like(self.beta)
