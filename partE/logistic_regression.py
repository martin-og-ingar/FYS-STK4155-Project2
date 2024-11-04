import autograd.numpy as np
from sklearn.model_selection import train_test_split
from autograd import grad
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from partA.gradient_methods import learning_schedule


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


class Logistic:
    def __init__(self, epochs, mb_size, lmb):
        self.epochs = epochs
        self.mb_size = mb_size
        self.lmb = lmb
        self.tolerance = 1e-6

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def predict(self, X):
        probabilities = sigmoid(X @ self.beta)
        return (probabilities >= 0.5).astype(int)

    def cost_function(self, beta, X, y):
        y_pred = sigmoid(X @ beta)
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        cross_e_loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        l2_penalty = (self.lmb / 2) * np.sum(beta**2)

        return cross_e_loss + l2_penalty

    def sgd(self, X_train, y_train):
        X_train, X_validate, y_train, y_validate = train_test_split(
            X_train, y_train, test_size=0.2
        )
        training_gradient = grad(self.cost_function)
        m, n = X_train.shape
        iterations = int(m / self.mb_size)
        self.beta = np.zeros((n, 1))
        prev_score = float("inf")

        for epoch in range(self.epochs):
            for i in range(iterations):
                index = self.mb_size * np.random.randint(iterations)
                Xi = (
                    X_train[index : index + self.mb_size]
                    if index + self.mb_size <= m
                    else X_train[index:m]
                )
                yi = (
                    y_train[index : index + self.mb_size]
                    if index + self.mb_size <= m
                    else y_train[index:m]
                )
                gradient = (1 / self.mb_size) * training_gradient(self.beta, Xi, yi)
                learning_rate = learning_schedule(epoch * iterations + i)
                self.beta -= learning_rate * gradient

                self.training_score = self.score(X_validate, y_validate)

                if abs(self.training_score - prev_score) < self.tolerance:
                    return
                prev_score = self.training_score

    def reset_beta(self):
        self.beta = np.zeros_like(self.beta)
