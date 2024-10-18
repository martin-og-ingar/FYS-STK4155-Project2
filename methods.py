import numpy as np
from autograd import grad
import autograd.numpy as anp


def gradient_descent_ols(X, y, learning_rate, iterations, tolerance=1e-6):
    m, n = X.shape
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    for i in range(iterations):
        y_pred = X @ beta
        gradient = (1 / m) * X.T @ (y_pred - y)
        beta -= learning_rate * gradient

        mse = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            print(f"Converged after {i} iterations")
            return beta, i
        prev_loss = mse

    return beta, iterations


def gradient_descent_ridge(X, y, lmd, learning_rate, iterations, tolerance=1e-6):
    m, n = X.shape
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    for i in range(iterations):
        y_pred = X @ beta
        gradient = (1 / m) * (X.T @ (y_pred - y) + lmd * beta)
        beta -= learning_rate * gradient

        mse = (1 / (2 * m)) * np.sum((y_pred - y) ** 2) + (lmd / (2 * m)) * np.sum(
            beta**2
        )

        if abs(prev_loss - mse) < tolerance:
            return beta, i
        prev_loss = mse

    return beta, iterations


def gradient_descent_momentum_ols(
    X, y, learning_rate, iterations, momentum, tolerance=1e-6
):
    m, n = X.shape
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    velocity = np.zeros_like(beta)

    for i in range(iterations):
        y_pred = X @ beta
        gradient = (1 / m) * X.T @ (y_pred - y)
        velocity = momentum * velocity - learning_rate * gradient
        beta += velocity

        mse = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i
        prev_loss = mse

    return beta, iterations


def gradient_descent_momentum_ridge(
    X, y, lmd, learning_rate, iterations, momentum, tolerance=1e-6
):
    m, n = X.shape
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    velocity = np.zeros_like(beta)

    for i in range(iterations):
        y_pred = X @ beta
        gradient = (1 / m) * (X.T @ (y_pred - y) + lmd * beta)
        velocity = momentum * velocity - learning_rate * gradient
        beta += velocity

        mse = (1 / (2 * m)) * np.sum((y_pred - y) ** 2) + (lmd / (2 * m)) * np.sum(
            beta**2
        )

        if abs(prev_loss - mse) < tolerance:
            return beta, i
        prev_loss = mse

    return beta, iterations


def sgd_ols(X, y, learning_rate, epochs, mb_size, tolerance=1e-6):
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")

    for epoch in range(epochs):
        for i in range(iterations):
            index = mb_size * np.random.randint(iterations)
            Xi = X[index : index + mb_size] if index + mb_size <= m else X[index:m]
            yi = y[index : index + mb_size] if index + mb_size <= m else y[index:m]
            y_pred = Xi @ beta
            gradient = (1 / mb_size) * Xi.T @ (y_pred - yi)
            beta -= learning_rate * gradient

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2)

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch
            prev_loss = mse

    return beta, iterations, epochs


def sgd_ridge(X, y, lmd, learning_rate, epochs, mb_size, tolerance=1e-6):
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")

    for epoch in range(epochs):
        for i in range(iterations):
            index = mb_size * np.random.randint(iterations)
            Xi = X[index : index + mb_size] if index + mb_size <= m else X[index:m]
            yi = y[index : index + mb_size] if index + mb_size <= m else y[index:m]
            y_pred = Xi @ beta
            gradient = (1 / mb_size) * (Xi.T @ (y_pred - yi) + lmd * beta)
            beta -= learning_rate * gradient

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2) + (lmd / (2 * m)) * np.sum(
                beta**2
            )

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch
            prev_loss = mse

    return beta, iterations, epochs


def sgd_momentum_ols(X, y, learning_rate, epochs, mb_size, momentum, tolerance=1e-6):
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    velocity = np.zeros_like(beta)

    for epoch in range(epochs):
        for i in range(iterations):
            index = mb_size * np.random.randint(iterations)
            Xi = X[index : index + mb_size] if index + mb_size <= m else X[index:m]
            yi = y[index : index + mb_size] if index + mb_size <= m else y[index:m]
            y_pred = Xi @ beta
            gradient = (1 / mb_size) * Xi.T @ (y_pred - yi)
            velocity = momentum * velocity - learning_rate * gradient
            beta += velocity

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2)

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch
            prev_loss = mse

    return beta, iterations, epochs


def sgd_momentum_ridge(
    X, y, lmd, learning_rate, epochs, mb_size, momentum, tolerance=1e-6
):
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    velocity = np.zeros_like(beta)

    for epoch in range(epochs):
        for i in range(iterations):
            index = mb_size * np.random.randint(iterations)
            Xi = X[index : index + mb_size] if index + mb_size <= m else X[index:m]
            yi = y[index : index + mb_size] if index + mb_size <= m else y[index:m]
            y_pred = Xi @ beta
            gradient = (1 / mb_size) * (Xi.T @ (y_pred - yi) + lmd * beta)
            velocity = momentum * velocity - learning_rate * gradient
            beta += velocity

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2) + (lmd / (2 * m)) * np.sum(
                beta**2
            )

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch
            prev_loss = mse

    return beta, iterations, epochs


def CostOLS(beta, X, y):
    return anp.sum((y - X @ beta) ** 2)


def CostRidge(beta, X, y, lmd):
    m = X.shape[0]
    return anp.sum((y - X @ beta) ** 2) / m + (lmd / (2 * m)) * anp.sum(beta**2)


def adagrad_pgd_ols(X, y, learning_rate, iterations, tolerance=1e-6):
    training_gradient = grad(CostOLS)
    m, n = X.shape
    beta = anp.zeros((n, 1))
    prev_loss = float("inf")
    Giter = 0.0
    rho = 0.99
    delta = 1e-8

    for i in range(iterations):
        y_pred = X @ beta
        gradients = (1.0 / m) * training_gradient(beta, X, y)
        Giter = rho * Giter + (1 - rho) * gradients * gradients
        update = gradients * learning_rate / (delta + anp.sqrt(Giter))
        beta -= update

        mse = (1 / (2 * m)) * anp.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i
        prev_loss = mse

    return beta, iterations


def adagrad_pgd_ridge(X, y, lmd, learning_rate, iterations, tolerance=1e-6):
    training_gradient = grad(CostRidge)
    m, n = X.shape
    beta = anp.zeros((n, 1))
    prev_loss = float("inf")
    Giter = 0.0
    rho = 0.99
    delta = 1e-8

    for i in range(iterations):
        y_pred = X @ beta
        gradients = (1.0 / m) * training_gradient(beta, X, y, lmd)
        Giter = rho * Giter + (1 - rho) * gradients * gradients
        update = gradients * learning_rate / (delta + anp.sqrt(Giter))
        beta -= update

        mse = (1 / (2 * m)) * anp.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i
        prev_loss = mse

    return beta, iterations


def adagrad_gdm_ols(X, y, learning_rate, iterations, momentum, tolerance=1e-6):
    training_gradient = grad(CostOLS)
    m, n = X.shape
    beta = anp.zeros((n, 1))
    prev_loss = float("inf")
    Giter = 0.0
    rho = 0.99
    delta = 1e-8
    prev_beta = anp.zeros_like(beta)

    for i in range(iterations):
        y_pred = X @ beta
        gradients = (1.0 / m) * training_gradient(beta, X, y)
        Giter = rho * Giter + (1 - rho) * gradients * gradients
        update = gradients * learning_rate / (delta + anp.sqrt(Giter))
        prev_beta = momentum * prev_beta + update
        beta -= prev_beta

        mse = (1 / (2 * m)) * anp.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i
        prev_loss = mse

    return beta, iterations


def adagrad_gdm_ridge(X, y, lmd, learning_rate, iterations, momentum, tolerance=1e-6):
    training_gradient = grad(CostRidge)
    m, n = X.shape
    beta = anp.zeros((n, 1))
    prev_loss = float("inf")
    Giter = 0.0
    rho = 0.99
    delta = 1e-8
    prev_beta = anp.zeros_like(beta)

    for i in range(iterations):
        y_pred = X @ beta
        gradients = (1.0 / m) * training_gradient(beta, X, y, lmd)
        Giter = rho * Giter + (1 - rho) * gradients * gradients
        update = gradients * learning_rate / (delta + anp.sqrt(Giter))
        prev_beta = momentum * prev_beta + update
        beta -= prev_beta

        mse = (1 / (2 * m)) * anp.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i
        prev_loss = mse

    return beta, iterations


def adagrad_sgd_ols(X, y, learning_rate, epochs, mb_size, tolerance=1e-6):
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    training_gradient = grad(CostOLS)
    Giter = 0.0
    rho = 0.99
    delta = 1e-8

    for epoch in range(epochs):
        for i in range(iterations):
            index = mb_size * np.random.randint(iterations)
            Xi = X[index : index + mb_size] if index + mb_size <= m else X[index:m]
            yi = y[index : index + mb_size] if index + mb_size <= m else y[index:m]
            y_pred = Xi @ beta
            gradients = (1.0 / m) * training_gradient(beta, X, y)
            Giter = rho * Giter + (1 - rho) * gradients * gradients
            update = gradients * learning_rate / (delta + anp.sqrt(Giter))
            beta -= update

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2)

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch
            prev_loss = mse

    return beta, iterations, epochs


def adagrad_sgd_ridge(X, y, lmd, learning_rate, epochs, mb_size, tolerance=1e-6):
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    training_gradient = grad(CostRidge)
    Giter = 0.0
    rho = 0.99
    delta = 1e-8
    for epoch in range(epochs):
        for i in range(iterations):
            index = mb_size * np.random.randint(iterations)
            Xi = X[index : index + mb_size] if index + mb_size <= m else X[index:m]
            yi = y[index : index + mb_size] if index + mb_size <= m else y[index:m]
            y_pred = Xi @ beta
            gradients = (1.0 / m) * training_gradient(beta, X, y, lmd)
            Giter = rho * Giter + (1 - rho) * gradients * gradients
            update = gradients * learning_rate / (delta + anp.sqrt(Giter))
            beta -= update

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2) + (lmd / (2 * m)) * np.sum(
                beta**2
            )

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch
            prev_loss = mse

    return beta, iterations, epochs


def adagrad_sgd_momentum_ols(
    X, y, learning_rate, epochs, mb_size, momentum, tolerance=1e-6
):
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    training_gradient = grad(CostOLS)
    Giter = 0.0
    rho = 0.99
    delta = 1e-8
    prev_beta = anp.zeros_like(beta)

    for epoch in range(epochs):
        for i in range(iterations):
            index = mb_size * np.random.randint(iterations)
            Xi = X[index : index + mb_size] if index + mb_size <= m else X[index:m]
            yi = y[index : index + mb_size] if index + mb_size <= m else y[index:m]
            y_pred = Xi @ beta
            gradients = (1.0 / m) * training_gradient(beta, X, y)
            Giter = rho * Giter + (1 - rho) * gradients * gradients
            update = gradients * learning_rate / (delta + anp.sqrt(Giter))
            prev_beta = momentum * prev_beta + update
            beta -= prev_beta

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2)

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch
            prev_loss = mse

    return beta, iterations, epochs


def adagrad_sgd_momentum_ridge(
    X, y, lmd, learning_rate, epochs, mb_size, momentum, tolerance=1e-6
):
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    training_gradient = grad(CostRidge)
    Giter = 0.0
    rho = 0.99
    delta = 1e-8
    prev_beta = anp.zeros_like(beta)

    for epoch in range(epochs):
        for i in range(iterations):
            index = mb_size * np.random.randint(iterations)
            Xi = X[index : index + mb_size] if index + mb_size <= m else X[index:m]
            yi = y[index : index + mb_size] if index + mb_size <= m else y[index:m]
            y_pred = Xi @ beta
            gradients = (1.0 / m) * training_gradient(beta, X, y, lmd)
            Giter = rho * Giter + (1 - rho) * gradients * gradients
            update = gradients * learning_rate / (delta + anp.sqrt(Giter))
            prev_beta = momentum * prev_beta + update
            beta -= prev_beta

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2) + (lmd / (2 * m)) * np.sum(
                beta**2
            )

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch
            prev_loss = mse

    return beta, iterations, epochs


def rms_pgd_ols(X, y, learning_rate, iterations, tolerance=1e-6):
    training_gradient = grad(CostOLS)
    m, n = X.shape
    beta = anp.zeros((n, 1))
    prev_loss = float("inf")
    Giter = 0.0
    rho = 0.99
    delta = 1e-8

    for i in range(iterations):
        y_pred = X @ beta
        gradients = (1.0 / m) * training_gradient(beta, X, y)
        Giter = rho * Giter + (1 - rho) * anp.square(gradients)
        update = gradients * learning_rate / (delta + anp.sqrt(Giter))
        beta -= update

        mse = (1 / (2 * m)) * anp.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i
        prev_loss = mse

    return beta, iterations


def rms_pgd_ridge(X, y, lmd, learning_rate, iterations, tolerance=1e-6):
    training_gradient = grad(CostRidge)
    m, n = X.shape
    beta = anp.zeros((n, 1))
    prev_loss = float("inf")
    Giter = 0.0
    rho = 0.99
    delta = 1e-8

    for i in range(iterations):
        y_pred = X @ beta
        gradients = (1.0 / m) * training_gradient(beta, X, y, lmd)
        Giter = rho * Giter + (1 - rho) * anp.square(gradients)
        update = gradients * learning_rate / (delta + anp.sqrt(Giter))
        beta -= update

        mse = (1 / (2 * m)) * anp.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i
        prev_loss = mse

    return beta, iterations


def rms_gdm_ols(X, y, learning_rate, iterations, momentum, tolerance=1e-6):
    training_gradient = grad(CostOLS)
    m, n = X.shape
    beta = anp.zeros((n, 1))
    prev_loss = float("inf")
    Giter = 0.0
    rho = 0.99
    delta = 1e-8
    prev_beta = anp.zeros_like(beta)

    for i in range(iterations):
        y_pred = X @ beta
        gradients = (1.0 / m) * training_gradient(beta, X, y)
        Giter = rho * Giter + (1 - rho) * anp.square(gradients)
        update = gradients * learning_rate / (delta + anp.sqrt(Giter))
        prev_beta = momentum * prev_beta + update
        beta -= prev_beta

        mse = (1 / (2 * m)) * anp.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i
        prev_loss = mse

    return beta, iterations


def rms_gdm_ridge(X, y, lmd, learning_rate, iterations, momentum, tolerance=1e-6):
    training_gradient = grad(CostRidge)
    m, n = X.shape
    beta = anp.zeros((n, 1))
    prev_loss = float("inf")
    Giter = 0.0
    rho = 0.99
    delta = 1e-8
    prev_beta = anp.zeros_like(beta)

    for i in range(iterations):
        y_pred = X @ beta
        gradients = (1.0 / m) * training_gradient(beta, X, y, lmd)
        Giter = rho * Giter + (1 - rho) * anp.square(gradients)
        update = gradients * learning_rate / (delta + anp.sqrt(Giter))
        prev_beta = momentum * prev_beta + update
        beta -= prev_beta

        mse = (1 / (2 * m)) * anp.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i
        prev_loss = mse

    return beta, iterations


def rms_sgd_ols(X, y, learning_rate, epochs, mb_size, tolerance=1e-6):
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    training_gradient = grad(CostOLS)
    Giter = 0.0
    rho = 0.99
    delta = 1e-8

    for epoch in range(epochs):
        for i in range(iterations):
            index = mb_size * np.random.randint(iterations)
            Xi = X[index : index + mb_size] if index + mb_size <= m else X[index:m]
            yi = y[index : index + mb_size] if index + mb_size <= m else y[index:m]
            y_pred = Xi @ beta
            gradients = (1.0 / m) * training_gradient(beta, X, y)
            Giter = rho * Giter + (1 - rho) * anp.square(gradients)
            update = gradients * learning_rate / (delta + anp.sqrt(Giter))
            beta -= update

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2)

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch
            prev_loss = mse

    return beta, iterations, epochs


def rms_sgd_ridge(X, y, lmd, learning_rate, epochs, mb_size, tolerance=1e-6):
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    training_gradient = grad(CostRidge)
    Giter = 0.0
    rho = 0.99
    delta = 1e-8
    for epoch in range(epochs):
        for i in range(iterations):
            index = mb_size * np.random.randint(iterations)
            Xi = X[index : index + mb_size] if index + mb_size <= m else X[index:m]
            yi = y[index : index + mb_size] if index + mb_size <= m else y[index:m]
            y_pred = Xi @ beta
            gradients = (1.0 / m) * training_gradient(beta, X, y, lmd)
            Giter = rho * Giter + (1 - rho) * anp.square(gradients)
            update = gradients * learning_rate / (delta + anp.sqrt(Giter))
            beta -= update

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2) + (lmd / (2 * m)) * np.sum(
                beta**2
            )

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch
            prev_loss = mse

    return beta, iterations, epochs


def rms_sgd_momentum_ols(
    X, y, learning_rate, epochs, mb_size, momentum, tolerance=1e-6
):
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    training_gradient = grad(CostOLS)
    Giter = 0.0
    rho = 0.99
    delta = 1e-8
    prev_beta = anp.zeros_like(beta)

    for epoch in range(epochs):
        for i in range(iterations):
            index = mb_size * np.random.randint(iterations)
            Xi = X[index : index + mb_size] if index + mb_size <= m else X[index:m]
            yi = y[index : index + mb_size] if index + mb_size <= m else y[index:m]
            y_pred = Xi @ beta
            gradients = (1.0 / m) * training_gradient(beta, X, y)
            Giter = rho * Giter + (1 - rho) * anp.square(gradients)
            update = gradients * learning_rate / (delta + anp.sqrt(Giter))
            prev_beta = momentum * prev_beta + update
            beta -= prev_beta

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2)

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch
            prev_loss = mse

    return beta, iterations, epochs


def rms_sgd_momentum_ridge(
    X, y, lmd, learning_rate, epochs, mb_size, momentum, tolerance=1e-6
):
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    training_gradient = grad(CostRidge)
    Giter = 0.0
    rho = 0.99
    delta = 1e-8
    prev_beta = anp.zeros_like(beta)

    for epoch in range(epochs):
        for i in range(iterations):
            index = mb_size * np.random.randint(iterations)
            Xi = X[index : index + mb_size] if index + mb_size <= m else X[index:m]
            yi = y[index : index + mb_size] if index + mb_size <= m else y[index:m]
            y_pred = Xi @ beta
            gradients = (1.0 / m) * training_gradient(beta, X, y, lmd)
            Giter = rho * Giter + (1 - rho) * anp.square(gradients)
            update = gradients * learning_rate / (delta + anp.sqrt(Giter))
            prev_beta = momentum * prev_beta + update
            beta -= prev_beta

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2) + (lmd / (2 * m)) * np.sum(
                beta**2
            )

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch
            prev_loss = mse

    return beta, iterations, epochs


def adam_pgd_ols(X, y, learning_rate, iterations, tolerance=1e-6):
    training_gradient = grad(CostOLS)
    m, n = X.shape
    beta = anp.zeros((n, 1))
    beta1 = 0.9
    beta2 = 0.999
    prev_loss = float("inf")
    delta = 1e-8
    first_moment = 0.0
    second_moment = 0.0

    for i in range(1, iterations + 1):
        y_pred = X @ beta
        gradients = (1.0 / m) * training_gradient(beta, X, y)
        first_moment = beta1 * first_moment + (1 - beta1) * gradients
        second_moment = beta2 * second_moment + (1 - beta2) * gradients * gradients
        first_term = first_moment / (1.0 - beta1**i)
        second_term = second_moment / (1.0 - beta2**i)
        update = learning_rate * first_term / (np.sqrt(second_term) + delta)
        beta -= update

        mse = (1 / (2 * m)) * anp.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i
        prev_loss = mse

    return beta, iterations


def adam_pgd_ridge(X, y, lmd, learning_rate, iterations, tolerance=1e-6):
    training_gradient = grad(CostRidge)
    m, n = X.shape
    beta = anp.zeros((n, 1))
    prev_loss = float("inf")
    delta = 1e-8
    first_moment = 0.0
    second_moment = 0.0
    beta1 = 0.9
    beta2 = 0.999

    for i in range(1, iterations + 1):
        y_pred = X @ beta
        gradients = (1.0 / m) * training_gradient(beta, X, y, lmd)
        first_moment = beta1 * first_moment + (1 - beta1) * gradients
        second_moment = beta2 * second_moment + (1 - beta2) * gradients * gradients
        first_term = first_moment / (1.0 - beta1**i)
        second_term = second_moment / (1.0 - beta2**i)
        update = learning_rate * first_term / (np.sqrt(second_term) + delta)
        beta -= update

        mse = (1 / (2 * m)) * anp.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i
        prev_loss = mse

    return beta, iterations


def adam_gdm_ols(X, y, learning_rate, iterations, momentum, tolerance=1e-6):
    training_gradient = grad(CostOLS)
    m, n = X.shape
    beta = anp.zeros((n, 1))
    prev_loss = float("inf")
    delta = 1e-8
    prev_beta = anp.zeros_like(beta)
    first_moment = anp.zeros_like(beta)
    second_moment = anp.zeros_like(beta)
    beta1 = 0.9
    beta2 = 0.999

    for i in range(1, iterations + 1):
        y_pred = X @ beta
        gradients = (1.0 / m) * training_gradient(beta, X, y)
        first_moment = beta1 * first_moment + (1 - beta1) * gradients
        second_moment = beta2 * second_moment + (1 - beta2) * gradients * gradients
        first_term = first_moment / (1.0 - beta1**i)
        second_term = second_moment / (1.0 - beta2**i)
        update = learning_rate * first_term / (np.sqrt(second_term) + delta)
        prev_beta = momentum * prev_beta + update
        beta -= prev_beta

        mse = (1 / (2 * m)) * anp.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i
        prev_loss = mse

    return beta, iterations


def adam_gdm_ridge(X, y, lmd, learning_rate, iterations, momentum, tolerance=1e-6):
    training_gradient = grad(CostRidge)
    m, n = X.shape
    beta = anp.zeros((n, 1))
    prev_loss = float("inf")
    delta = 1e-8
    prev_beta = anp.zeros_like(beta)
    first_moment = 0.0
    second_moment = 0.0
    beta1 = 0.9
    beta2 = 0.999

    for i in range(1, iterations + 1):
        y_pred = X @ beta
        gradients = (1.0 / m) * training_gradient(beta, X, y, lmd)
        first_moment = beta1 * first_moment + (1 - beta1) * gradients
        second_moment = beta2 * second_moment + (1 - beta2) * gradients * gradients
        first_term = first_moment / (1.0 - beta1**i)
        second_term = second_moment / (1.0 - beta2**i)
        update = learning_rate * first_term / (np.sqrt(second_term) + delta)
        prev_beta = momentum * prev_beta + update
        beta -= prev_beta

        mse = (1 / (2 * m)) * anp.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i
        prev_loss = mse

    return beta, iterations


def adam_sgd_ols(X, y, learning_rate, epochs, mb_size, tolerance=1e-6):
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    training_gradient = grad(CostOLS)
    delta = 1e-8
    first_moment = 0.0
    second_moment = 0.0
    beta1 = 0.9
    beta2 = 0.999

    for epoch in range(epochs):
        for i in range(1, iterations + 1):
            index = mb_size * np.random.randint(iterations)
            Xi = X[index : index + mb_size] if index + mb_size <= m else X[index:m]
            yi = y[index : index + mb_size] if index + mb_size <= m else y[index:m]
            y_pred = Xi @ beta
            gradients = (1.0 / m) * training_gradient(beta, X, y)
            first_moment = beta1 * first_moment + (1 - beta1) * gradients
            second_moment = beta2 * second_moment + (1 - beta2) * gradients * gradients
            first_term = first_moment / (1.0 - beta1**i)
            second_term = second_moment / (1.0 - beta2**i)
            update = learning_rate * first_term / (np.sqrt(second_term) + delta)
            beta -= update

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2)

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch
            prev_loss = mse

    return beta, iterations, epochs


def adam_sgd_ridge(X, y, lmd, learning_rate, epochs, mb_size, tolerance=1e-6):
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    training_gradient = grad(CostRidge)
    first_moment = 0.0
    second_moment = 0.0
    beta1 = 0.9
    beta2 = 0.999
    delta = 1e-8
    for epoch in range(epochs):
        for i in range(1, iterations + 1):
            index = mb_size * np.random.randint(iterations)
            Xi = X[index : index + mb_size] if index + mb_size <= m else X[index:m]
            yi = y[index : index + mb_size] if index + mb_size <= m else y[index:m]
            y_pred = Xi @ beta
            gradients = (1.0 / m) * training_gradient(beta, X, y, lmd)
            first_moment = beta1 * first_moment + (1 - beta1) * gradients
            second_moment = beta2 * second_moment + (1 - beta2) * gradients * gradients
            first_term = first_moment / (1.0 - beta1**i)
            second_term = second_moment / (1.0 - beta2**i)
            update = learning_rate * first_term / (np.sqrt(second_term) + delta)
            beta -= update

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2) + (lmd / (2 * m)) * np.sum(
                beta**2
            )

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch
            prev_loss = mse

    return beta, iterations, epochs


def adam_sgd_momentum_ols(
    X, y, learning_rate, epochs, mb_size, momentum, tolerance=1e-6
):
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    training_gradient = grad(CostOLS)
    first_moment = 0.0
    second_moment = 0.0
    beta1 = 0.9
    beta2 = 0.999
    delta = 1e-8
    prev_beta = anp.zeros_like(beta)

    for epoch in range(epochs):
        for i in range(1, iterations + 1):
            index = mb_size * np.random.randint(iterations)
            Xi = X[index : index + mb_size] if index + mb_size <= m else X[index:m]
            yi = y[index : index + mb_size] if index + mb_size <= m else y[index:m]
            y_pred = Xi @ beta
            gradients = (1.0 / m) * training_gradient(beta, X, y)
            first_moment = beta1 * first_moment + (1 - beta1) * gradients
            second_moment = beta2 * second_moment + (1 - beta2) * gradients * gradients
            first_term = first_moment / (1.0 - beta1**i)
            second_term = second_moment / (1.0 - beta2**i)
            update = learning_rate * first_term / (np.sqrt(second_term) + delta)
            prev_beta = momentum * prev_beta + update
            beta -= prev_beta

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2)

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch
            prev_loss = mse

    return beta, iterations, epochs


def adam_sgd_momentum_ridge(
    X, y, lmd, learning_rate, epochs, mb_size, momentum, tolerance=1e-6
):
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    training_gradient = grad(CostRidge)
    delta = 1e-8
    prev_beta = anp.zeros_like(beta)
    first_moment = 0.0
    second_moment = 0.0
    beta1 = 0.9
    beta2 = 0.999

    for epoch in range(epochs):
        for i in range(1, iterations + 1):
            index = mb_size * np.random.randint(iterations)
            Xi = X[index : index + mb_size] if index + mb_size <= m else X[index:m]
            yi = y[index : index + mb_size] if index + mb_size <= m else y[index:m]
            y_pred = Xi @ beta
            gradients = (1.0 / m) * training_gradient(beta, X, y, lmd)
            first_moment = beta1 * first_moment + (1 - beta1) * gradients
            second_moment = beta2 * second_moment + (1 - beta2) * gradients * gradients
            first_term = first_moment / (1.0 - beta1**i)
            second_term = second_moment / (1.0 - beta2**i)
            update = learning_rate * first_term / (np.sqrt(second_term) + delta)
            prev_beta = momentum * prev_beta + update
            beta -= prev_beta

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2) + (lmd / (2 * m)) * np.sum(
                beta**2
            )

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch
            prev_loss = mse

    return beta, iterations, epochs


def franke_function(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4
