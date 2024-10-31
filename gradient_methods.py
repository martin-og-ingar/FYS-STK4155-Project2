import autograd.numpy as np
from autograd import grad
import autograd.numpy as anp
from global_values import USE_GRAD
from sklearn.model_selection import train_test_split


def CostOLS(beta, X, y):
    return anp.sum((y - X @ beta) ** 2)


def CostRidge(beta, X, y, lmd):
    m = X.shape[0]
    return anp.sum((y - X @ beta) ** 2) / m + (lmd / (2 * m)) * anp.sum(beta**2)


def learning_schedule(t):
    return 5 / (t + 50)


def gradient_descent_ols(X, y, learning_rate, iterations, tolerance=1e-6):
    if USE_GRAD:
        training_gradient = grad(CostOLS)
    m, n = X.shape
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    for i in range(iterations):
        y_pred = X @ beta
        if USE_GRAD:
            gradient = (1 / m) * training_gradient(beta, X, y)
        else:
            gradient = (1 / m) * X.T @ (y_pred - y)
        beta -= learning_rate * gradient

        mse = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i, mse
        prev_loss = mse

    return beta, iterations, mse


def gradient_descent_ridge(X, y, lmd, learning_rate, iterations, tolerance=1e-6):
    if USE_GRAD:
        training_descent = grad(CostRidge)
    m, n = X.shape
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    for i in range(iterations):
        y_pred = X @ beta
        if USE_GRAD:
            gradient = (1 / m) * training_descent(beta, X, y, lmd)
        else:
            gradient = (1 / m) * (X.T @ (y_pred - y) + lmd * beta)
        beta -= learning_rate * gradient

        mse = (1 / (2 * m)) * np.sum((y_pred - y) ** 2) + (lmd / (2 * m)) * np.sum(
            beta**2
        )

        if abs(prev_loss - mse) < tolerance:
            return beta, i, mse
        prev_loss = mse

    return beta, iterations, mse


def gradient_descent_momentum_ols(
    X, y, learning_rate, iterations, momentum, tolerance=1e-6
):
    if USE_GRAD:
        training_descent = grad(CostOLS)
    m, n = X.shape
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    velocity = np.zeros_like(beta)

    for i in range(iterations):
        y_pred = X @ beta
        if USE_GRAD:
            gradient = (1 / m) * training_descent(beta, X, y)
        else:
            gradient = (1 / m) * X.T @ (y_pred - y)
        velocity = momentum * velocity - learning_rate * gradient
        beta += velocity

        mse = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i, mse
        prev_loss = mse

    return beta, iterations, mse


def gradient_descent_momentum_ridge(
    X, y, lmd, learning_rate, iterations, momentum, tolerance=1e-6
):
    if USE_GRAD:
        training_gradient = grad(CostRidge)
    m, n = X.shape
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    velocity = np.zeros_like(beta)

    for i in range(iterations):
        y_pred = X @ beta
        if USE_GRAD:
            gradient = (1 / m) * training_gradient(beta, X, y, lmd)
        else:
            gradient = (1 / m) * (X.T @ (y_pred - y) + lmd * beta)
        velocity = momentum * velocity - learning_rate * gradient
        beta += velocity

        mse = (1 / (2 * m)) * np.sum((y_pred - y) ** 2) + (lmd / (2 * m)) * np.sum(
            beta**2
        )

        if abs(prev_loss - mse) < tolerance:
            return beta, i, mse
        prev_loss = mse

    return beta, iterations, mse


def sgd_ols(X, y, epochs, mb_size, tolerance=1e-6):
    if USE_GRAD:
        training_gradient = grad(CostOLS)
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
            if USE_GRAD:
                gradient = (1 / mb_size) * training_gradient(beta, Xi, yi)
            else:
                gradient = (1 / mb_size) * Xi.T @ (y_pred - yi)
            learning_rate = learning_schedule(epoch * iterations + i)
            beta -= learning_rate * gradient

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2)

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch, mse
            prev_loss = mse

    return beta, iterations, epochs, mse


def sgd_ridge(X, y, lmd, epochs, mb_size, tolerance=1e-6):
    if USE_GRAD:
        training_gradient = grad(CostRidge)
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
            if USE_GRAD:
                gradient = (1 / mb_size) * training_gradient(beta, Xi, yi, lmd)
            else:
                gradient = (1 / mb_size) * (Xi.T @ (y_pred - yi) + lmd * beta)
            learning_rate = learning_schedule(epoch * iterations + i)
            beta -= learning_rate * gradient

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2) + (lmd / (2 * m)) * np.sum(
                beta**2
            )

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch, mse
            prev_loss = mse

    return beta, iterations, epochs, mse


def sgd_momentum_ols(X, y, epochs, mb_size, momentum, tolerance=1e-6):
    if USE_GRAD:
        training_gradient = grad(CostOLS)
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
            if USE_GRAD:
                gradient = (1 / mb_size) * training_gradient(beta, Xi, yi)
            else:
                gradient = (1 / mb_size) * Xi.T @ (y_pred - yi)
            learning_rate = learning_schedule(epoch * iterations + i)
            velocity = momentum * velocity - learning_rate * gradient
            beta += velocity

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2)

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch, mse
            prev_loss = mse

    return beta, iterations, epochs, mse


def sgd_momentum_ridge(X, y, lmd, epochs, mb_size, momentum, tolerance=1e-6):
    if USE_GRAD:
        training_gradient = grad(CostRidge)
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
            if USE_GRAD:
                gradient = (1 / mb_size) * training_gradient(beta, Xi, yi, lmd)
            else:
                gradient = (1 / mb_size) * (Xi.T @ (y_pred - yi) + lmd * beta)
            learning_rate = learning_schedule(epoch * iterations + i)
            velocity = momentum * velocity - learning_rate * gradient
            beta += velocity

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2) + (lmd / (2 * m)) * np.sum(
                beta**2
            )

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch, mse
            prev_loss = mse

    return beta, iterations, epochs, mse


def adagrad_pgd_ols(X, y, learning_rate, iterations, tolerance=1e-6):
    if USE_GRAD:
        training_gradient = grad(CostOLS)
    m, n = X.shape
    beta = anp.zeros((n, 1))
    prev_loss = float("inf")
    Giter = 0.0
    rho = 0.99
    delta = 1e-8

    for i in range(iterations):
        y_pred = X @ beta
        if USE_GRAD:
            gradients = (1.0 / m) * training_gradient(beta, X, y)
        else:
            gradients = (1.0 / m) * X.T @ (y_pred - y)
        Giter = rho * Giter + (1 - rho) * gradients * gradients
        update = gradients * learning_rate / (delta + anp.sqrt(Giter))
        beta -= update

        mse = (1 / (2 * m)) * anp.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i, mse
        prev_loss = mse

    return beta, iterations, mse


def adagrad_pgd_ridge(X, y, lmd, learning_rate, iterations, tolerance=1e-6):
    if USE_GRAD:
        training_gradient = grad(CostRidge)
    m, n = X.shape
    beta = anp.zeros((n, 1))
    prev_loss = float("inf")
    Giter = 0.0
    rho = 0.99
    delta = 1e-8

    for i in range(iterations):
        y_pred = X @ beta
        if USE_GRAD:
            gradients = (1.0 / m) * training_gradient(beta, X, y, lmd)
        else:
            gradients = (1.0 / m) * (X.T @ (y_pred - y) + lmd * beta)
        Giter = rho * Giter + (1 - rho) * gradients * gradients
        update = gradients * learning_rate / (delta + anp.sqrt(Giter))
        beta -= update

        mse = (1 / (2 * m)) * anp.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i, mse
        prev_loss = mse

    return beta, iterations, mse


def adagrad_gdm_ols(X, y, learning_rate, iterations, momentum, tolerance=1e-6):
    if USE_GRAD:
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
        if USE_GRAD:
            gradients = (1.0 / m) * training_gradient(beta, X, y)
        else:
            gradients = (1.0 / m) * X.T @ (y_pred - y)
        Giter = rho * Giter + (1 - rho) * gradients * gradients
        update = gradients * learning_rate / (delta + anp.sqrt(Giter))
        prev_beta = momentum * prev_beta + update
        beta -= prev_beta

        mse = (1 / (2 * m)) * anp.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i, mse
        prev_loss = mse

    return beta, iterations, mse


def adagrad_gdm_ridge(X, y, lmd, learning_rate, iterations, momentum, tolerance=1e-6):
    if USE_GRAD:
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
        if USE_GRAD:
            gradients = (1.0 / m) * training_gradient(beta, X, y, lmd)
        else:
            gradients = (1.0 / m) * (X.T @ (y_pred - y) + lmd * beta)
        Giter = rho * Giter + (1 - rho) * gradients * gradients
        update = gradients * learning_rate / (delta + anp.sqrt(Giter))
        prev_beta = momentum * prev_beta + update
        beta -= prev_beta

        mse = (1 / (2 * m)) * anp.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i, mse
        prev_loss = mse

    return beta, iterations, mse


def adagrad_sgd_ols(X, y, learning_rate, epochs, mb_size, tolerance=1e-6):
    if USE_GRAD:
        training_gradient = grad(CostOLS)
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    Giter = 0.0
    rho = 0.99
    delta = 1e-8

    for epoch in range(epochs):
        for i in range(iterations):
            index = mb_size * np.random.randint(iterations)
            Xi = X[index : index + mb_size] if index + mb_size <= m else X[index:m]
            yi = y[index : index + mb_size] if index + mb_size <= m else y[index:m]
            y_pred = Xi @ beta
            if USE_GRAD:
                gradients = (1.0 / m) * training_gradient(beta, Xi, yi)
            else:
                gradients = (1.0 / m) * Xi.T @ (y_pred - yi)
            Giter = rho * Giter + (1 - rho) * gradients * gradients
            update = gradients * learning_rate / (delta + anp.sqrt(Giter))
            beta -= update

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2)

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch, mse
            prev_loss = mse

    return beta, iterations, epochs, mse


def adagrad_sgd_ridge(X, y, lmd, learning_rate, epochs, mb_size, tolerance=1e-6):
    if USE_GRAD:
        training_gradient = grad(CostRidge)
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    Giter = 0.0
    rho = 0.99
    delta = 1e-8
    for epoch in range(epochs):
        for i in range(iterations):
            index = mb_size * np.random.randint(iterations)
            Xi = X[index : index + mb_size] if index + mb_size <= m else X[index:m]
            yi = y[index : index + mb_size] if index + mb_size <= m else y[index:m]
            y_pred = Xi @ beta
            if USE_GRAD:
                gradients = (1.0 / m) * training_gradient(beta, Xi, yi, lmd)
            else:
                gradients = (1.0 / m) * (Xi.T @ (y_pred - yi) + lmd * beta)
            Giter = rho * Giter + (1 - rho) * gradients * gradients
            update = gradients * learning_rate / (delta + anp.sqrt(Giter))
            beta -= update

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2) + (lmd / (2 * m)) * np.sum(
                beta**2
            )

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch, mse
            prev_loss = mse

    return beta, iterations, epochs, mse


def adagrad_sgd_momentum_ols(
    X, y, learning_rate, epochs, mb_size, momentum, tolerance=1e-6
):
    if USE_GRAD:
        training_gradient = grad(CostOLS)
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
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
            if USE_GRAD:
                gradients = (1.0 / m) * training_gradient(beta, Xi, yi)
            else:
                gradients = (1.0 / m) * Xi.T @ (y_pred - yi)
            Giter = rho * Giter + (1 - rho) * gradients * gradients
            update = gradients * learning_rate / (delta + anp.sqrt(Giter))
            prev_beta = momentum * prev_beta + update
            beta -= prev_beta

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2)

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch, mse
            prev_loss = mse

    return beta, iterations, epochs, mse


def adagrad_sgd_momentum_ridge(
    X, y, lmd, learning_rate, epochs, mb_size, momentum, tolerance=1e-6
):
    if USE_GRAD:
        training_gradient = grad(CostRidge)
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
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
            if USE_GRAD:
                gradients = (1.0 / m) * training_gradient(beta, Xi, yi, lmd)
            else:
                gradients = (1.0 / m) * (Xi.T @ (y_pred - yi) + lmd * beta)
            Giter = rho * Giter + (1 - rho) * gradients * gradients
            update = gradients * learning_rate / (delta + anp.sqrt(Giter))
            prev_beta = momentum * prev_beta + update
            beta -= prev_beta

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2) + (lmd / (2 * m)) * np.sum(
                beta**2
            )

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch, mse
            prev_loss = mse

    return beta, iterations, epochs, mse


def rms_pgd_ols(X, y, learning_rate, iterations, tolerance=1e-6):
    if USE_GRAD:
        training_gradient = grad(CostOLS)
    m, n = X.shape
    beta = anp.zeros((n, 1))
    prev_loss = float("inf")
    Giter = 0.0
    rho = 0.99
    delta = 1e-8

    for i in range(iterations):
        y_pred = X @ beta
        if USE_GRAD:
            gradients = (1.0 / m) * training_gradient(beta, X, y)
        else:
            gradients = (1.0 / m) * X.T @ (y_pred - y)
        Giter = rho * Giter + (1 - rho) * anp.square(gradients)
        update = gradients * learning_rate / (delta + anp.sqrt(Giter))
        beta -= update

        mse = (1 / (2 * m)) * anp.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i, mse
        prev_loss = mse

    return beta, iterations, mse


def rms_pgd_ridge(X, y, lmd, learning_rate, iterations, tolerance=1e-6):
    if USE_GRAD:
        training_gradient = grad(CostRidge)
    m, n = X.shape
    beta = anp.zeros((n, 1))
    prev_loss = float("inf")
    Giter = 0.0
    rho = 0.99
    delta = 1e-8

    for i in range(iterations):
        y_pred = X @ beta
        if USE_GRAD:
            gradients = (1.0 / m) * training_gradient(beta, X, y, lmd)
        else:
            gradients = (1.0 / m) * (X.T @ (y_pred - y) + lmd * beta)
        Giter = rho * Giter + (1 - rho) * anp.square(gradients)
        update = gradients * learning_rate / (delta + anp.sqrt(Giter))
        beta -= update

        mse = (1 / (2 * m)) * anp.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i, mse
        prev_loss = mse

    return beta, iterations, mse


def rms_gdm_ols(X, y, learning_rate, iterations, momentum, tolerance=1e-6):
    if USE_GRAD:
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
        if USE_GRAD:
            gradients = (1.0 / m) * training_gradient(beta, X, y)
        else:
            gradients = (1.0 / m) * X.T @ (y_pred - y)
        Giter = rho * Giter + (1 - rho) * anp.square(gradients)
        update = gradients * learning_rate / (delta + anp.sqrt(Giter))
        prev_beta = momentum * prev_beta + update
        beta -= prev_beta

        mse = (1 / (2 * m)) * anp.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i, mse
        prev_loss = mse

    return beta, iterations, mse


def rms_gdm_ridge(X, y, lmd, learning_rate, iterations, momentum, tolerance=1e-6):
    if USE_GRAD:
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
        if USE_GRAD:
            gradients = (1.0 / m) * training_gradient(beta, X, y, lmd)
        else:
            gradients = (1.0 / m) * (X.T @ (y_pred - y) + lmd * beta)
        Giter = rho * Giter + (1 - rho) * anp.square(gradients)
        update = gradients * learning_rate / (delta + anp.sqrt(Giter))
        prev_beta = momentum * prev_beta + update
        beta -= prev_beta

        mse = (1 / (2 * m)) * anp.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i, mse
        prev_loss = mse

    return beta, iterations, mse


def rms_sgd_ols(X, y, learning_rate, epochs, mb_size, tolerance=1e-6):
    if USE_GRAD:
        training_gradient = grad(CostOLS)
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    Giter = 0.0
    rho = 0.99
    delta = 1e-8

    for epoch in range(epochs):
        for i in range(iterations):
            index = mb_size * np.random.randint(iterations)
            Xi = X[index : index + mb_size] if index + mb_size <= m else X[index:m]
            yi = y[index : index + mb_size] if index + mb_size <= m else y[index:m]
            y_pred = Xi @ beta
            if USE_GRAD:
                gradients = (1.0 / m) * training_gradient(beta, Xi, yi)
            else:
                gradients = (1.0 / m) * Xi.T @ (y_pred - yi)
            Giter = rho * Giter + (1 - rho) * anp.square(gradients)
            update = gradients * learning_rate / (delta + anp.sqrt(Giter))
            beta -= update

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2)

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch, mse
            prev_loss = mse

    return beta, iterations, epochs, mse


def rms_sgd_ridge(X, y, lmd, learning_rate, epochs, mb_size, tolerance=1e-6):
    if USE_GRAD:
        training_gradient = grad(CostRidge)
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
    Giter = 0.0
    rho = 0.99
    delta = 1e-8
    for epoch in range(epochs):
        for i in range(iterations):
            index = mb_size * np.random.randint(iterations)
            Xi = X[index : index + mb_size] if index + mb_size <= m else X[index:m]
            yi = y[index : index + mb_size] if index + mb_size <= m else y[index:m]
            y_pred = Xi @ beta
            if USE_GRAD:
                gradients = (1.0 / m) * training_gradient(beta, Xi, yi, lmd)
            else:
                gradients = (1.0 / m) * (Xi.T @ (y_pred - yi) + lmd * beta)
            Giter = rho * Giter + (1 - rho) * anp.square(gradients)
            update = gradients * learning_rate / (delta + anp.sqrt(Giter))
            beta -= update

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2) + (lmd / (2 * m)) * np.sum(
                beta**2
            )

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch, mse
            prev_loss = mse

    return beta, iterations, epochs, mse


def rms_sgd_momentum_ols(
    X, y, learning_rate, epochs, mb_size, momentum, tolerance=1e-6
):
    if USE_GRAD:
        training_gradient = grad(CostOLS)
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
            if USE_GRAD:
                gradients = (1.0 / m) * training_gradient(beta, Xi, yi)
            else:
                gradients = (1.0 / m) * Xi.T @ (y_pred - yi)
            Giter = rho * Giter + (1 - rho) * anp.square(gradients)
            update = gradients * learning_rate / (delta + anp.sqrt(Giter))
            prev_beta = momentum * prev_beta + update
            beta -= prev_beta

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2)

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch, mse
            prev_loss = mse

    return beta, iterations, epochs, mse


def rms_sgd_momentum_ridge(
    X, y, lmd, learning_rate, epochs, mb_size, momentum, tolerance=1e-6
):
    if USE_GRAD:
        training_gradient = grad(CostRidge)
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
            if USE_GRAD:
                gradients = (1.0 / m) * training_gradient(beta, Xi, yi, lmd)
            else:
                gradients = (1.0 / m) * (Xi.T @ (y_pred - yi) + lmd * beta)
            Giter = rho * Giter + (1 - rho) * anp.square(gradients)
            update = gradients * learning_rate / (delta + anp.sqrt(Giter))
            prev_beta = momentum * prev_beta + update
            beta -= prev_beta

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2) + (lmd / (2 * m)) * np.sum(
                beta**2
            )

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch, mse
            prev_loss = mse

    return beta, iterations, epochs, mse


def adam_pgd_ols(X, y, learning_rate, iterations, tolerance=1e-6):
    if USE_GRAD:
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
        if USE_GRAD:
            gradients = (1.0 / m) * training_gradient(beta, X, y)
        else:
            gradients = (1.0 / m) * X.T @ (y_pred - y)
        first_moment = beta1 * first_moment + (1 - beta1) * gradients
        second_moment = beta2 * second_moment + (1 - beta2) * gradients * gradients
        first_term = first_moment / (1.0 - beta1**i)
        second_term = second_moment / (1.0 - beta2**i)
        update = learning_rate * first_term / (np.sqrt(second_term) + delta)
        beta -= update

        mse = (1 / (2 * m)) * anp.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i, mse
        prev_loss = mse

    return beta, iterations, mse


def adam_pgd_ridge(X, y, lmd, learning_rate, iterations, tolerance=1e-6):
    if USE_GRAD:
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
        if USE_GRAD:
            gradients = (1.0 / m) * training_gradient(beta, X, y, lmd)
        else:
            gradients = (1.0 / m) * (X.T @ (y_pred - y) + lmd * beta)
        first_moment = beta1 * first_moment + (1 - beta1) * gradients
        second_moment = beta2 * second_moment + (1 - beta2) * gradients * gradients
        first_term = first_moment / (1.0 - beta1**i)
        second_term = second_moment / (1.0 - beta2**i)
        update = learning_rate * first_term / (np.sqrt(second_term) + delta)
        beta -= update

        mse = (1 / (2 * m)) * anp.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i, mse
        prev_loss = mse

    return beta, iterations, mse


def adam_gdm_ols(X, y, learning_rate, iterations, momentum, tolerance=1e-6):
    if USE_GRAD:
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
        if USE_GRAD:
            gradients = (1.0 / m) * training_gradient(beta, X, y)
        else:
            gradients = (1.0 / m) * X.T @ (y_pred - y)
        first_moment = beta1 * first_moment + (1 - beta1) * gradients
        second_moment = beta2 * second_moment + (1 - beta2) * gradients * gradients
        first_term = first_moment / (1.0 - beta1**i)
        second_term = second_moment / (1.0 - beta2**i)
        update = learning_rate * first_term / (np.sqrt(second_term) + delta)
        prev_beta = momentum * prev_beta + update
        beta -= prev_beta

        mse = (1 / (2 * m)) * anp.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i, mse
        prev_loss = mse

    return beta, iterations, mse


def adam_gdm_ridge(X, y, lmd, learning_rate, iterations, momentum, tolerance=1e-6):
    if USE_GRAD:
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
        if USE_GRAD:
            gradients = (1.0 / m) * training_gradient(beta, X, y, lmd)
        else:
            gradients = (1.0 / m) * (X.T @ (y_pred - y) + lmd * beta)
        first_moment = beta1 * first_moment + (1 - beta1) * gradients
        second_moment = beta2 * second_moment + (1 - beta2) * gradients * gradients
        first_term = first_moment / (1.0 - beta1**i)
        second_term = second_moment / (1.0 - beta2**i)
        update = learning_rate * first_term / (np.sqrt(second_term) + delta)
        prev_beta = momentum * prev_beta + update
        beta -= prev_beta

        mse = (1 / (2 * m)) * anp.sum((y_pred - y) ** 2)

        if abs(prev_loss - mse) < tolerance:
            return beta, i, mse
        prev_loss = mse

    return beta, iterations, mse


def adam_sgd_ols(X, y, learning_rate, epochs, mb_size, tolerance=1e-6):
    if USE_GRAD:
        training_gradient = grad(CostOLS)
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
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
            if USE_GRAD:
                gradients = (1.0 / m) * training_gradient(beta, Xi, yi)
            else:
                gradients = (1.0 / m) * Xi.T @ (y_pred - yi)
            first_moment = beta1 * first_moment + (1 - beta1) * gradients
            second_moment = beta2 * second_moment + (1 - beta2) * gradients * gradients
            first_term = first_moment / (1.0 - beta1**i)
            second_term = second_moment / (1.0 - beta2**i)
            update = learning_rate * first_term / (np.sqrt(second_term) + delta)
            beta -= update

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2)

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch, mse
            prev_loss = mse

    return beta, iterations, epochs, mse


def adam_sgd_ridge(X, y, lmd, learning_rate, epochs, mb_size, tolerance=1e-6):
    if USE_GRAD:
        training_gradient = grad(CostRidge)
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
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
            if USE_GRAD:
                gradients = (1.0 / m) * training_gradient(beta, Xi, yi, lmd)
            else:
                gradients = (1.0 / m) * (Xi.T @ (y_pred - yi) + lmd * beta)
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
                return beta, i, epoch, mse
            prev_loss = mse

    return beta, iterations, epochs, mse


def adam_sgd_momentum_ols(
    X, y, learning_rate, epochs, mb_size, momentum, tolerance=1e-6
):
    if USE_GRAD:
        training_gradient = grad(CostOLS)
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
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
            if USE_GRAD:
                gradients = (1.0 / m) * training_gradient(beta, Xi, yi)
            else:
                gradients = (1.0 / m) * Xi.T @ (y_pred - yi)
            first_moment = beta1 * first_moment + (1 - beta1) * gradients
            second_moment = beta2 * second_moment + (1 - beta2) * gradients * gradients
            first_term = first_moment / (1.0 - beta1**i)
            second_term = second_moment / (1.0 - beta2**i)
            update = learning_rate * first_term / (np.sqrt(second_term) + delta)
            prev_beta = momentum * prev_beta + update
            beta -= prev_beta

            mse = (1 / (2 * m)) * np.sum((y_pred - yi) ** 2)

            if abs(prev_loss - mse) < tolerance:
                return beta, i, epoch, mse
            prev_loss = mse

    return beta, iterations, epochs, mse


def adam_sgd_momentum_ridge(
    X, y, lmd, learning_rate, epochs, mb_size, momentum, tolerance=1e-6
):
    if USE_GRAD:
        training_gradient = grad(CostRidge)
    m, n = X.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
    prev_loss = float("inf")
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
            if USE_GRAD:
                gradients = (1.0 / m) * training_gradient(beta, Xi, yi, lmd)
            else:
                gradients = (1.0 / m) * (Xi.T @ (y_pred - yi) + lmd * beta)
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
                return beta, i, epoch, mse
            prev_loss = mse

    return beta, iterations, epochs, mse


def sigmoid(z):
    # Clip input to avoid overflow in exp
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def CostLog(beta, X, y):
    # Calculate predictions with sigmoid
    y_pred = sigmoid(X @ beta)

    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


def sgd_log(X, y, mb_size, epochs, tolerance=1e-6):
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2)
    training_gradient = grad(CostLog)
    m, n = X_train.shape
    iterations = int(m / mb_size)
    beta = np.zeros((n, 1))
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
            gradient = (1 / mb_size) * training_gradient(beta, Xi, yi)
            learning_rate = learning_schedule(epoch * iterations + i)
            beta -= learning_rate * gradient

            score = CostLog(beta, X_validate, y_validate)

            if abs(score - prev_score) < tolerance:
                return beta, score
            prev_score = score

    return beta, score
