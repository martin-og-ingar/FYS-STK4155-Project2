import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods import (
    gradient_descent_ols,
    gradient_descent_ridge,
    gradient_descent_momentum_ols,
    gradient_descent_momentum_ridge,
    sgd_ols,
    sgd_ridge,
    sgd_momentum_ols,
    sgd_momentum_ridge,
    adagrad_pgd_ols,
    adagrad_pgd_ridge,
    adagrad_gdm_ols,
    adagrad_gdm_ridge,
    adagrad_sgd_ols,
    adagrad_sgd_ridge,
    adagrad_sgd_momentum_ols,
    adagrad_sgd_momentum_ridge,
    rms_pgd_ols,
    rms_pgd_ridge,
    rms_gdm_ols,
    rms_gdm_ridge,
    rms_sgd_ols,
    rms_sgd_ridge,
    rms_sgd_momentum_ols,
    rms_sgd_momentum_ridge,
    adam_pgd_ols,
    adam_pgd_ridge,
    adam_gdm_ols,
    adam_gdm_ridge,
    adam_sgd_ols,
    adam_sgd_ridge,
    adam_sgd_momentum_ols,
    adam_sgd_momentum_ridge,
)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import StandardScaler

"""
    Optimizing the parameters with matrix inversion gives us the parameters 2.1 and 1.7.
    In most cases of gradient descent, the result is the same. However some cases does not reach convergence before number of iterations is done,
    or perhaps trapped in a local minima. Based on the results, we see that it is often the case in the extreme cases, where learning rate and momentum
    is either small or large. 

    When optimizing with small learning rate and momentum, the learning is too conservative and the optimal parameters are not found in time.
    When optimizing with larger values, the gradient descent is spending too much time jumping over the global minima.
"""


"""
1 - GD
2 - GDM
3 - SGD
4 - SGDM
5 - AadaGD
6 - AdaGDM
7 - AdaSGD
8 - AdaSGDM
9 - RSPGD
10 - RSPGDM
11 - RSPSGD
12 - RSPSGDM
13 - AdamGD
14 - AdamGDM
15 - AdamSGD
16 - AdamSGDM
"""

SHOULD_RUN = [1, 2, 3, 4]


np.random.seed(100)

n = 100
x = 2 * np.random.rand(n, 1)
x_scaler = StandardScaler()
x_scaled = x_scaler.fit_transform(x)
y = x + x**2
X = np.c_[np.ones((n, 1)), x_scaled]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

beta_OLS_minv = np.linalg.pinv(x_train.T @ x_train) @ x_train.T @ y_train
beta_ridge_minv = (
    np.linalg.pinv(x_train.T @ x_train + 0.01 * np.eye(x_train.shape[1]))
    @ x_train.T
    @ y_train
)
learning_rates = [0.01, 0.1, 0.2, 0.5]
momentum = [0.1, 0.2, 0.3, 0.4]

if 1 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Plain gradient descent \n"
        + "========================================================================================"
    )

    for lr in learning_rates:
        beta_OLS_gd, iterations_before_convergence_ols = gradient_descent_ols(
            x_train, y_train, learning_rate=lr, iterations=1000
        )

        beta_ridge_gd, iterations_before_convergence_ridge = gradient_descent_ridge(
            x_train, y_train, 0.01, learning_rate=lr, iterations=1000
        )
        print("OLS: Learning rate " + str(lr) + ": \n" + str(beta_OLS_gd))
        print("...and converged at " + str(iterations_before_convergence_ols))
        print("Ridge: Learning rate " + str(lr) + ": \n" + str(beta_ridge_gd))
        print("...and converged at " + str(iterations_before_convergence_ridge))

    print("Bheta for OLS with matrix inversion: " + str(beta_OLS_minv))
    print("Bheta for ridge regression with matrix inversion: " + str(beta_ridge_minv))

if 2 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Gradient descent with momentum \n"
        + "========================================================================================"
    )

    for mom in momentum:
        print("Momentum: " + str(mom))
        for lr in learning_rates:
            beta_OLS_gd, iterations_before_convergence_ols = (
                gradient_descent_momentum_ols(
                    x_train, y_train, learning_rate=lr, iterations=1000, momentum=mom
                )
            )

            beta_ridge_gd, iterations_before_convergence_ridge = (
                gradient_descent_momentum_ridge(
                    x_train,
                    y_train,
                    0.01,
                    learning_rate=lr,
                    iterations=1000,
                    momentum=mom,
                )
            )
            print("OLS: Learning rate " + str(lr) + ": \n" + str(beta_OLS_gd))
            print("...and converged at " + str(iterations_before_convergence_ols))
            print("Ridge: Learning rate " + str(lr) + ": \n" + str(beta_ridge_gd))
            print("...and converged at " + str(iterations_before_convergence_ridge))

if 3 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Stochastic gradient descent \n"
        + "========================================================================================"
    )

    for lr in learning_rates:
        beta_OLS_gd, iterations_before_convergence_ols, epoch_ols = sgd_ols(
            x_train, y_train, learning_rate=lr, epochs=50, mb_size=5
        )

        beta_ridge_gd, iterations_before_convergence_ridge, epoch_ridge = sgd_ridge(
            x_train, y_train, 0.01, learning_rate=lr, epochs=50, mb_size=5
        )
        print("OLS: Learning rate " + str(lr) + ": \n" + str(beta_OLS_gd))
        print("...and converged at " + str(iterations_before_convergence_ols))
        print(" in epoch " + str(epoch_ols))
        print("Ridge: Learning rate " + str(lr) + ": \n" + str(beta_ridge_gd))
        print("...and converged at " + str(iterations_before_convergence_ridge))
        print(" in epoch " + str(epoch_ridge))

if 4 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Stochastic gradient descent with momentum \n"
        + "========================================================================================"
    )

    for mom in momentum:
        print("Momentum: " + str(mom))
        for lr in learning_rates:
            beta_OLS_gd, iterations_before_convergence_ols, epoch_ols = (
                sgd_momentum_ols(
                    x_train,
                    y_train,
                    learning_rate=lr,
                    epochs=60,
                    mb_size=5,
                    momentum=mom,
                )
            )

            beta_ridge_gd, iterations_before_convergence_ridge, epoch_ridge = (
                sgd_momentum_ridge(
                    x_train,
                    y_train,
                    0.01,
                    learning_rate=lr,
                    epochs=60,
                    mb_size=5,
                    momentum=mom,
                )
            )
            print("OLS: Learning rate " + str(lr) + ": \n" + str(beta_OLS_gd))
            print("...and converged at " + str(iterations_before_convergence_ols))
            print(" in epoch " + str(epoch_ols))
            print("Ridge: Learning rate " + str(lr) + ": \n" + str(beta_ridge_gd))
            print("...and converged at " + str(iterations_before_convergence_ridge))
            print(" in epoch " + str(epoch_ridge))

if 5 in SHOULD_RUN:

    print(
        "========================================================================================\n"
        + "Plain gradient descent with Adagrad \n"
        + "========================================================================================"
    )
    for lr in learning_rates:
        beta_OLS_pgd_adagrad, iterations_before_convergence_ols = adagrad_pgd_ols(
            x_train, y_train, learning_rate=lr, iterations=1000
        )
        beta_ridge_pgd_adagrad, iterations_before_convergence_ridge = adagrad_pgd_ridge(
            x_train, y_train, 0.01, learning_rate=lr, iterations=1000
        )
        print("OLS: Learning rate " + str(lr) + ": \n" + str(beta_OLS_pgd_adagrad))
        print("...and converged at " + str(iterations_before_convergence_ols))
        print("Ridge: Learning rate " + str(lr) + ": \n" + str(beta_ridge_pgd_adagrad))
        print("...and converged at " + str(iterations_before_convergence_ridge))

if 6 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Gradient descent with Adagrad and momentum \n"
        + "========================================================================================"
    )
    for mom in momentum:
        print("Momentum: " + str(mom))
        for lr in learning_rates:
            beta_OLS_gdm_adagrad, iterations_before_convergence_ols = adagrad_gdm_ols(
                x_train, y_train, learning_rate=lr, momentum=mom, iterations=1000
            )
            beta_ridge_gdm_adagrad, iterations_before_convergence_ridge = (
                adagrad_gdm_ridge(
                    x_train,
                    y_train,
                    0.01,
                    learning_rate=lr,
                    momentum=mom,
                    iterations=1000,
                )
            )
            print("OLS: Learning rate " + str(lr) + ": \n" + str(beta_OLS_gdm_adagrad))
            print("...and converged at " + str(iterations_before_convergence_ols))
            print(
                "Ridge: Learning rate " + str(lr) + ": \n" + str(beta_ridge_gdm_adagrad)
            )
            print("...and converged at " + str(iterations_before_convergence_ridge))

if 7 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Stochastic gradient descent with Adagrad \n"
        + "========================================================================================"
    )
    for lr in learning_rates:
        beta_OLS_sgd_adagrad, iterations_before_convergence_ols, epoch_ols = (
            adagrad_sgd_ols(x_train, y_train, learning_rate=lr, epochs=60, mb_size=5)
        )
        beta_ridge_sgd_adagrad, iterations_before_convergence_ridge, epoch_ridge = (
            adagrad_sgd_ridge(
                x_train, y_train, 0.01, learning_rate=lr, epochs=60, mb_size=5
            )
        )
        print("OLS: Learning rate " + str(lr) + ": \n" + str(beta_OLS_sgd_adagrad))
        print("...and converged at " + str(iterations_before_convergence_ols))
        print("Ridge: Learning rate " + str(lr) + ": \n" + str(beta_ridge_sgd_adagrad))
        print("...and converged at " + str(iterations_before_convergence_ridge))

if 8 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Stochastic gradient descent with Adagrad and momentum \n"
        + "========================================================================================"
    )
    for mom in momentum:
        print("Momentum: " + str(mom))
        for lr in learning_rates:
            beta_OLS_sgdm_adagrad, iterations_before_convergence_ols, epoch_ols = (
                adagrad_sgd_momentum_ols(
                    x_train,
                    y_train,
                    learning_rate=lr,
                    epochs=60,
                    mb_size=5,
                    momentum=mom,
                )
            )
            (
                beta_ridge_sgdm_adagrad,
                iterations_before_convergence_ridge,
                epoch_ridge,
            ) = adagrad_sgd_momentum_ridge(
                x_train,
                y_train,
                0.01,
                learning_rate=lr,
                epochs=60,
                mb_size=5,
                momentum=mom,
            )
            print("OLS: Learning rate " + str(lr) + ": \n" + str(beta_OLS_sgdm_adagrad))
            print("...and converged at " + str(iterations_before_convergence_ols))
            print(" in epoch " + str(epoch_ols))
            print(
                "Ridge: Learning rate "
                + str(lr)
                + ": \n"
                + str(beta_ridge_sgdm_adagrad)
            )
            print("...and converged at " + str(iterations_before_convergence_ridge))
            print(" in epoch " + str(epoch_ridge))

if 9 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Plain gradient descent with RMSProp \n"
        + "========================================================================================"
    )
    for lr in learning_rates:
        beta_OLS_pgd_adagrad, iterations_before_convergence_ols = rms_pgd_ols(
            x_train, y_train, learning_rate=lr, iterations=1000
        )
        beta_ridge_pgd_adagrad, iterations_before_convergence_ridge = rms_pgd_ridge(
            x_train, y_train, 0.01, learning_rate=lr, iterations=1000
        )
        print("OLS: Learning rate " + str(lr) + ": \n" + str(beta_OLS_pgd_adagrad))
        print("...and converged at " + str(iterations_before_convergence_ols))
        print("Ridge: Learning rate " + str(lr) + ": \n" + str(beta_ridge_pgd_adagrad))
        print("...and converged at " + str(iterations_before_convergence_ridge))
if 10 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Gradient descent with RMSProp and momentum \n"
        + "========================================================================================"
    )
    for mom in momentum:
        print("Momentum: " + str(mom))
        for lr in learning_rates:
            beta_OLS_gdm_adagrad, iterations_before_convergence_ols = rms_gdm_ols(
                x_train, y_train, learning_rate=lr, momentum=mom, iterations=1000
            )
            beta_ridge_gdm_adagrad, iterations_before_convergence_ridge = rms_gdm_ridge(
                x_train, y_train, 0.01, learning_rate=lr, momentum=mom, iterations=1000
            )
            print("OLS: Learning rate " + str(lr) + ": \n" + str(beta_OLS_gdm_adagrad))
            print("...and converged at " + str(iterations_before_convergence_ols))
            print(
                "Ridge: Learning rate " + str(lr) + ": \n" + str(beta_ridge_gdm_adagrad)
            )
            print("...and converged at " + str(iterations_before_convergence_ridge))
if 11 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Stochastic gradient descent with RMSProp \n"
        + "========================================================================================"
    )
    for lr in learning_rates:
        beta_OLS_sgd_adagrad, iterations_before_convergence_ols, epoch_ols = (
            rms_sgd_ols(x_train, y_train, learning_rate=lr, epochs=60, mb_size=5)
        )
        beta_ridge_sgd_adagrad, iterations_before_convergence_ridge, epoch_ridge = (
            rms_sgd_ridge(
                x_train, y_train, 0.01, learning_rate=lr, epochs=60, mb_size=5
            )
        )
        print("OLS: Learning rate " + str(lr) + ": \n" + str(beta_OLS_sgd_adagrad))
        print("...and converged at " + str(iterations_before_convergence_ols))
        print("Ridge: Learning rate " + str(lr) + ": \n" + str(beta_ridge_sgd_adagrad))
        print("...and converged at " + str(iterations_before_convergence_ridge))

if 12 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Stochastic gradient descent with RMSProp and momentum \n"
        + "========================================================================================"
    )
    for mom in momentum:
        print("Momentum: " + str(mom))
        for lr in learning_rates:
            beta_OLS_sgdm_adagrad, iterations_before_convergence_ols, epoch_ols = (
                rms_sgd_momentum_ols(
                    x_train,
                    y_train,
                    learning_rate=lr,
                    epochs=60,
                    mb_size=5,
                    momentum=mom,
                )
            )
            (
                beta_ridge_sgdm_adagrad,
                iterations_before_convergence_ridge,
                epoch_ridge,
            ) = rms_sgd_momentum_ridge(
                x_train,
                y_train,
                0.01,
                learning_rate=lr,
                epochs=60,
                mb_size=5,
                momentum=mom,
            )
            print("OLS: Learning rate " + str(lr) + ": \n" + str(beta_OLS_sgdm_adagrad))
            print("...and converged at " + str(iterations_before_convergence_ols))
            print(" in epoch " + str(epoch_ols))
            print(
                "Ridge: Learning rate "
                + str(lr)
                + ": \n"
                + str(beta_ridge_sgdm_adagrad)
            )
            print("...and converged at " + str(iterations_before_convergence_ridge))
            print(" in epoch " + str(epoch_ridge))

if 13 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Plain gradient descent with Adam \n"
        + "========================================================================================"
    )
    for lr in learning_rates:
        beta_OLS_pgd_adagrad, iterations_before_convergence_ols = adam_pgd_ols(
            x_train, y_train, learning_rate=lr, iterations=1000
        )
        beta_ridge_pgd_adagrad, iterations_before_convergence_ridge = adam_pgd_ridge(
            x_train, y_train, 0.01, learning_rate=lr, iterations=1000
        )
        print("OLS: Learning rate " + str(lr) + ": \n" + str(beta_OLS_pgd_adagrad))
        print("...and converged at " + str(iterations_before_convergence_ols))
        print("Ridge: Learning rate " + str(lr) + ": \n" + str(beta_ridge_pgd_adagrad))
        print("...and converged at " + str(iterations_before_convergence_ridge))

if 14 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Gradient descent with Adam and momentum \n"
        + "========================================================================================"
    )
    for mom in momentum:
        print("Momentum: " + str(mom))
        for lr in learning_rates:
            beta_OLS_gdm_adagrad, iterations_before_convergence_ols = adam_gdm_ols(
                x_train, y_train, learning_rate=lr, momentum=mom, iterations=1000
            )
            beta_ridge_gdm_adagrad, iterations_before_convergence_ridge = (
                adam_gdm_ridge(
                    x_train,
                    y_train,
                    0.01,
                    learning_rate=lr,
                    momentum=mom,
                    iterations=1000,
                )
            )
            print("OLS: Learning rate " + str(lr) + ": \n" + str(beta_OLS_gdm_adagrad))
            print("...and converged at " + str(iterations_before_convergence_ols))
            print(
                "Ridge: Learning rate " + str(lr) + ": \n" + str(beta_ridge_gdm_adagrad)
            )
            print("...and converged at " + str(iterations_before_convergence_ridge))

if 15 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Stochastic gradient descent with Adam \n"
        + "========================================================================================"
    )
    for lr in learning_rates:
        beta_OLS_sgd_adagrad, iterations_before_convergence_ols, epoch_ols = (
            adam_sgd_ols(x_train, y_train, learning_rate=lr, epochs=60, mb_size=5)
        )
        beta_ridge_sgd_adagrad, iterations_before_convergence_ridge, epoch_ridge = (
            adam_sgd_ridge(
                x_train, y_train, 0.01, learning_rate=lr, epochs=60, mb_size=5
            )
        )
        print("OLS: Learning rate " + str(lr) + ": \n" + str(beta_OLS_sgd_adagrad))
        print("...and converged at " + str(iterations_before_convergence_ols))
        print("Ridge: Learning rate " + str(lr) + ": \n" + str(beta_ridge_sgd_adagrad))
        print("...and converged at " + str(iterations_before_convergence_ridge))

if 16 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Stochastic gradient descent with Adam and momentum \n"
        + "========================================================================================"
    )
    for mom in momentum:
        print("Momentum: " + str(mom))
        for lr in learning_rates:
            beta_OLS_sgdm_adagrad, iterations_before_convergence_ols, epoch_ols = (
                adam_sgd_momentum_ols(
                    x_train,
                    y_train,
                    learning_rate=lr,
                    epochs=60,
                    mb_size=5,
                    momentum=mom,
                )
            )
            (
                beta_ridge_sgdm_adagrad,
                iterations_before_convergence_ridge,
                epoch_ridge,
            ) = adam_sgd_momentum_ridge(
                x_train,
                y_train,
                0.01,
                learning_rate=lr,
                epochs=60,
                mb_size=5,
                momentum=mom,
            )
            print("OLS: Learning rate " + str(lr) + ": \n" + str(beta_OLS_sgdm_adagrad))
            print("...and converged at " + str(iterations_before_convergence_ols))
            print(" in epoch " + str(epoch_ols))
            print(
                "Ridge: Learning rate "
                + str(lr)
                + ": \n"
                + str(beta_ridge_sgdm_adagrad)
            )
            print("...and converged at " + str(iterations_before_convergence_ridge))
            print(" in epoch " + str(epoch_ridge))

print(
    "========================================================================================\n"
    + "Bheta values with matrix inversion \n"
    + "========================================================================================"
)
print("Bheta for OLS with matrix inversion: " + str(beta_OLS_minv))
print("Bheta for ridge regression with matrix inversion: " + str(beta_ridge_minv))
