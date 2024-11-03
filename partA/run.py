import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import StandardScaler
from gradient_methods import (
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
from partA.OptimalParameters import OptimalParameters

np.random.seed(100)

n = 100
x = 2 * np.random.rand(n, 1)
x_scaler = StandardScaler()
x_scaled = x_scaler.fit_transform(x)
y = x + x**2
X = np.c_[np.ones((n, 1)), x_scaled]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

"""
The following numbers represent:
1. Plain gradient descent
2. Gradient descent with momentum
3. Stochastic gradient descent
3.5. Stochastic gradient descent with momentum
4. Plain gradient descent with Adagrad
4.25. Gradient descent with Adagrad and momentum
4.5. Stochastic gradient descent with Adagrad
4.75. Stochastic gradient descent with Adagrad and momentum
5. Plain gradient descent with RMSProp
5.125. Gradient descent with RMSProp and momentum
5.25. Stochastic gradient descent with RMSProp
5.375. Stochastic gradient descent with RMSProp and momentum
5.5. Plain gradient descent with Adam
5.625. Gradient descent with Adam and momentum
5.75. Stochastic gradient descent with Adam
5.875. Stochastic gradient descent with Adam and momentum
"""
SHOULD_RUN = [
    1,
    2,
    3,
    3.5,
    4,
    4.25,
    4.5,
    4.75,
    5,
    5.125,
    5.25,
    5.375,
    5.5,
    5.625,
    5.75,
    5.875,
]
learning_rates = [0.01, 0.1, 0.2, 0.5]
momentum = [0.1, 0.2, 0.3, 0.4]
mb_sizes = [5, 10, 15]
epochs = [51, 75]
lambdas = [
    0.01,
    0.05,
    0.1,
    0.5,
]

if 1 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Plain gradient descent \n"
        + "========================================================================================"
    )
    osl_parameters = OptimalParameters(
        convergence_at_iteration=1001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="pgd_osl",
    )
    ridge_parameters = OptimalParameters(
        convergence_at_iteration=1001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="pdg_ridge",
    )

    for lr in learning_rates:
        beta_OLS_gd, iterations_before_convergence_ols, mse = gradient_descent_ols(
            x_train, y_train, learning_rate=lr, iterations=1000
        )
        osl_parameters.parameter_comparison(
            convergence=iterations_before_convergence_ols, lr=lr, mse=mse
        )
        for lmb in lambdas:
            beta_ridge_gd, iterations_before_convergence_ridge, mse = (
                gradient_descent_ridge(
                    x_train,
                    y_train,
                    lmb,
                    learning_rate=lr,
                    iterations=1000,
                )
            )
            ridge_parameters.parameter_comparison(
                convergence=iterations_before_convergence_ridge, lr=lr, mse=mse, lmb=lmb
            )
    print("Optimal parameters")
    osl_parameters.print_optimal_parameters("OLS")
    print(
        "----------------------------------------------------------------------------------------"
    )
    ridge_parameters.print_optimal_parameters("Ridge")

if 2 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Gradient descent with momentum \n"
        + "========================================================================================"
    )
    ols_parameters = OptimalParameters(
        convergence_at_iteration=1001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="gdm_ols",
    )
    ridge_parameters = OptimalParameters(
        convergence_at_iteration=1001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="gdm_ridge",
    )

    for mom in momentum:
        for lr in learning_rates:
            beta_OLS_gd, iterations_before_convergence_ols, mse = (
                gradient_descent_momentum_ols(
                    x_train,
                    y_train,
                    learning_rate=lr,
                    iterations=1000,
                    momentum=mom,
                )
            )
            ols_parameters.parameter_comparison(
                convergence=iterations_before_convergence_ols, lr=lr, mom=mom, mse=mse
            )
            for lmb in lambdas:
                beta_ridge_gd, iterations_before_convergence_ridge, mse = (
                    gradient_descent_momentum_ridge(
                        x_train,
                        y_train,
                        lmb,
                        learning_rate=lr,
                        iterations=1000,
                        momentum=mom,
                    )
                )
                ridge_parameters.parameter_comparison(
                    convergence=iterations_before_convergence_ridge,
                    lr=lr,
                    mom=mom,
                    mse=mse,
                    lmb=lmb,
                )
    print("Optimal parameters")
    ols_parameters.print_optimal_parameters("OLS")
    print(
        "----------------------------------------------------------------------------------------"
    )
    ridge_parameters.print_optimal_parameters("Ridge")

if 3 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Stochastic gradient descent \n"
        + "========================================================================================"
    )
    ols_parameters = OptimalParameters(
        convergence_at_iteration=10001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="sgd_ols",
    )
    ridge_parameters = OptimalParameters(
        convergence_at_iteration=10001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="sgd_ridge",
    )

    for epoch in epochs:
        for mb in mb_sizes:
            beta_OLS_gd, iterations_before_convergence_ols, epoch_ols, mse = sgd_ols(
                x_train,
                y_train,
                epochs=epoch,
                mb_size=mb,
            )
            ols_parameters.parameter_comparison(
                convergence=iterations_before_convergence_ols,
                epoch=epoch_ols,
                mb_size=mb,
                mse=mse,
            )
            for lmb in lambdas:
                (
                    beta_ridge_gd,
                    iterations_before_convergence_ridge,
                    epoch_ridge,
                    mse,
                ) = sgd_ridge(
                    x_train,
                    y_train,
                    lmb,
                    epochs=epoch,
                    mb_size=mb,
                )
                ridge_parameters.parameter_comparison(
                    convergence=iterations_before_convergence_ridge,
                    epoch=epoch_ridge,
                    mb_size=mb,
                    mse=mse,
                    lmb=lmb,
                )
    print("Optimal parameters")
    ols_parameters.print_optimal_parameters("OLS")
    print(
        "----------------------------------------------------------------------------------------"
    )
    ridge_parameters.print_optimal_parameters("Ridge")
if 3.5 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Stochastic gradient descent with momentum \n"
        + "========================================================================================"
    )
    ols_parameters = OptimalParameters(
        convergence_at_iteration=10001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="sgdm_ols",
    )
    ridge_parameters = OptimalParameters(
        convergence_at_iteration=10001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="sgdm_ridge",
    )

    for epoch in epochs:
        for mb in mb_sizes:
            for mom in momentum:
                beta_OLS_gd, iterations_before_convergence_ols, epoch_ols, mse = (
                    sgd_momentum_ols(
                        x_train,
                        y_train,
                        epochs=epoch,
                        mb_size=mb,
                        momentum=mom,
                    )
                )
                ols_parameters.parameter_comparison(
                    convergence=iterations_before_convergence_ols,
                    epoch=epoch_ols,
                    mb_size=mb,
                    mse=mse,
                )
                for lmb in lambdas:
                    (
                        beta_ridge_gd,
                        iterations_before_convergence_ridge,
                        epoch_ridge,
                        mse,
                    ) = sgd_momentum_ridge(
                        x_train,
                        y_train,
                        lmb,
                        epochs=epoch,
                        mb_size=mb,
                        momentum=mom,
                    )
                    ridge_parameters.parameter_comparison(
                        convergence=iterations_before_convergence_ridge,
                        epoch=epoch_ridge,
                        mb_size=mb,
                        mse=mse,
                        lmb=lmb,
                    )
    print("Optimal parameters")
    ols_parameters.print_optimal_parameters("OLS")
    print(
        "----------------------------------------------------------------------------------------"
    )
    ridge_parameters.print_optimal_parameters("Ridge")

if 4 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Plain gradient descent with Adagrad \n"
        + "========================================================================================"
    )
    osl_parameters = OptimalParameters(
        convergence_at_iteration=1001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="pgd_ada_ols",
    )
    ridge_parameters = OptimalParameters(
        convergence_at_iteration=1001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="pgd_ada_ols",
    )

    for lr in learning_rates:
        beta_OLS_gd, iterations_before_convergence_ols, mse = adagrad_pgd_ols(
            x_train,
            y_train,
            learning_rate=lr,
            iterations=1000,
        )
        osl_parameters.parameter_comparison(
            convergence=iterations_before_convergence_ols, lr=lr, mse=mse
        )
        for lmb in lambdas:
            beta_ridge_gd, iterations_before_convergence_ridge, mse = adagrad_pgd_ridge(
                x_train,
                y_train,
                lmb,
                learning_rate=lr,
                iterations=1000,
            )
            ridge_parameters.parameter_comparison(
                convergence=iterations_before_convergence_ridge, lr=lr, mse=mse, lmb=lmb
            )
    print("Optimal parameters")
    osl_parameters.print_optimal_parameters("OLS")
    print(
        "----------------------------------------------------------------------------------------"
    )
    ridge_parameters.print_optimal_parameters("Ridge")

if 4.25 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Gradient descent with Adagrad and momentum \n"
        + "========================================================================================"
    )
    ols_parameters = OptimalParameters(
        convergence_at_iteration=1001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="gdm_ada",
    )
    ridge_parameters = OptimalParameters(
        convergence_at_iteration=1001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="gdm_ridge",
    )

    for mom in momentum:
        for lr in learning_rates:
            beta_OLS_gd, iterations_before_convergence_ols, mse = adagrad_gdm_ols(
                x_train,
                y_train,
                learning_rate=lr,
                iterations=1000,
                momentum=mom,
            )
            ols_parameters.parameter_comparison(
                convergence=iterations_before_convergence_ols, lr=lr, mom=mom, mse=mse
            )
            for lmb in lambdas:
                beta_ridge_gd, iterations_before_convergence_ridge, mse = (
                    adagrad_gdm_ridge(
                        x_train,
                        y_train,
                        lmb,
                        learning_rate=lr,
                        iterations=1000,
                        momentum=mom,
                    )
                )
                ridge_parameters.parameter_comparison(
                    convergence=iterations_before_convergence_ridge,
                    lr=lr,
                    mom=mom,
                    mse=mse,
                    lmb=lmb,
                )
    print("Optimal parameters")
    ols_parameters.print_optimal_parameters("OLS")
    print(
        "----------------------------------------------------------------------------------------"
    )
    ridge_parameters.print_optimal_parameters("Ridge")

if 4.5 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Stochastic gradient descent with Adagrad \n"
        + "========================================================================================"
    )
    ols_parameters = OptimalParameters(
        convergence_at_iteration=10001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="sgd_ada_ols",
    )
    ridge_parameters = OptimalParameters(
        convergence_at_iteration=10001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="sgd_ada_ridge",
    )
    for epoch in epochs:
        for mb in mb_sizes:
            for lr in learning_rates:
                beta_OLS_gd, iterations_before_convergence_ols, epoch_ols, mse = (
                    adagrad_sgd_ols(
                        x_train,
                        y_train,
                        learning_rate=lr,
                        epochs=epoch,
                        mb_size=mb,
                    )
                )
                ols_parameters.parameter_comparison(
                    convergence=iterations_before_convergence_ols,
                    lr=lr,
                    epoch=epoch_ols,
                    mb_size=mb,
                    mse=mse,
                )
                for lmb in lambdas:
                    (
                        beta_ridge_gd,
                        iterations_before_convergence_ridge,
                        epoch_ridge,
                        mse,
                    ) = adagrad_sgd_ridge(
                        x_train,
                        y_train,
                        lmb,
                        learning_rate=lr,
                        epochs=epoch,
                        mb_size=mb,
                    )
                    ridge_parameters.parameter_comparison(
                        convergence=iterations_before_convergence_ridge,
                        lr=lr,
                        epoch=epoch_ridge,
                        mb_size=mb,
                        mse=mse,
                        lmb=lmb,
                    )
    print("Optimal parameters")
    ols_parameters.print_optimal_parameters("OLS")
    print(
        "----------------------------------------------------------------------------------------"
    )
    ridge_parameters.print_optimal_parameters("Ridge")

if 4.75 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Stochastic gradient descent with Adagrad and momentum \n"
        + "========================================================================================"
    )
    ols_parameters = OptimalParameters(
        convergence_at_iteration=10001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="sgdm_ada_ols",
    )
    ridge_parameters = OptimalParameters(
        convergence_at_iteration=10001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="sgdm_ada_ridge",
    )

    for epoch in epochs:
        for mb in mb_sizes:
            for mom in momentum:
                for lr in learning_rates:
                    beta_OLS_gd, iterations_before_convergence_ols, epoch_ols, mse = (
                        adagrad_sgd_momentum_ols(
                            x_train,
                            y_train,
                            learning_rate=lr,
                            epochs=epoch,
                            mb_size=mb,
                            momentum=mom,
                        )
                    )
                    ols_parameters.parameter_comparison(
                        convergence=iterations_before_convergence_ols,
                        lr=lr,
                        epoch=epoch_ols,
                        mb_size=mb,
                        mse=mse,
                    )
                    for lmb in lambdas:
                        (
                            beta_ridge_gd,
                            iterations_before_convergence_ridge,
                            epoch_ridge,
                            mse,
                        ) = adagrad_sgd_momentum_ridge(
                            x_train,
                            y_train,
                            lmb,
                            learning_rate=lr,
                            epochs=epoch,
                            mb_size=mb,
                            momentum=mom,
                        )
                        ridge_parameters.parameter_comparison(
                            convergence=iterations_before_convergence_ridge,
                            lr=lr,
                            epoch=epoch_ridge,
                            mb_size=mb,
                            mse=mse,
                            lmb=lmb,
                        )
    print("Optimal parameters")
    ols_parameters.print_optimal_parameters("OLS")
    print(
        "----------------------------------------------------------------------------------------"
    )
    ridge_parameters.print_optimal_parameters("Ridge")

if 5 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Plain gradient descent with RMSProp \n"
        + "========================================================================================"
    )
    osl_parameters = OptimalParameters(
        convergence_at_iteration=1001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="pgd_rms_ols",
    )
    ridge_parameters = OptimalParameters(
        convergence_at_iteration=1001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="pgd_rms_ridge",
    )
    for lr in learning_rates:
        beta_OLS_gd, iterations_before_convergence_ols, mse = rms_pgd_ols(
            x_train, y_train, learning_rate=lr, iterations=1000
        )
        osl_parameters.parameter_comparison(
            convergence=iterations_before_convergence_ols, lr=lr, mse=mse
        )
        for lmb in lambdas:
            beta_ridge_gd, iterations_before_convergence_ridge, mse = rms_pgd_ridge(
                x_train,
                y_train,
                lmb,
                learning_rate=lr,
                iterations=1000,
            )
            ridge_parameters.parameter_comparison(
                convergence=iterations_before_convergence_ridge, lr=lr, mse=mse, lmb=lmb
            )
    print("Optimal parameters")
    osl_parameters.print_optimal_parameters("OLS")
    print(
        "----------------------------------------------------------------------------------------"
    )
    ridge_parameters.print_optimal_parameters("Ridge")

if 5.125 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Gradient descent with RMSProp and momentum \n"
        + "========================================================================================"
    )
    ols_parameters = OptimalParameters(
        convergence_at_iteration=1001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="gdm_rms_ols",
    )
    ridge_parameters = OptimalParameters(
        convergence_at_iteration=1001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="gdm_rms_ridge",
    )
    for mom in momentum:
        for lr in learning_rates:
            beta_OLS_gd, iterations_before_convergence_ols, mse = rms_gdm_ols(
                x_train,
                y_train,
                learning_rate=lr,
                iterations=1000,
                momentum=mom,
            )
            ols_parameters.parameter_comparison(
                convergence=iterations_before_convergence_ols, lr=lr, mom=mom, mse=mse
            )
            for lmb in lambdas:
                beta_ridge_gd, iterations_before_convergence_ridge, mse = rms_gdm_ridge(
                    x_train,
                    y_train,
                    lmb,
                    learning_rate=lr,
                    iterations=1000,
                    momentum=mom,
                )
                ridge_parameters.parameter_comparison(
                    convergence=iterations_before_convergence_ridge,
                    lr=lr,
                    mom=mom,
                    mse=mse,
                    lmb=lmb,
                )
    print("Optimal parameters")
    ols_parameters.print_optimal_parameters("OLS")
    print(
        "----------------------------------------------------------------------------------------"
    )
    ridge_parameters.print_optimal_parameters("Ridge")

if 5.25 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Stochastic gradient descent with RMSProp \n"
        + "========================================================================================"
    )
    ols_parameters = OptimalParameters(
        convergence_at_iteration=10001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="sgd_rms_ols",
    )
    ridge_parameters = OptimalParameters(
        convergence_at_iteration=10001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="sgd_rms_ridge",
    )

    for epoch in epochs:
        for mb in mb_sizes:
            for lr in learning_rates:
                beta_OLS_gd, iterations_before_convergence_ols, epoch_ols, mse = (
                    rms_sgd_ols(
                        x_train,
                        y_train,
                        learning_rate=lr,
                        epochs=epoch,
                        mb_size=mb,
                    )
                )
                ols_parameters.parameter_comparison(
                    convergence=iterations_before_convergence_ols,
                    lr=lr,
                    epoch=epoch_ols,
                    mb_size=mb,
                    mse=mse,
                )
                for lmb in lambdas:
                    (
                        beta_ridge_gd,
                        iterations_before_convergence_ridge,
                        epoch_ridge,
                        mse,
                    ) = rms_sgd_ridge(
                        x_train,
                        y_train,
                        lmb,
                        learning_rate=lr,
                        epochs=epoch,
                        mb_size=mb,
                    )
                    ridge_parameters.parameter_comparison(
                        convergence=iterations_before_convergence_ridge,
                        lr=lr,
                        epoch=epoch_ridge,
                        mb_size=mb,
                        mse=mse,
                        lmb=lmb,
                    )
    print("Optimal parameters")
    ols_parameters.print_optimal_parameters("OLS")
    print(
        "----------------------------------------------------------------------------------------"
    )
    ridge_parameters.print_optimal_parameters("Ridge")

if 5.375 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Stochastic gradient descent with RMSProp and momentum \n"
        + "========================================================================================"
    )
    ols_parameters = OptimalParameters(
        convergence_at_iteration=10001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="sgdm_rms_ols",
    )
    ridge_parameters = OptimalParameters(
        convergence_at_iteration=10001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="sgdm_rms_ridge",
    )
    for epoch in epochs:
        for mb in mb_sizes:
            for mom in momentum:
                for lr in learning_rates:
                    beta_OLS_gd, iterations_before_convergence_ols, epoch_ols, mse = (
                        rms_sgd_momentum_ols(
                            x_train,
                            y_train,
                            learning_rate=lr,
                            epochs=epoch,
                            mb_size=mb,
                            momentum=mom,
                        )
                    )
                    ols_parameters.parameter_comparison(
                        convergence=iterations_before_convergence_ols,
                        lr=lr,
                        epoch=epoch_ols,
                        mb_size=mb,
                        mse=mse,
                    )
                    for lmb in lambdas:
                        (
                            beta_ridge_gd,
                            iterations_before_convergence_ridge,
                            epoch_ridge,
                            mse,
                        ) = rms_sgd_momentum_ridge(
                            x_train,
                            y_train,
                            lmb,
                            learning_rate=lr,
                            epochs=epoch,
                            mb_size=mb,
                            momentum=mom,
                        )
                        ridge_parameters.parameter_comparison(
                            convergence=iterations_before_convergence_ridge,
                            lr=lr,
                            epoch=epoch_ridge,
                            mb_size=mb,
                            mse=mse,
                            lmb=lmb,
                        )
    print("Optimal parameters")
    ols_parameters.print_optimal_parameters("OLS")
    print(
        "----------------------------------------------------------------------------------------"
    )
    ridge_parameters.print_optimal_parameters("Ridge")

if 5.5 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Plain gradient descent with Adam \n"
        + "========================================================================================"
    )
    osl_parameters = OptimalParameters(
        convergence_at_iteration=1001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="pgd_adam_ols",
    )
    ridge_parameters = OptimalParameters(
        convergence_at_iteration=1001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="pgd_adam_ridge",
    )
    for lr in learning_rates:
        beta_OLS_gd, iterations_before_convergence_ols, mse = adam_pgd_ols(
            x_train,
            y_train,
            learning_rate=lr,
            iterations=1000,
        )
        osl_parameters.parameter_comparison(
            convergence=iterations_before_convergence_ols, lr=lr, mse=mse
        )
        for lmb in lambdas:
            beta_ridge_gd, iterations_before_convergence_ridge, mse = adam_pgd_ridge(
                x_train,
                y_train,
                lmb,
                learning_rate=lr,
                iterations=1000,
            )
            ridge_parameters.parameter_comparison(
                convergence=iterations_before_convergence_ridge, lr=lr, mse=mse, lmb=lmb
            )
    print("Optimal parameters")
    osl_parameters.print_optimal_parameters("OLS")
    print(
        "----------------------------------------------------------------------------------------"
    )
    ridge_parameters.print_optimal_parameters("Ridge")

if 5.625 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Gradient descent with Adam and momentum \n"
        + "========================================================================================"
    )
    ols_parameters = OptimalParameters(
        convergence_at_iteration=1001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="gdm_adam_ols",
    )
    ridge_parameters = OptimalParameters(
        convergence_at_iteration=1001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="gdm_adam_ridge",
    )
    for mom in momentum:
        for lr in learning_rates:
            beta_OLS_gd, iterations_before_convergence_ols, mse = adam_gdm_ols(
                x_train,
                y_train,
                learning_rate=lr,
                iterations=1000,
                momentum=mom,
            )
            ols_parameters.parameter_comparison(
                convergence=iterations_before_convergence_ols, lr=lr, mom=mom, mse=mse
            )
            for lmb in lambdas:
                beta_ridge_gd, iterations_before_convergence_ridge, mse = (
                    adam_gdm_ridge(
                        x_train,
                        y_train,
                        lmb,
                        learning_rate=lr,
                        iterations=1000,
                        momentum=mom,
                    )
                )
                ridge_parameters.parameter_comparison(
                    convergence=iterations_before_convergence_ridge,
                    lr=lr,
                    mom=mom,
                    mse=mse,
                    lmb=lmb,
                )
    print("Optimal parameters")
    ols_parameters.print_optimal_parameters("OLS")
    print(
        "----------------------------------------------------------------------------------------"
    )
    ridge_parameters.print_optimal_parameters("Ridge")

if 5.75 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Stochastic gradient descent with Adam \n"
        + "========================================================================================"
    )
    ols_parameters = OptimalParameters(
        convergence_at_iteration=10001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="sgd_adam_ols",
    )
    ridge_parameters = OptimalParameters(
        convergence_at_iteration=10001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="sgd_adam_ridge",
    )

    for epoch in epochs:
        for mb in mb_sizes:
            for lr in learning_rates:
                beta_OLS_gd, iterations_before_convergence_ols, epoch_ols, mse = (
                    adam_sgd_ols(
                        x_train,
                        y_train,
                        learning_rate=lr,
                        epochs=epoch,
                        mb_size=mb,
                    )
                )
                ols_parameters.parameter_comparison(
                    convergence=iterations_before_convergence_ols,
                    lr=lr,
                    epoch=epoch_ols,
                    mb_size=mb,
                    mse=mse,
                )
                for lmb in lambdas:
                    (
                        beta_ridge_gd,
                        iterations_before_convergence_ridge,
                        epoch_ridge,
                        mse,
                    ) = adam_sgd_ridge(
                        x_train,
                        y_train,
                        lmb,
                        learning_rate=lr,
                        epochs=epoch,
                        mb_size=mb,
                    )
                    ridge_parameters.parameter_comparison(
                        convergence=iterations_before_convergence_ridge,
                        lr=lr,
                        epoch=epoch_ridge,
                        mb_size=mb,
                        mse=mse,
                        lmb=lmb,
                    )
    print("Optimal parameters")
    ols_parameters.print_optimal_parameters("OLS")
    print(
        "----------------------------------------------------------------------------------------"
    )
    ridge_parameters.print_optimal_parameters("Ridge")

if 5.875 in SHOULD_RUN:
    print(
        "========================================================================================\n"
        + "Stochastic gradient descent with Adam and momentum \n"
        + "========================================================================================"
    )
    ols_parameters = OptimalParameters(
        convergence_at_iteration=10001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="sgdm_adam_ols",
    )
    ridge_parameters = OptimalParameters(
        convergence_at_iteration=10001,
        learning_rate=None,
        momentum=None,
        epoch=None,
        mini_batch_size=None,
        function_name="sgdm_adam_ridge",
    )

    for epoch in epochs:
        for mb in mb_sizes:
            for mom in momentum:
                for lr in learning_rates:
                    beta_OLS_gd, iterations_before_convergence_ols, epoch_ols, mse = (
                        adam_sgd_momentum_ols(
                            x_train,
                            y_train,
                            learning_rate=lr,
                            epochs=epoch,
                            mb_size=mb,
                            momentum=mom,
                        )
                    )
                    ols_parameters.parameter_comparison(
                        convergence=iterations_before_convergence_ols,
                        lr=lr,
                        epoch=epoch_ols,
                        mb_size=mb,
                        mse=mse,
                    )
                    for lmb in lambdas:
                        (
                            beta_ridge_gd,
                            iterations_before_convergence_ridge,
                            epoch_ridge,
                            mse,
                        ) = adam_sgd_momentum_ridge(
                            x_train,
                            y_train,
                            lmb,
                            learning_rate=lr,
                            epochs=epoch,
                            mb_size=mb,
                            momentum=mom,
                        )
                        ridge_parameters.parameter_comparison(
                            convergence=iterations_before_convergence_ridge,
                            lr=lr,
                            epoch=epoch_ridge,
                            mb_size=mb,
                            mse=mse,
                            lmb=lmb,
                        )
    print("Optimal parameters")
    ols_parameters.print_optimal_parameters("OLS")
    print(
        "----------------------------------------------------------------------------------------"
    )
    ridge_parameters.print_optimal_parameters("Ridge")
