import csv
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys, os
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from partB_to_D.neural_network import (
    eval_classification_ffnn,
    test_classification_ffnn,
)
from logistic_regression import Logistic
from partB_to_D.methods import add_to_compare_log_training_score_csv
import seaborn as sns


def eval_logistic_regression(X_train, y_train, save_plot=False):
    mb_sizes = [5, 10, 15, 20]
    lambda_values = [0.001, 0.01, 0.1, 0.5]
    epochs = [100, 200, 500]
    current_score = 0
    current_mb_size = None
    current_lmb = None
    for epoch in epochs:
        accuracy_matrix = np.zeros((len(mb_sizes), len(lambda_values)))
        for i, mbs in enumerate(mb_sizes):
            for j, lmb in enumerate(lambda_values):
                model = Logistic(epochs=1000, mb_size=mbs, lmb=lmb)
                model.sgd(X_train, y_train)
                add_to_compare_log_training_score_csv(
                    "log reg", None, lmb, mbs, model.training_score, epochs=epoch
                )
                if model.training_score > current_score:
                    current_score = model.training_score
                    current_mb_size = mbs
                    current_lmb = lmb
                accuracy_matrix[i, j] = model.training_score

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                accuracy_matrix,
                annot=True,
                fmt=".4f",
                xticklabels=mb_sizes,
                yticklabels=lambda_values,
                cmap="YlGnBu",
                cbar_kws={"label": "Accuracy"},
            )
            plt.xlabel("Mini batch sizes")
            plt.ylabel("Lambda (lmb)")
            plt.title(f"Accuracy Heatmap for epoch {epoch}")
            if save_plot:
                plt.savefig(f"partE/Figures/log_reg_epoch{epoch}_heatmap.png")
            else:
                plt.show()
            plt.close()
    return current_mb_size, current_lmb


def test_logistic_regression(X_train, X_test, y_train, y_test, mb_size, lmb):
    model = Logistic(epochs=1000, mb_size=mb_size, lmb=lmb)
    model.sgd(X_train, y_train)
    return model.score(X_test, y_test)


def test_scikits_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train.reshape(-1))
    return model.score(X_test, y_test.reshape(-1))


data = load_breast_cancer()

X = data.data
y = data.target.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
"""
test_classification_ffnn trains the nn with this learning rate and returns the test score 
"""
optimal_mb_size, optimal_epoch, optimal_lmb, optimal_lr = eval_classification_ffnn(
    add_to_csv=True, save_plot=True
)
ffnn_score = test_classification_ffnn(
    X_train,
    y_train,
    X_test,
    y_test,
    optimal_lmb,
    optimal_mb_size,
    optimal_lr,
    optimal_epoch,
)
print("FFNN get the test score: " + str(ffnn_score))
"""
Since Logistic regression must be evaluated first to find the optimal learning rate.
eval_logistic regression will use the same cost function and the same array of learning rates
"""
optimal_mb_size, optimal_lmb = eval_logistic_regression(
    X_train, y_train, save_plot=True
)
log_score = test_logistic_regression(
    X_train, X_test, y_train, y_test, optimal_mb_size, optimal_lmb
)
print("Logistic regression get the test score: " + str(log_score))

scikit_log_score = test_scikits_logistic_regression(X_train, X_test, y_train, y_test)
print("Scikit's logistic regression get the test score: " + str(scikit_log_score))
