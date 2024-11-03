from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys, os
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from partB_to_D.neural_network import (
    test_classification_ffnn,
)
from logistic_regression import Logistic


def eval_logistic_regression(X_train, y_train):
    mb_sizes = [5, 10, 15, 20]
    current_score = 1000  # random large number
    current_mb_size = None
    for mbs in mb_sizes:
        model = Logistic()
        model.sgd(X_train, y_train, epochs=1000, mb_size=mbs)
        if model.training_score < current_score:
            current_score = model.training_score
            current_mb_size = mbs
    return current_mb_size


def test_logistic_regression(X_train, X_test, y_train, y_test, mb_size):
    model = Logistic()
    model.sgd(X_train, y_train, epochs=1000, mb_size=mb_size)
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
eval_classification_ffnn tells us that the optimal learning rate is 0.1
test_classification_ffnn trains the nn with this learning rate and returns the test score 
"""
ffnn_score = test_classification_ffnn(X_train, y_train, X_test, y_test, 0.1)
print("FFNN get the test score: " + str(ffnn_score))
"""
Since Logistic regression must be evaluated first to find the optimal learning rate.
eval_logistic regression will use the same cost function and the same array of learning rates
"""
optimal_mb_size = eval_logistic_regression(X_train, y_train)
log_score = test_logistic_regression(X_train, X_test, y_train, y_test, optimal_mb_size)
print("Logistic regression get the test score: " + str(log_score))

scikit_log_score = test_scikits_logistic_regression(X_train, X_test, y_train, y_test)
print("Scikit's logistic regression get the test score: " + str(scikit_log_score))
