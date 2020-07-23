import warnings

warnings.filterwarnings("ignore")

import argparse

import numpy as np

import random
import time

from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score,
    precision_score,
    confusion_matrix,
    recall_score,
    accuracy_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from preprocess import prepare_data, load_csv

from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path", type=str, default="dataset/NDTX.csv", help="path of csv file"
    )

    parser.add_argument(
        "--trading_days", type=int, default=30, help="Number of trading days"
    )

    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        help="the kernel for SVM",
        choices=["linear", "rbf", "poly", "custom", "cobb-douglas"],
    )

    parser.add_argument(
        "--degree", type=int, default=3, help="value of p in polynomial/custom kernel"
    )

    parser.add_argument(
        "--C",
        nargs="+",
        default=[10 ** i for i in range(-100, 101)],
        help="the regularisation parameter for SVM",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="the inner product coefficient in polynomial kernel",
    )

    parser.add_argument(
        "--coef0", type=float, default=0.0, help="coefficient for polynomial kernel"
    )

    parser.add_argument(
        "--train_test_ratio", type=float, default=0.75, help="fraction of train samples"
    )

    parser.add_argument(
        "--folds", type=int, default=5, help="k in k-fold cross validation"
    )

    args = parser.parse_args()
    path = args.path
    trading_days = args.trading_days
    kernel = args.kernel
    degree = args.degree
    C = C = [float(i) for i in args.C]
    gamma = args.gamma

    if kernel == "poly":
        gamma = 1.0

    if kernel == "rbf":
        gamma = "scale"

    coef0 = args.coef0
    train_test_ratio = args.train_test_ratio
    folds = args.folds

    print("Details: ")
    print("Extracting data from: " + str(path))
    print("Trading Days: " + str(trading_days))

    if kernel == "poly":
        assert (
            gamma == 1.0
        ), "Polynomial should have gamma=1.0, otherwise it is Cobb-Douglas Kernel"

        print("Kernel: polynomial")
        print("Degree: " + str(degree))

    elif kernel == "cobb-douglas":
        assert (
            gamma != 1.0
        ), "Cobb-Douglas should have gamma!=1.0, otherwise it is Polynomial Kernel"
        print("Kernel: cobb-douglas")
        print("Gamma: " + str(gamma))
        print("Degree: " + str(degree))
        kernel = "poly"

    elif kernel == "custom":
        print("Kernel: custom")
        print("Degree: " + str(degree))

    else:
        print("Kernel: " + str(kernel))

    print("Regularisation Parameter, C: " + str(C))

    # define the custom kernels
    def poly_cobb_kernel(X, Y):
        return gamma * (np.dot(X, Y.T)) ** degree

    def custom_kernel(X, Y):
        return 1 / (1 + np.dot(X, Y.T) ** degree)

    # load the dataset
    df = load_csv(path)
    data = prepare_data(data_f=df, horizon=trading_days, alpha=0.9,)

    # remove the output from the input
    features = [x for x in data.columns if x not in ["gain"]]

    dataA = np.array_split(data[features], 1)

    i = 0
    features = [x for x in data.columns if x not in ["gain", "pred"]]
    X = dataA[i][features]
    y = dataA[i]["pred"]

    X_train = X[: int(train_test_ratio * len(X))]
    y_train = y[: int(train_test_ratio * len(y))]

    X_test = X[int(train_test_ratio * len(X)) :]
    y_test = y[int(train_test_ratio * len(y)) :]

    if kernel == "custom":
        clf = make_pipeline(
            StandardScaler(),
            SVC(kernel=custom_kernel, class_weight="balanced", cache_size=100000),
        )

    elif kernel == "poly":
        clf = make_pipeline(
            StandardScaler(),
            SVC(kernel=poly_cobb_kernel, class_weight="balanced", cache_size=100000),
        )

    else:
        clf = make_pipeline(
            StandardScaler(),
            SVC(
                kernel=kernel,
                degree=degree,
                coef0=coef0,
                gamma=gamma,
                class_weight="balanced",
                cache_size=100000,
            ),
        )

    print(clf.get_params().keys())
    param_grid = {"svc__C": C}
    grid = GridSearchCV(clf, param_grid, cv=folds, refit=True, verbose=2, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("\n")

    print("Performed Grid Search on:\n")
    print("C: " + str(C))
    print("gamma: " + str(gamma))

    print("\n")

    print("The best choice of hyperparameters:")
    print(str(grid.best_estimator_) + "\n")

    y_train_pred = grid.predict(X_train)
    print("The results on the train set: ")
    y_train_label = np.array(y_train)
    train_set_acc = accuracy_score(y_train_label, y_train_pred)
    print(train_set_acc)

    print("\n")

    y_test_pred = grid.predict(X_test)
    print("The results on the test set: ")
    y_test_label = np.array(y_test)
    test_set_acc = accuracy_score(y_test_label, y_test_pred)
    print(test_set_acc)

    print(clf.get_params().keys())


if __name__ == "__main__":
    main()
