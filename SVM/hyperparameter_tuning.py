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
    classification_report,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

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

    # print("Regularisation Parameter, C: " + str(C))

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

    features = [x for x in data.columns if x not in ["gain", "pred"]]
    X = np.array(dataA[0][features])
    y = np.array(dataA[0]["pred"])

    tscv = TimeSeriesSplit(n_splits=folds)
    param_grid = {"svc__C": C}
    i = 0
    print("\n")
    for C in param_grid["svc__C"]:

        print("Performing Grid (Time Series) Search on:\n")
        print("C: " + str(C))
        print("gamma: " + str(gamma))

        for tr_index, val_index in tscv.split(X):

            print("Fold #" + str(i + 1))
            print("\n")

            X_train, X_test = X[tr_index], X[val_index]
            y_train, y_test = y[tr_index], y[val_index]

            if kernel == "custom":
                clf = make_pipeline(
                    StandardScaler(),
                    SVC(
                        kernel=custom_kernel,
                        C=C,
                        class_weight="balanced",
                        cache_size=100000,
                    ),
                )

            elif kernel == "poly":
                clf = make_pipeline(
                    StandardScaler(),
                    SVC(
                        kernel=poly_cobb_kernel,
                        C=C,
                        class_weight="balanced",
                        cache_size=100000,
                    ),
                )

            else:
                clf = make_pipeline(
                    StandardScaler(),
                    SVC(
                        kernel=kernel,
                        degree=degree,
                        C=C,
                        coef0=coef0,
                        gamma=gamma,
                        class_weight="balanced",
                        cache_size=100000,
                    ),
                )

            clf.fit(X_train, y_train)

            print("Training Report:")
            y_train_pred = clf.predict(X_train)
            print(classification_report(y_train, y_train_pred))
            print("\n")
            print("Test Report:")
            y_test_pred = clf.predict(X_test)
            print(classification_report(y_test, y_test_pred))

        i += 1


if __name__ == "__main__":
    main()
