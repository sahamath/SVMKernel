import warnings
import sys

# sys.stdout = open('output.txt','a')
warnings.filterwarnings("ignore")

import argparse
from pipe2 import trainer
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

from statistics import mean

from preprocess import prepare_data, load_csv

from tqdm.auto import tqdm


def rem_inf(arr):
    check_for_inf = np.where(np.isinf(arr))
    if check_for_inf:
        for i in range(len(check_for_inf[0])):
            # print((check_for_inf[0][i],check_for_inf[1][i]))
            arr[check_for_inf[0][i], check_for_inf[1][i]] = 0
    return arr


def tuner(args, f, writer):
    path = args.path
    trading_days = args.trading_days
    kernel = args.kernel
    degree = args.degree
    C = [float(i) for i in args.C]
    gamma = args.gamma

    if kernel == "poly":
        gamma = 1.0

    if kernel == "rbf":
        gamma = "scale"

    coef0 = args.coef0
    train_test_ratio = args.train_test_ratio
    folds = args.folds

    # print("Details: ")
    # print("Extracting data from: " + str(path))
    # print("Trading Days: " + str(trading_days))

    if kernel == "poly":
        assert (
            gamma == 1.0
        ), "Polynomial should have gamma=1.0, otherwise it is Cobb-Douglas Kernel"

        # print("Kernel: polynomial")
        # print("Degree: " + str(degree))

    elif kernel == "cobb-douglas":
        assert (
            gamma != 1.0
        ), "Cobb-Douglas should have gamma!=1.0, otherwise it is Polynomial Kernel"
        # print("Degree in pipe1: " + str(degree))
        kernel = "poly"

    def poly_cobb_kernel(X, Y):
        return gamma * (np.dot(X, Y.T)) ** degree

    def custom_kernel(X, Y):
        return 1 / (1 + np.dot(X, Y.T) ** degree)

    # load the dataset
    df = load_csv(path)
    data = prepare_data(data_f=df, horizon=trading_days, alpha=0.9)
    # print(data)
    # remove the output from the input
    features = [x for x in data.columns if x not in ["gain"]]

    dataA = np.array_split(data[features], 1)

    features = [x for x in data.columns if x not in ["gain", "pred"]]
    X = np.array(dataA[0][features])
    y = np.array(dataA[0]["pred"])

    tscv = TimeSeriesSplit(n_splits=folds)
    param_grid = {"svc__C": C}

    # print("\n")
    metrics = {}
    for C in param_grid["svc__C"]:

        metrics[C] = {"accuracy": [], "precision": [], "recall": [], "f1": []}
        i = 0

        for tr_index, val_index in tscv.split(X):

            # print("Fold #" + str(i + 1))
            # print("\n")

            X_train, X_test = (
                X[tr_index].astype("float64"),
                X[val_index].astype("float64"),
            )
            y_train, y_test = (
                y[tr_index].astype("float64"),
                y[val_index].astype("float64"),
            )

            X_train = rem_inf(X_train)
            X_test = rem_inf(X_test)

            # print(np.where(np.isinf(X_train)))
            # print(np.where(np.isinf(X_test)))

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

            # print("Training Report:")
            y_train_pred = clf.predict(X_train)
            train_res = classification_report(y_train, y_train_pred, output_dict=True)
            # print(classification_report(y_train, y_train_pred))
            # print("\n")
            # print("Test Report:")
            y_test_pred = clf.predict(X_test)
            test_res = classification_report(y_test, y_test_pred, output_dict=True)
            # print(test_res)
            # print(classification_report(y_test, y_test_pred))
            metrics[C]["accuracy"].append(test_res["accuracy"])
            metrics[C]["precision"].append(test_res["macro avg"]["precision"])
            metrics[C]["recall"].append(test_res["macro avg"]["recall"])
            metrics[C]["f1"].append(test_res["macro avg"]["f1-score"])

            i += 1

    # print(metrics)
    # print("\n")
    max_f1_C = list(metrics.keys())[0]
    for C in metrics:
        if mean(metrics[C]["f1"]) > mean(metrics[max_f1_C]["f1"]):
            max_f1_C = C
        # print("\n")

    print("Best Results:\n")
    print(max_f1_C)
    print("F1: " + str(mean(metrics[max_f1_C]["f1"])))

    f.write("\n\nBest Results of Tuner:\n")
    f.write("\nDegree: " + str(args.degree))
    f.write("\nC: " + str(max_f1_C))
    f.write("\ngamma: " + str(gamma))
    f.write("\nBest F1 Score: " + str(mean(metrics[max_f1_C]["f1"])))

    args1 = args
    args1.currC = max_f1_C
    trainer(args1, f, writer)


if __name__ == "__main__":
    main()
