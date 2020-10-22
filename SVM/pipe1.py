import warnings
import sys
import pandas as pd

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
    # print(arr)
    dataF = pd.DataFrame(arr)
    # print(dataF)
    dataF = dataF.replace([np.inf, -np.inf], np.nan)
    dataF = dataF.fillna(0)
    return dataF.to_numpy()


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

    dataA = np.array_split(data[features], folds)

    features = [x for x in data.columns if x not in ["gain", "pred"]]
    X = np.array(dataA[0][features])
    y = np.array(dataA[0]["pred"])

    #tscv = TimeSeriesSplit(n_splits=folds)
    param_grid = {"svc__C": C}

    # print("\n")
    metrics = {}
    for C in param_grid["svc__C"]:

        metrics[C] = {"accuracy": [], "precision": [], "recall": [], "f1": []}
        i = 0

        for i in tqdm(range(no_of_subsamples)):
            features = [x for x in data.columns if x not in ["gain", "pred"]]
            X = dataA[i][features]
            y = dataA[i]["pred"]
            # print("Fold #" + str(i + 1))
            # print("\n")

            X_train = X[: int(train_test_ratio * len(X))]
            y_train = y[: int(train_test_ratio * len(y))]

            X_test = X[int(train_test_ratio * len(X)) :]
            y_test = y[int(train_test_ratio * len(y)) :]


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
    max_recall_C = list(metrics.keys())[0]
    for C in metrics:
        if mean(metrics[C]["recall"]) > mean(metrics[max_recall_C]["recall"]):
            max_recall_C = C
        # print("\n")

    print("Best Results:\n")
    print(max_recall_C)
    print("Recall: " + str(mean(metrics[max_recall_C]["recall"])))

    f.write("\n\nBest Results of Tuner:\n")
    f.write("\nDegree: " + str(args.degree))
    f.write("\nC: " + str(max_recall_C))
    f.write("\ngamma: " + str(gamma))
    f.write("\nBest Recall Score: " + str(mean(metrics[max_recall_C]["recall"])))

    args1 = args
    args1.currC = max_recall_C
    trainer(args1, f, writer)


if __name__ == "__main__":
    main()
