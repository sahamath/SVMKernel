import warnings

warnings.filterwarnings("ignore")

import numpy as np

from sklearn.svm import SVC

from sklearn.metrics import (
    f1_score,
    precision_score,
    confusion_matrix,
    recall_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split

import argparse

from preprocess import prepare_data, load_csv

import time


def compute_acc(clf, X_train, y_train, X_test, y_test):

    # predictions on training set
    y_train_pred = clf.predict(X_train)
    y_train_label = np.array(y_train)
    training_acc = accuracy_score(y_train_label, y_train_pred)
    training_prec = precision_score(y_train_label, y_train_pred)
    training_recall = recall_score(y_train_label, y_train_pred)
    training_f1 = f1_score(y_train_label, y_train_pred)
    training_confusion_matrix = confusion_matrix(y_train_label, y_train_pred)

    print("Training Accuracy: " + str(training_acc))
    print("Training Precision Score: " + str(training_prec))
    print("Training Recall Score: " + str(training_recall))
    print("Training F1 Score: " + str(training_f1))
    print("Training Confusion Matrix:\n " + str(training_confusion_matrix))

    print("\n\n")

    # predictions on test set
    y_test_pred = clf.predict(X_test)
    y_test_label = np.array(y_test)
    test_acc = accuracy_score(y_test_label, y_test_pred)
    test_prec = precision_score(y_test_label, y_test_pred)
    test_recall = recall_score(y_test_label, y_test_pred)
    test_f1 = f1_score(y_test_label, y_test_pred)
    test_confusion_matrix = confusion_matrix(y_test_label, y_test_pred)

    print("Test Accuracy: " + str(test_acc))
    print("Test Precision Score: " + str(test_prec))
    print("Test Recall Score: " + str(test_recall))
    print("Test F1 Score: " + str(test_f1))
    print("Test Confusion Matrix:\n " + str(test_confusion_matrix))

    return [training_acc, test_acc]


def custom_kernel(X, Y):
    return 1 / (1 + np.dot(X, Y.T) ** degree)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path", type=str, default="dataset/^NSEI (3).csv", help="path of csv file"
    )

    parser.add_argument(
        "--no_of_subsamples",
        type=int,
        default=2,
        help="Number of samples to take from csv file",
    )

    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        help="the kernel for SVM",
        choices=["linear", "rbf", "poly", "custom"],
    )

    parser.add_argument(
        "--degree", type=int, default=3, help="value of p in polynomial/custom kernel"
    )

    parser.add_argument(
        "--C", type=float, default=1.0, help="the regularisation parameter for SVM"
    )

    parser.add_argument(
        "--coef0", type=float, default=0.0, help="coefficient for polynomial kernel"
    )

    args = parser.parse_args()
    path = args.path
    no_of_subsamples = args.no_of_subsamples
    kernel = args.kernel
    degree = args.degree
    C = args.C
    coef0 = args.coef0

    data_f = load_csv(path)
    df_subsamples = np.array_split(data_f, no_of_subsamples)

    for df_c in df_subsamples:

        df = df_c.copy()

        data = prepare_data(data_f=df, horizon=10, alpha=0.9)

        y = data["pred"]

        # remove the output from the input
        features = [x for x in data.columns if x not in ["gain", "pred"]]
        X = data[features]

        X_train = X[: int(0.75 * len(X))]
        y_train = y[: int(0.75 * len(y))]

        X_test = X[int(0.75 * len(X)) :]
        y_test = y[int(0.75 * len(y)) :]

        t0 = time.time()

        if kernel == "custom":
            clf = SVC(kernel=custom_kernel, C=C, coef0=coef0)
        else:
            clf = SVC(kernel=kernel, C=C, degree=degree, coef0=coef0)

        clf.fit(X_train, y_train)

        print("Time taken: " + str((time.time() - t0) / 60) + " minutes")
        compute_acc(clf, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
