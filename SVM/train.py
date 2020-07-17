import warnings
import random
warnings.filterwarnings("ignore")
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

from sklearn.svm import SVC

from sklearn.metrics import (
    f1_score,
    precision_score,
    confusion_matrix,
    recall_score,
    accuracy_score,
)

import argparse

from preprocess import prepare_data, load_csv

import time

from tqdm.auto import tqdm


def compute_acc(clf, X_train, y_train, X_test, y_test):

    # predictions on training set
    y_train_pred = clf.predict(X_train)
    y_train_label = np.array(y_train)
    training_acc = accuracy_score(y_train_label, y_train_pred)
    training_prec = precision_score(y_train_label, y_train_pred)
    training_recall = recall_score(y_train_label, y_train_pred)
    training_f1 = f1_score(y_train_label, y_train_pred)
    training_confusion_matrix = confusion_matrix(y_train_label, y_train_pred)

    """
	print("Training Accuracy: " + str(training_acc))
	print("Training Precision Score: " + str(training_prec))
	print("Training Recall Score: " + str(training_recall))
	print("Training F1 Score: " + str(training_f1))
	print("Training Confusion Matrix:\n " + str(training_confusion_matrix))
	"""

    # predictions on test set
    y_test_pred = clf.predict(X_test)
    y_test_label = np.array(y_test)
    test_set_acc = accuracy_score(y_test_label, y_test_pred)
    test_set_prec = precision_score(y_test_label, y_test_pred)
    test_set_recall = recall_score(y_test_label, y_test_pred)
    test_set_f1 = f1_score(y_test_label, y_test_pred)
    test_set_confusion_matrix = confusion_matrix(y_test_label, y_test_pred)

    """
	print("Test Accuracy: " + str(test_acc))
	print("Test Precision Score: " + str(test_prec))
	print("Test Recall Score: " + str(test_recall))
	print("Test F1 Score: " + str(test_f1))
	print("Test Confusion Matrix:\n " + str(test_confusion_matrix))
	"""

    return {
        "training": [training_acc, training_prec, training_recall, training_f1],
        "test": [test_set_acc, test_set_prec, test_set_recall, test_set_f1],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path", type=str, default="dataset/NDTX.csv", help="path of csv file"
    )

    parser.add_argument(
        "--trading_days", type=int, default=30, help="Number of trading days"
    )

    parser.add_argument(
        "--no_of_subsamples",
        type=int,
        default=1,
        help="Number of samples to take from csv file",
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
        "--C", type=float, default=1.0, help="the regularisation parameter for SVM"
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

    args = parser.parse_args()
    path = args.path
    trading_days = args.trading_days
    no_of_subsamples = args.no_of_subsamples
    kernel = args.kernel
    degree = args.degree
    C = args.C
    gamma = args.gamma
    if kernel == "poly":
        gamma = 1.0

    if kernel == "rbf":
        gamma = "scale"

    coef0 = args.coef0
    train_test_ratio = args.train_test_ratio

    trading_days = [trading_days]

    print("Details: ")
    print("Extracting data from: " + str(path))
    print("Trading Days: " + str(trading_days[0]))
    print("Number of Subsamples: " + str(no_of_subsamples))

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

    def custom_kernel(X, Y):
        return 1 / (1 + np.dot(X, Y.T) ** degree)

    df = load_csv(path)

    data = prepare_data(
        data_f=df, horizon=trading_days[0], alpha=0.9, trading_days=trading_days
    )

    # remove the output from the input
    features = [x for x in data.columns if x not in ["gain"]]

    dataA = np.array_split(data[features], no_of_subsamples)
    print(dataA)
    train_acc, train_prec, train_recall, train_f1 = (0, 0, 0, 0)
    test_acc, test_prec, test_recall, test_f1 = (0, 0, 0, 0)

    t0 = time.time()

    stats = []
    for i in tqdm(range(no_of_subsamples)):
        features = [x for x in data.columns if x not in ["gain", "pred"]]
        X = dataA[i][features]
        y = dataA[i]["pred"]
        X_train = X[: int(train_test_ratio * len(X))]
        y_train = y[: int(train_test_ratio * len(y))]
        print(X_train.shape)

        X_test = X[int(train_test_ratio * len(X)) :]
        y_test = y[int(train_test_ratio * len(y)) :]
        a=random.seed()
        if kernel == "custom":
        	clf = make_pipeline(StandardScaler(),SVC(kernel=custom_kernel, C=C,class_weight='balanced',cache_size=1000))
        else:
        	clf = make_pipeline(StandardScaler(),SVC(kernel=kernel, C=C, degree=degree, coef0=coef0, gamma=gamma,class_weight='balanced',cache_size=1000))
        clf.fit(X_train, y_train)

        metrics = compute_acc(clf, X_train, y_train, X_test, y_test)

        stats.append(metrics)

        train_acc += metrics["training"][0]
        train_prec += metrics["training"][1]
        train_recall += metrics["training"][2]
        train_f1 += metrics["training"][3]

        test_acc += metrics["test"][0]
        test_prec += metrics["test"][1]
        test_recall += metrics["test"][2]
        test_f1 += metrics["test"][3]

    print("\nTime taken: " + str((time.time() - t0) / 60) + " minutes")

    for i in range(no_of_subsamples):

        print("Stats for Subsample#" + str(i + 1))
        print("Training Accuracy:\t" + str(stats[i]["training"][0]))
        print("Training Precision:\t" + str(stats[i]["training"][1]))
        print("Training Recall:\t" + str(stats[i]["training"][2]))
        print("Training F1:\t\t" + str(stats[i]["training"][3]))

        print("\n")

        print("Test Accuracy:\t\t" + str(stats[i]["test"][0]))
        print("Test Precision:\t\t" + str(stats[i]["test"][1]))
        print("Test Recall:\t\t" + str(stats[i]["test"][2]))
        print("Test F1:\t\t" + str(stats[i]["test"][3]))

        print("\n")

    print("Average Results")
    print("Average Training Accuracy:\t" + str(train_acc / no_of_subsamples))
    print("Average Training Precision:\t" + str(train_prec / no_of_subsamples))
    print("Average Training Recall:\t" + str(train_recall / no_of_subsamples))
    print("Average Training F1:\t\t" + str(train_f1 / no_of_subsamples))

    print("\n")

    print("Average Test Accuracy:\t\t" + str(test_acc / no_of_subsamples))
    print("Average Test Precision:\t\t" + str(test_prec / no_of_subsamples))
    print("Average Test Recall:\t\t" + str(test_recall / no_of_subsamples))
    print("Average Test F1:\t\t" + str(test_f1 / no_of_subsamples))


if __name__ == "__main__":
    main()
