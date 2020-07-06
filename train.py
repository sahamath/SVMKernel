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
        "--path", type=str, default="dataset/^NSEI (3).csv", help="path of csv file"
    )

    parser.add_argument(
        "--no_of_subsamples",
        type=int,
        default=4,
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

    parser.add_argument(
        "--train_test_ratio", type=float, default=0.75, help="fraction of train samples"
    )

    args = parser.parse_args()
    path = args.path
    no_of_subsamples = args.no_of_subsamples
    kernel = args.kernel
    degree = args.degree
    C = args.C
    coef0 = args.coef0
    train_test_ratio = args.train_test_ratio

    def custom_kernel(X, Y):
        return 1 / (1 + np.dot(X, Y.T) ** degree)

    df = load_csv(path)

    data = prepare_data(data_f=df, horizon=10, alpha=0.9)

    # remove the output from the input
    features = [x for x in data.columns if x not in ["gain"]]

    dataA = np.array_split(data[features], no_of_subsamples)

    train_acc, train_prec, train_recall, train_f1 = (0, 0, 0, 0)
    test_acc, test_prec, test_recall, test_f1 = (0, 0, 0, 0)

    t0 = time.time()

    for i in tqdm(range(no_of_subsamples)):
        features = [x for x in data.columns if x not in ["gain", "pred"]]
        X = dataA[i][features]
        y = dataA[i]["pred"]
        X_train = X[: int(train_test_ratio * len(X))]
        y_train = y[: int(train_test_ratio * len(y))]

        X_test = X[int(train_test_ratio * len(X)) :]
        y_test = y[int(train_test_ratio * len(y)) :]

        if kernel == "custom":
            clf = SVC(kernel=custom_kernel, C=C, coef0=coef0)
        else:
            clf = SVC(kernel=kernel, C=C, degree=degree, coef0=coef0)

        clf.fit(X_train, y_train)

        metrics = compute_acc(clf, X_train, y_train, X_test, y_test)

        train_acc += metrics["training"][0]
        train_prec += metrics["training"][1]
        train_recall += metrics["training"][2]
        train_f1 += metrics["training"][3]

        test_acc += metrics["test"][0]
        test_prec += metrics["test"][1]
        test_recall += metrics["test"][2]
        test_f1 += metrics["test"][3]

    print("\nTime taken: " + str((time.time() - t0) / 60) + " minutes")

    print("\n")

    print("Average Training Accuracy:\t" + str(train_acc / no_of_subsamples))
    print("Average Training Precision:\t" + str(train_prec / no_of_subsamples))
    print("Average Training Recall:\t" + str(train_recall / no_of_subsamples))
    print("Average Training F1:\t\t" + str(train_f1 / no_of_subsamples))

    print("\n")

    print("Average Test Accuracy:\t" + str(test_acc / no_of_subsamples))
    print("Average Test Precision:\t" + str(test_prec / no_of_subsamples))
    print("Average Test Recall:\t" + str(test_recall / no_of_subsamples))
    print("Average Test F1:\t" + str(test_f1 / no_of_subsamples))


if __name__ == "__main__":
    main()
