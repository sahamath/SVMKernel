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
    classification_report,
)

from sklearn.model_selection import GridSearchCV

import argparse

from preprocess import prepare_data, load_csv

import time


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path", type=str, default="dataset/^NSEI (3).csv", help="path of csv file"
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
        "--C",
        nargs="+",
        default=[0.1, 1, 10, 100],
        help="the regularisation parameter for SVM",
    )

    parser.add_argument(
        "--gamma",
        nargs="+",
        default=[1, 0.1, 0.01, 0.001],
        help="list of gamma values for the grid search",
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
    no_of_subsamples = 1
    kernel = args.kernel
    degree = args.degree
    C = [float(i) for i in args.C]
    gamma = [float(i) for i in args.gamma]
    coef0 = args.coef0
    train_test_ratio = args.train_test_ratio
    folds = args.folds

    print("C: " + str(C))

    print("gamma: " + str(gamma))

    def custom_kernel(X, Y):
        return 1 / (1 + np.dot(X, Y.T) ** degree)

    df = load_csv(path)

    data = prepare_data(data_f=df, horizon=10, alpha=0.9)

    # remove the output from the input
    features = [x for x in data.columns if x not in ["gain"]]

    dataA = np.array_split(data[features], no_of_subsamples)

    stats = []
    for i in range(no_of_subsamples):
        features = [x for x in data.columns if x not in ["gain", "pred"]]
        X = dataA[i][features]
        y = dataA[i]["pred"]
        X_train = X[: int(train_test_ratio * len(X))]
        y_train = y[: int(train_test_ratio * len(y))]

        X_test = X[int(train_test_ratio * len(X)) :]
        y_test = y[int(train_test_ratio * len(y)) :]

        if kernel in ["rbf", "poly", "sigmoid"]:
            param_grid = {"C": C, "gamma": gamma}
        else:
            param_grid = {"C": C}

        if kernel == "custom":
            clf = SVC(kernel=custom_kernel, coef0=coef0)
        else:
            clf = SVC(kernel=kernel, degree=degree, coef0=coef0)

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


if __name__ == "__main__":
    main()
