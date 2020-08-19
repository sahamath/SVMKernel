import os
import sys
from pipe1 import tuner
import csv


class args:
    path = "dataset/sensexData/sensex_daily_2000_2007.csv"
    trading_days = 7
    kernel = "cobb-douglas"
    degree = 3
    C = [10 ** i for i in range(-100, 100)]
    gamma = 0.1
    coef0 = 0.0
    train_test_ratio = 0.75
    folds = 5
    no_of_subsamples = 5
    currC = -1


f = open("summary.txt", "w")
f_csv = open("summary.csv", "w", newline="")
writer = csv.writer(f_csv)
writer.writerow(
    ["degree", "gamma", "C", "training F1", "test F1", "training acc", "test acc"]
)
print("Tuning and results for cobb-douglas: \n")
degrees = [2, 3, 4, 5, 6, 7, 8, 9]
# degrees = [2]
tradingDays = [5]
kernel = "cobb-douglas"
gammas = [0.1]
# gammas = [0.1]
print("Lets start this:\n")
for degree in degrees:
    args.degree = degree
    for gamma in gammas:
        args.gamma = gamma
        for day in tradingDays:
            args.trading_days = day
            print(
                "Result for degree: " + str(args.degree) + ", gamma: " + str(args.gamma)
            )
            tuner(args, f, writer)
f.close()
f_csv.close()
