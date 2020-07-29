import os
import sys
from pipe1 import tuner
file = open("output.txt","w")
file. truncate(0)
file. close()
# sys.stdout = open('output.txt','a')
class args:
    path = "dataset/NSEIdaily.csv"
    trading_days = 5
    kernel = "cobb-douglas"
    degree = 3
    C = [10 ** i for i in range(-25, 2)]
    gamma = 0.1
    coef0 = 0.0
    train_test_ratio = 0.75
    folds = 5
    no_of_subsamples = 5
    currC = -1
print('Tuning and results for cobb-douglas: \n')
degrees = [4,5,6,7,8,9]
tradingDays = [5]
kernel = 'cobb-douglas'
gammas = [0.1]
print('Lets start this :\n')
for degree in degrees:
    args.degree = degree
    for gamma in gammas:
        args.gamma = gamma
        for day in tradingDays:
            args.trading_days = day
            print('Result for degree :' + str(args.degree) + ' ,gamma : ' + str(args.gamma))
            tuner(args)
