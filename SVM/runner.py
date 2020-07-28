import os
import sys
file = open("output.txt","w")
file. truncate(0)
file. close()
# sys.stdout = open('output.txt','a')
print('Tuning and results for cobb-douglas: \n')
degrees = [4,5,6,7,8,9]
tradingDays = [5]
kernel = 'cobb-douglas'
gammas = [0.1]
print('Lets start this :\n')
for degree in degrees:
    for gamma in gammas:
        for day in tradingDays:
            print('Result for degree :' + str(degree) + ' ,gamma : ' + str(gamma))
            os.system('python3 pipe1.py '+'--degree '+str(degree)+' --kernel '+str(kernel)+' --gamma '+str(gamma)+' --trading_days '+str(day))
