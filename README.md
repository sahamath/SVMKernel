## Support-Vector-Economics


### To train the SVM Classifier and report the results:

```
python train.py --kernel poly --degree 5
```
Kernel must be one of linear/poly/rbf/custom.


### Guide

- The datasets are in the ```dataset``` folder.
- The ```Results``` folder contains the **average** results.
- The code reports results for all subsamples, in addition to reporting the average results.


### Number of Subsamples Chosen
- *dataset/^ NSEI (3).csv*: 4 
- *dataset/novartis.csv*: 1 (cannot do more than 1 subsample here, since with >=2 subsamples, at least one subsample has only one class)
- *dataset/roche.csv*: 2
- *dataset/pfizer.csv*: 3


### To do

- Run the code on the dataset Sir provided and report the results.
- Update the Latex file.
- Run the code for linear kernel on the *dataset/^ NSEI (3).csv* dataset (SVC with linear kernel takes hours to train).
