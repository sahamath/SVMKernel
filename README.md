## Support-Vector-Economics

### To train the SVM Classifier and report the results:

```
python train.py --kernel poly --degree 5
```
Kernel must be one of linear/poly/rbf/custom.


### Guide

- Metrics Table.xlsx contains the **average** results on the "*^ NSEI (3).csv*" dataset.
- The code reports results for all subsamples, in addition to reporting the average results.


### Number of Subsamples Chosen
- *^ NSEI (3).csv*: 4 
- *novartis.csv*: 1 (cannot do more than 1 subsamples here, since with <=2 subsamples, at least one subsample has only one class)
- *roche.csv*: 2
- *pfizer.csv*: 3
### To do

- Run the code on the dataset Sir provided and report the results.
- Update the Latex file.
- Run the code for linear kernel (SVC with linear kernel takes hours to train).
