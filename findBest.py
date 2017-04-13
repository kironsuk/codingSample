import numpy as np
import pandas as pd
from loadCsvs import myloader
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR

#load and preprocess
loadingObj = myloader()
loadingObj.imputeData('median')
loadingObj.transformToPCA()
#loadingObj.reduceDimByAgglo()

#extract arrays
dataDictionary=loadingObj.getData()
features = dataDictionary['data']
targets = dataDictionary['targets']


#Preform Kernel Ridge Regression
def preformCrossVal(kern,epsi, g):
    regr = SVR(kernel = kern)

    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.1);

    regr.fit(X_train, y_train)

    # The mean squared error
    print("Mean percentage error: %.2f"
          % np.mean(np.abs(regr.predict(X_test) - y_test)/y_test ))

    #Cross Validation
    scores = cross_val_score(regr, features, targets,cv=10, scoring='r2')
    return np.abs(np.mean(scores))



#try different kernels and print out
kernels = ['linear','rbf']
for k in kernels:
    print preformCrossVal(k,.1,1)
