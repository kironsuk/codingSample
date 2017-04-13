import numpy as np
import pandas as pd
from loadCsvs import myloader
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.kernel_ridge import KernelRidge

#load and preprocess
loadingObj = myloader()
loadingObj.imputeData('median')
loadingObj.transformToPCA()
#loadingObj.reduceDimByAgglo()

#extract arrays
dataDictionary=loadingObj.getData()
features = dataDictionary['data']
targets = dataDictionary['targets']

#get training and testing set
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.1);

#Preform Kernel Ridge Regression
regr = KernelRidge (kernel='rbf', alpha=.1, gamma=5)
regr.fit(X_train, y_train)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test,  y_test))

#Cross Validation
scores = cross_val_score(regr, features, targets,cv=10, scoring='neg_median_absolute_error')
print 'Scores: ', np.abs(scores)
print 'Mean score', np.mean(scores)
