import numpy as np
import pandas as pd
from loadCsvs import myloader
from sklearn.model_selection import train_test_split
from sklearn import linear_model

loadingObj = myloader()
loadingObj.imputeData('mean')

dataDictionary=loadingObj.getData()
features = dataDictionary['data']
targets = dataDictionary['targets']

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.1);


#Preform RIdge Regression
regr = linear_model.Ridge (alpha = .5)
regr.fit(X_train, y_train)
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test,  y_test))
