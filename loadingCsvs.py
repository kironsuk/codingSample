#Trying to load data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model


features = np.genfromtxt ('features.csv', delimiter=',')
#headers = np.genfromtxt('headers.csv',delimiter=',')
headers = pd.read_csv('headers.csv')
targets = np.genfromtxt('targets.csv',delimiter=',')
myDataset = (features,targets)
myDict = {'data':features,'targets':targets}

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.33, random_state=42);

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test,  y_test))
