import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA


class myloader:
    def __init__(self):
        self.features = np.genfromtxt ('features.csv', delimiter=',')
        self.headers = pd.read_csv('headers.csv')
        self.targets = np.genfromtxt('targets.csv',delimiter=',')

    def getData(self):
        myDict = {'data':self.features,'targets':self.targets, 'header':self.headers}
        return myDict

    def imputeData(self,strat ):
        imp = Imputer(missing_values=-1, strategy=strat, axis=0)
        imp.fit(self.features)

    def transformToPCA(self):
        pca = PCA(n_components=50, svd_solver='randomized',
                  whiten=True).fit(self.features)
        self.features = pca.transform(self.features)
