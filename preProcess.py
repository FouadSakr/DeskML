from numpy import loadtxt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA 

import config as cfg
import preProcess as pp

s_x = []
s_y = []
#u = []
#s = []
pca_components = [[]]
n_orig_features = 0

def preprocess(file, delimiter, skiprows, endColumn, startColumn=0, targetColumn=None, decimal=',', pca=None):
    #X, y = loadCSV(file, delimiter, skiprows, nColumns)
    if targetColumn == None:
        targetColumn = endColumn
    parser = pd.read_csv(file, header=None, skiprows=skiprows, sep=delimiter, decimal=decimal)
    parser.head()
    X = parser.iloc[:,startColumn:endColumn]
    y = parser.iloc[:,targetColumn]
    X = X.astype('float32')
    X.fillna(X.mean(),inplace = True)

    global n_orig_features
    n_orig_features = X.shape[1]

    if cfg.algo.lower() != 'dt' and cfg.normalization != None:
        y = np.array(y).reshape(-1,1)
        X, y = normalize(X, y)
    else: #No normalization for DT
        pp.s_x = np.ones(X.shape[1])
        pp.s_y = [1]
        X = X.to_numpy()
        y = np.array(y).reshape(-1,1)
    X = PrincipalComponentAnalysis(X, pca)
    return X, y

def loadCSV(file, delimiter, skiprows, nColumns):
    # load the dataset
    dataset = loadtxt(file, delimiter=delimiter, skiprows=skiprows)
    # split into input (X) and output (y) variables
    X = dataset[:,0:nColumns-1]
    y = dataset[:,nColumns-1]
    return X, y

#Normalizing the data
# https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
# https://medium.com/datadriveninvestor/building-neural-network-using-keras-for-regression-ceee5a9eadff 
def normalize(X, y):
    if cfg.normalization.lower() == 'standard':
        sc = StandardScaler()
    elif cfg.normalization.lower() == 'minmax':
        sc = MinMaxScaler()
    else:
        print('Error: normalization not recognized: '+cfg.normalization)
    X = sc.fit_transform(X)
    global s_x
    s_x = sc.scale_
    # These are needed at runtime z=(x-u)/s for standard
    # z=x*s for minmax

    if cfg.normalization.lower() == 'standard':
        global u_x
        u_x = sc.mean_

    #global s
    #s = sc.scale_ # These are needed at runtime z=(x-u)/s
    #global u
    #u = sc.mean_

    if cfg.regr == True:
        y_orig = y
        y = sc.fit_transform(y)
        global s_y
        s_y = sc.scale_
        if cfg.normalization.lower() == 'standard':
            global u_y
            u_y = sc.mean_
    return X, y

# https://towardsdatascience.com/dimensionality-reduction-does-pca-really-improve-classification-outcome-6e9ba21f0a32
def PrincipalComponentAnalysis(X, pca):
    global pca_components
    if pca == None:
        pca_components = np.identity(X.shape[1])
    else:
        pca = PCA(n_components=pca)
        X = pca.fit_transform(X)    
        pca_components = pca.components_ # These are needed at runtime
    return X


#Might be useful for linear and SVM
#https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
#https://towardsdatascience.com/feature-selection-techniques-for-classification-and-python-tips-for-their-application-10c0ddd7918b 
def featureSelection(X,y):
    
    from keras.models import Sequential
    from keras.models import clone_model
    from keras.layers import Dense
    from sklearn.feature_selection import RFE
    model = Sequential()
    model.add(Dense(80, input_dim=13, activation='tanh')) #relu
    model.add(Dense(30, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))

    rfe = RFE(model,5)    
    #Transforming data using RFE
    X_rfe = rfe.fit_transform(X,y)  
    #Fitting the data to model
    model.fit(X_rfe,y)

    return