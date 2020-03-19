from keras.models import Sequential
from keras.models import clone_model
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.base import clone
from math import sqrt
import numpy as np

import config as cfg
import preProcess as pp
import logger as lg

def getBestModel(X,y, loss, metrics):
    layerShape = genFromTxt('./config/' + cfg.ds_name + "/layerShape.dat", type='int')
    activeFuncs = genFromTxt('./config/' + cfg.ds_name + "/activeFuncs.dat")

    invert = False
    bestScore = 0
    if cfg.regr:
        invert = True
        bestScore = 100000000

    for a in activeFuncs:
        for s in layerShape:
            score = estimateModelScore(X, y, s, a, loss=loss, metrics=metrics, epochs=cfg.epochs, batch_size=cfg.batch_size, n_folds = cfg.n_folds, repeats = cfg.repeats)
            if invert:
                if score < bestScore:
                    bestScore = score
                    bestAF = a
                    bestS = s
            elif score > bestScore:
                bestScore = score
                bestAF = a
                bestS = s
    #logger.info(f'Dataset: {ds_name}')
    lg.logger.info(f'Score: {bestScore}')
    lg.logger.info(f'Activation: {bestAF}')
    lg.logger.info(f'Shape: {bestS}')
    return bestS, bestAF

def createModel(input_dim, shape, activation, loss, metrics):
    # define the keras model
    model = Sequential()
    
    for i, s in enumerate(shape):
        if i==0:
            model.add(Dense(s, input_dim=input_dim, activation=activation))
        else:
            model.add(Dense(s, activation=activation))
        if cfg.dropout != None:
                model.add(Dropout(cfg.dropout))
    if cfg.regr == True:
        # Regression
        model.add(Dense(1))
    elif cfg.nClass <= 2:
        # Binary
        model.add(Dense(1, activation='sigmoid')) 
    else:
        # Multiclass
        model.add(Dense(cfg.nClass, activation='softmax'))

    model.compile(loss=loss, optimizer='adam', metrics=metrics)
    #model.summary()

    return model

def estimateModelScore(X, y, s, a, loss, metrics, epochs=100, batch_size=10, n_folds = 10, repeats = 5):
    scores, cv_scores = list(), list()

    for i in range(repeats):
        # kfold = StratifiedKFold(n_splits=n_folds, shuffle=True)
        kfold = KFold(n_splits=n_folds)
        cv_scores = list()
        for train, test in kfold.split(X):

            model = createModel(len(X[0]), s, a, loss=loss, metrics=metrics)
            model.fit(X[train], y[train], batch_size=batch_size, epochs=epochs, verbose=2)
            _, score = model.evaluate(X[test], y[test])
            
            # Useful for debugging
            if cfg.regr == True:
                pred_y = model.predict(X[test])
                from sklearn.metrics import mean_squared_error

                if cfg.normalization == None:
                    mse = mean_squared_error(y[test], pred_y)
                elif cfg.normalization.lower() == 'standard':
                    pred_y_back = pp.u_y + pred_y*pp.s_y
                    y_test_back = pp.u_y + y[test]*pp.s_y
                    mse = mean_squared_error(y_test_back, pred_y_back)
                elif cfg.normalization.lower() == 'minmax':
                    pred_y_back = pred_y/pp.s_y
                    y_test_back = y[test]/pp.s_y
                    mse = mean_squared_error(y_test_back, pred_y_back)

                #mse = mean_squared_error(y[test]/pp.s_y, pred_y/pp.s_y)
                from sklearn.metrics import r2_score 
                r2 = r2_score(y[test], pred_y)
            
            print('>%.3f' % score)
            cv_scores.append(score)
        print('Estimated Score %.3f (%.3f)' % (np.mean(cv_scores), np.std(cv_scores)))
        scores.append(np.mean(cv_scores))
 
    egs = np.mean(scores)
    print('Estimated Grand Score %.3f (%.3f)' % (egs, np.std(scores)))
    standard_error = np.std(scores) / sqrt(len(scores))
    print('Std error: %.3f', standard_error)
    return egs

def genFromTxt(file, delimiter=",", type='string'):
    import csv
    datafile = open(file, 'r')
    
    reader = csv.reader(datafile)
    rows = list(reader)
    result = list()
    if type == 'int':
        for row in rows:
            result.append([int(i) for i in row]) # list comprehension
    else:
        result = [r.replace('\'', '').strip() for r in rows[0]]
    return result