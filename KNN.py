from sklearn.model_selection import GridSearchCV
import numpy as np
from math import sqrt

import config as cfg
import logger as lg
import fileUtils as fu
import preProcess as pp

def process(X_train, X_test, y_train, y_test):
    
    #create new a knn model
    if cfg.regr == False:
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier()

        #create a dictionary of all values we want to test for n_neighbors
        param_grid = {'n_neighbors': np.arange(cfg.cvl_knn, cfg.cvu_knn)}
        #use gridsearch to test all values for n_neighbors
        knn_gscv = GridSearchCV(knn, param_grid, cv=cfg.n_folds, verbose=2)
        #fit model to data
        y_train = np.reshape(y_train, (y_train.size, ))
        knn_gscv.fit(X_train, y_train)
        
        #check top performing n_neighbors value
        best_params = knn_gscv.best_params_
        best_k = best_params['n_neighbors']

        #check mean score for the top performing value of n_neighbors
        best_score = knn_gscv.best_score_

        knn = KNeighborsClassifier(n_neighbors=best_k)
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)

        lg.logger.info(f'Best K: {best_k}, Test score: {score}')

    else: # Regresion
        from sklearn.neighbors import KNeighborsRegressor
        knn = KNeighborsRegressor()

        #create a dictionary of all values we want to test for n_neighbors
        param_grid = {'n_neighbors': np.arange(cfg.cvl_knn, cfg.cvu_knn)}
        #use gridsearch to test all values for n_neighbors
        knn_gscv = GridSearchCV(knn, param_grid, cv=cfg.n_folds, verbose=2) # scoring='neg_mean_squared_error'
        #fit model to data
        y_train = np.reshape(y_train, (y_train.size, ))
        knn_gscv.fit(X_train, y_train)
        
        #check top performing n_neighbors value
        best_params = knn_gscv.best_params_
        best_k = best_params['n_neighbors']

        #check mean score for the top performing value of n_neighbors
        best_score = knn_gscv.best_score_

        knn = KNeighborsRegressor(n_neighbors=best_k)
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test) # R2 # scoring='neg_mean_squared_error' #https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
        # https://www.datatechnotes.com/2019/04/regression-example-with-k-nearest.html
        pred_y = knn.predict(X_test)
        from sklearn.metrics import mean_squared_error 
        if cfg.normalization == None:
            mse = mean_squared_error(y_test, pred_y)
        elif cfg.normalization.lower() == 'standard':
            pred_y_back = pp.u_y + pred_y*pp.s_y
            y_test_back = pp.u_y + y_test*pp.s_y
            mse = mean_squared_error(y_test_back, pred_y_back)
        elif cfg.normalization.lower() == 'minmax':
            pred_y_back = pred_y/pp.s_y
            y_test_back = y_test/pp.s_y
            mse = mean_squared_error(y_test_back, pred_y_back)
        #mse = mean_squared_error(y_test/pp.s_y, pred_y/pp.s_y)

        lg.logger.info(f'Best K: {best_k}, R2: {score}, mse: {mse}')

    fu.saveKNNParams(best_k)
    fu.savePPParams()