import numpy as np
from sklearn.model_selection import GridSearchCV

import fileUtils as fu
import config as cfg
import logger as lg
import preProcess as pp

def process(X_train, X_test, y_train, y_test):
    if cfg.regr == False:
        
        from sklearn.svm import LinearSVC
        clf = LinearSVC(random_state=0, tol=1e-5)

        #create a dictionary of all values we want to test for n_neighbors
        param_grid = cfg.svm_param_grid #{'C':[0.01, 0.1, 1, 10, 100]}
        #use gridsearch to test all values for n_neighbors
        svm_gscv = GridSearchCV(clf, param_grid, cv=cfg.n_folds, verbose=2)
        #fit model to data
        y_train = np.reshape(y_train, (y_train.size, ))
        svm_gscv.fit(X_train, y_train)
        
        #check top performing n_neighbors value
        best_params = svm_gscv.best_params_
        best_C = best_params['C']

        #check mean score for the top performing value of n_neighbors
        best_score = svm_gscv.best_score_

        model = LinearSVC(random_state=0, tol=1e-5, C=best_C)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        lg.logger.info(f'Best C: {best_C}, Test score: {score}')
        
        '''
        from sklearn.svm import LinearSVC
        clf = LinearSVC(random_state=0, tol=1e-5)
        clf.fit(X_train, y_train)
        w = clf.coef_
        bias = clf.intercept_
        y_pred = clf.predict(X_test)
        from sklearn.metrics import confusion_matrix, classification_report
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        '''
    else: # Regression
        '''
        from sklearn.svm import SVR
        regr = SVR(C=20)
        regr.fit(X_train, y_train)
        w = regr.coef_
        bias = regr.intercept_
        print(regr.score(X_test, y_test))
        '''
        from sklearn.svm import SVR
        regr = SVR(kernel='linear')

        #create a dictionary of all values we want to test for n_neighbors
        param_grid = cfg.svm_param_grid #{'C':[0.01, 0.1, 1, 10, 100]}
        #use gridsearch to test all values for n_neighbors
        svr_gscv = GridSearchCV(regr, param_grid, cv=cfg.n_folds, verbose=2)
        #fit model to data
        y_train = np.reshape(y_train, (y_train.size, ))
        svr_gscv.fit(X_train, y_train)
        
        #check top performing n_neighbors value
        best_params = svr_gscv.best_params_
        best_C = best_params['C']

        #check mean score for the top performing value of n_neighbors
        best_score = svr_gscv.best_score_

        model = SVR(kernel='linear', tol=1e-5, C=best_C)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        pred_y = model.predict(X_test)
        
        from sklearn.metrics import r2_score 
        r2 = r2_score(y_test, pred_y)
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

        lg.logger.info(f'Best C: {best_C}, R2: {r2}, mse: {mse}')

    w = model.coef_
    bias = model.intercept_
    fu.saveSVMParams(w, bias)
    fu.savePPParams()