import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

import config as cfg
import logger as lg
import fileUtils as fu

def process(X_train, X_test, y_train, y_test):
    
    if cfg.regr == False:
        estimator = DecisionTreeClassifier(random_state=0) # max_leaf_nodes=10, 
    else:
        estimator = DecisionTreeRegressor(random_state=0) #(max_leaf_nodes=1000, random_state=0)

    #create a dictionary of all values we want to test
    max_depth = [3]#, 7]#, None]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 3, 10]
    if cfg.regr == False:
        criterion = ["gini"] #, "entropy"]
    else:
        criterion = ["mse"] #, "friedman_mse"]
    max_leaf_nodes = [80]#, 200]#, 1000, 5000]
    param_grid = [{'max_depth':max_depth,
            'min_samples_split':min_samples_split,
            'min_samples_leaf':min_samples_leaf,
            'criterion':criterion,
            'max_leaf_nodes':max_leaf_nodes}]
    #use gridsearch to test all values for n_neighbors
    gscv = GridSearchCV(estimator, param_grid, cv=cfg.n_folds, verbose=2)
    #fit model to data
    gscv.fit(X_train, y_train)
    
    #check top performing n_neighbors value
    best_params = gscv.best_params_
    #best_C = best_params['C']
    max_depth = best_params['max_depth']
    min_samples_split = best_params['min_samples_split']
    min_samples_leaf = best_params['min_samples_leaf']
    criterion = best_params['criterion']
    max_leaf_nodes = best_params['max_leaf_nodes']

    #check mean score for the top performing value of n_neighbors
    best_score = gscv.best_score_

    if cfg.regr == False:
        estimator = DecisionTreeClassifier(random_state=0, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion, max_leaf_nodes=max_leaf_nodes) # max_leaf_nodes=10, 
    else:
        estimator = DecisionTreeRegressor(random_state=0, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion, max_leaf_nodes=max_leaf_nodes) 
    estimator.fit(X_train, y_train)
    score = estimator.score(X_test, y_test)

    if cfg.regr == False:
        lg.logger.info(f'Max_depth: {max_depth}, min_samples_split: {min_samples_split}, min_samples_leaf: {min_samples_leaf}, criterion: {criterion}, max_leaf_nodes: {max_leaf_nodes}, Test score: {score}')
    else:
        pred_y = estimator.predict(X_test)
        from sklearn.metrics import mean_squared_error 
        mse = mean_squared_error(y_test, pred_y)
        from sklearn.metrics import r2_score 
        r2 = r2_score(y_test, pred_y)

        lg.logger.info(f'Max_depth: {max_depth}, min_samples_split: {min_samples_split}, min_samples_leaf: {min_samples_leaf}, criterion: {criterion}, max_leaf_nodes: {max_leaf_nodes}, Test score: {score}, mse: {mse}')

    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    values = estimator.tree_.value

    from sklearn import tree
    tree.plot_tree(estimator)
    import matplotlib.pyplot as plt
    plt.savefig('dt.png')
    plt.show()

    y = np.concatenate((y_train, y_test))
    target_classes = np.unique(y)

    fu.saveDTParams(n_nodes, children_left, children_right, feature, threshold, values, target_classes)
    fu.savePPParams()