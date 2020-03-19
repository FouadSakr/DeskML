
import config as cfg
import model as mdl
import fileUtils as fu
import preProcess as pp
import logger as lg

# https://scikit-learn.org/stable/modules/cross_validation.html

def process(X_train, X_test, y_train, y_test):
    bestS, bestAF = mdl.getBestModel(X_train, y_train, loss=cfg.loss, metrics=cfg.metrics)
    model = mdl.createModel(len(X_train[0]), bestS, bestAF, loss=cfg.loss, metrics=cfg.metrics)
    
    model.compile(loss=cfg.loss, optimizer='adam', metrics=cfg.metrics)
    
    # mdl.estimateScore(model, X, y, epochs=cfg.epochs, batch_size=cfg.batch_size, n_folds = cfg.n_folds, repeats = cfg.repeats)

    model.fit(X_train, y_train, batch_size=cfg.batch_size, epochs=cfg.epochs, verbose=2)
    _, score = model.evaluate(X_test, y_test)

    if cfg.regr == True:
        pred_y = model.predict(X_test)
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
        from sklearn.metrics import r2_score 
        r2 = r2_score(y_test, pred_y)
        lg.logger.info(f'Test score: {score}, R2: {r2}, mse: {mse}')
    else:
        lg.logger.info(f'Test score: {score}')
    
    fu.saveModel(model, bestAF, bestS)
    #fu.saveConfigNN_h() Now, substituted by savePPParams()
    fu.savePPParams()
    

    