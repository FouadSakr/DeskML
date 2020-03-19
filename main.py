import argparse

from sklearn.model_selection import train_test_split

import preProcess as pp
import model as mdl
import config as cfg
import fileUtils as fu

import logger as lg

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dset')
parser.add_argument('-a', '--algo')
# parser.add_argument('-r', action='store_true')
args = parser.parse_args()

cfg.config(args.dset, args.algo)
# cfg.config('energydata_complete')
# cfg.config('peugeot_207_01')
lg.initLogger(cfg.ds_name, cfg.algo)

X, y = pp.preprocess('../datasets/' + cfg.ds_name + '.csv', cfg.delimiter, cfg.skiprows, cfg.endColumn, startColumn= cfg.startColumn, targetColumn = cfg.targetColumn, pca = cfg.pca, decimal = cfg.decimal) 
train_size = None
if cfg.algo.lower() == 'k-nn':
    if cfg.training_set_cap != None:
        if X.shape[0] * (1 - cfg.test_size) > cfg.training_set_cap:
            train_size = cfg.training_set_cap / X.shape[0]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=cfg.test_size, random_state=1)

fu.cleanSIDirs('./out/')

if cfg.algo.lower() == 'ann':
    import ANN as ann
    ann.process(X_train, X_test, y_train, y_test)      
elif cfg.algo.lower() == 'k-nn':
    import KNN as knn
    knn.process(X_train, X_test, y_train, y_test)
elif cfg.algo.lower() == 'svm':
    import SVM as svm
    svm.process(X_train, X_test, y_train, y_test)
elif cfg.algo.lower() == 'dt':
    import DT as dt
    dt.process(X_train, X_test, y_train, y_test)

if cfg.ds_test:
    if cfg.nTests == 'full':
        fu.saveTestingSet(X_test, y_test) 
    elif cfg.nTests != None:
        fu.saveTestingSet(X_test[0:cfg.nTests], y_test[0:cfg.nTests], full=False) 
if cfg.algo.lower() == 'k-nn':
    fu.saveTrainingSet(X_train, y_train)
if cfg.export_dir != None:
    from distutils.dir_util import copy_tree
    '''
    if cfg.ds_test:
        fromDirectory = f"./out/{cfg.ds_name}/include/{cfg.algo}"
        toDirectory = f"{cfg.export_dir}/{cfg.ds_name}/include/{cfg.algo}"
        copy_tree(fromDirectory, toDirectory)
    else:
        fromDirectory = f"./out/include/{cfg.algo}"
        toDirectory = f"{cfg.export_dir}/include/{cfg.algo}"
        copy_tree(fromDirectory, toDirectory)
    '''
    fu.cleanSIDirs(f'{cfg.export_dir}/')
    fromDirectory = f"./out/include"
    toDirectory = f"{cfg.export_dir}/include"
    copy_tree(fromDirectory, toDirectory)
    fromDirectory = f"./out/source"
    toDirectory = f"{cfg.export_dir}/source"
    copy_tree(fromDirectory, toDirectory)