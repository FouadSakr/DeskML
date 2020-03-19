import os
import numpy as np

import config as cfg
import preProcess as pp

def createArray(type, arrName, arr, n_elements):
    stri = f'{type} {arrName}[{n_elements}] = {{'
    for i, n in enumerate(arr):
        stri = stri + ' ' + str(n) + ', '
    stri = stri + '};\n'
    stri = stri.replace(', }', '}')
    return stri

def createMatrix(type, matName, mat, dim0, dim1):
    #stri = f'{type} {matName}[{mat.shape[0]}][{mat.shape[1]}] = {{ '
    stri = f'{type} {matName}[{dim0}][{dim1}] = {{ '
    for i, row in enumerate(mat):
        if i != 0:
            stri = stri + '\t\t\t'
        stri = stri + '{ '
        for j, val in enumerate(row):
            if type=='int':
                val = val.astype(int)
            stri = stri + str(val)
            if j < (row.size - 1):
                stri = stri + ', '
        stri = stri + ' }'
        if i < (mat.shape[0] - 1):
            stri = stri + ',\n'
    stri = stri + ' };\n'
    return stri

def createMatrix2(type, matName, mat, dim0, dim1):
    #stri = f'{type} {matName}[{mat.shape[0]}][{mat.shape[2]}] = {{ '
    stri = f'{type} {matName}[{dim0}][{dim1}] = {{ '
    for i, row in enumerate(mat):
        if i != 0:
            stri = stri + '\t\t\t'
        stri = stri + '{ '
        for j, val in enumerate(row[0]):
            if type=='int':
                val = int(val)
            stri = stri + str(val)
            if j < (row.size - 1):
                stri = stri + ', '
        stri = stri + ' }'
        if i < (mat.shape[0] - 1):
            stri = stri + ',\n'
    stri = stri + ' };\n'
    return stri

def saveModel(model, bestAF, shape):   
    outdir = checkCreateDSDir()
    '''outdir = './out/' + cfg.ds_name
    if os.path.isdir(outdir) == False:
        os.mkdir(outdir)'''
    fileName = f'{cfg.ds_name}_{cfg.pca}_{bestAF}'
    for s in shape:
        fileName += f'_{s}'
    model.save(outdir + f'/{fileName}.h5')

    outdirG = checkCreateGeneralIncludeDir()
    from shutil import copyfile
    copyfile(f"{outdir}{fileName}.h5", f"{outdirG}{fileName}.h5")

def saveTrainingSet(X_train, y_train):
    outdir = checkCreateDSDir()

    myFile = open(f"{outdir}training_set.h","w+")
    myFile.write(f"#define N_TRAIN {y_train.size}\n\n")
    myFile.write(f"#ifndef N_FEATURE\n")
    myFile.write(f"#define N_FEATURE {X_train.shape[1]}\n")
    myFile.write(f"#endif\n\n")

    if cfg.regr:
        type_s = 'float'
    else:
        type_s = 'int'
    myFile.write(f"extern {type_s} y_train[N_TRAIN];\n")
    myFile.write(f"extern float X_train[N_TRAIN][N_FEATURE];\n")
    myFile.close()
    
    outdirG = checkCreateGeneralIncludeDir()
    from shutil import copyfile
    copyfile(f"{outdir}training_set.h", f"{outdirG}training_set.h")

    outdirS = checkCreateGeneralSourceDir()
    myFile = open(f"{outdirS}training_set_params.c","w+")
    #myFile.write(f"#include \"AI_main.h\"\n")
    myFile.write(f"#include \"training_set.h\"\n")

    if cfg.regr:
        type_s = 'float'
    else:
        type_s = 'int'
    stri = createArray(type_s, "y_train", np.reshape(y_train, (y_train.size, )), 'N_TRAIN')
    myFile.write(stri)
    stri = createMatrix('float', 'X_train', X_train, 'N_TRAIN', 'N_FEATURE')
    myFile.write(stri)
    myFile.close()

def saveTestingSet(X_test, y_test, full=True):
    outdir = checkCreateDSDir()

    if full:
        myFile = open(f"{outdir}testing_set.h","w+")
        myFile.write(f"#ifndef TESTINGSET_H\n")
        myFile.write(f"#define TESTINGSET_H\n\n")
    else:
        myFile = open(f"{outdir}minimal_testing_set.h","w+")
        myFile.write(f"#ifndef MINIMAL_TESTINGSET_H\n")
        myFile.write(f"#define MINIMAL_TESTINGSET_H\n\n")
    myFile.write(f"#define N_TEST {y_test.size}\n\n")
    myFile.write(f"#ifndef N_FEATURE\n")
    myFile.write(f"#define N_FEATURE {X_test.shape[1]}\n")
    myFile.write(f"#endif\n\n")
    myFile.write(f"#ifndef N_ORIG_FEATURE\n")
    myFile.write(f"#define N_ORIG_FEATURE {pp.n_orig_features}\n")
    myFile.write(f"#endif\n\n")
    if cfg.regr:
        type_s = 'float'
    else:
        type_s = 'int'
    myFile.write(f"extern {type_s} y_test[N_TEST];\n")
    myFile.write(f"extern float X_test[N_TEST][N_FEATURE];\n")
    
    if cfg.normalization!=None and cfg.regr and cfg.algo.lower() != 'dt':
        saveTestNormalization(myFile)

    myFile.write(f"#endif")
    myFile.close()
    outdirG = checkCreateGeneralIncludeDir()
    from shutil import copyfile
    if full:
        copyfile(f"{outdir}testing_set.h", f"{outdirG}testing_set.h")
    else:
        copyfile(f"{outdir}minimal_testing_set.h", f"{outdirG}minimal_testing_set.h")

    outdirS = checkCreateGeneralSourceDir()
    if full:
        myFile = open(f"{outdirS}testing_set.c","w+")
    else:
        myFile = open(f"{outdirS}minimal_testing_set.c","w+")
    #myFile.write(f"#include \"AI_main.h\"\n")
    if full:
        myFile.write(f"#include \"testing_set.h\"\n")
    else:
        myFile.write(f"#include \"minimal_testing_set.h\"\n")

    if cfg.regr:
        type_s = 'float'
    else:
        type_s = 'int'
    stri = createArray(type_s, "y_test", np.reshape(y_test, (y_test.size, )), 'N_TEST')
    myFile.write(stri)
    
    stri = createMatrix('float', 'X_test', X_test, 'N_TEST', 'N_FEATURE')
    myFile.write(stri)
    myFile.close()

# Pre-procssing parameters
def savePPParams():
    outdir = checkCreateDSDir()

    myFile = open(f"{outdir}PPParams.h","w+")
    myFile.write(f"#ifndef PPPARAMS_H\n")
    myFile.write(f"#define PPPARAMS_H\n\n")

    myFile.write(f"#ifndef N_FEATURE\n")
    myFile.write(f"#define N_FEATURE {pp.pca_components.shape[0]}\n")
    myFile.write(f"#endif\n\n")
    myFile.write(f"#ifndef N_ORIG_FEATURE\n")
    myFile.write(f"#define N_ORIG_FEATURE {pp.pca_components.shape[1]}\n")
    myFile.write(f"#endif\n\n")
    myFile.write(f"extern float pca_components[N_FEATURE][N_ORIG_FEATURE];\n")
    myFile.write(f"\n")

    if cfg.normalization!=None and cfg.algo.lower() != 'dt':
        if cfg.normalization.lower()=='standard':
            myFile.write(f"#define STANDARD_NORMALIZATION\n\n")
            myFile.write(f"extern float s_x[N_ORIG_FEATURE];\n")
            myFile.write(f"extern float u_x[N_ORIG_FEATURE];\n")
        elif cfg.normalization.lower()=='minmax':
            myFile.write(f"#define MINMAX_NORMALIZATION\n\n")
            myFile.write(f"extern float s_x[N_ORIG_FEATURE];\n")

    if cfg.normalization!=None and cfg.regr and cfg.algo.lower() != 'dt':
        saveTestNormalization(myFile)

    myFile.write(f"#endif")
    myFile.close()
    outdirG = checkCreateGeneralIncludeDir()
    from shutil import copyfile
    copyfile(f"{outdir}PPParams.h", f"{outdirG}PPParams.h")

    outdirS = checkCreateGeneralSourceDir()
    myFile = open(f"{outdirS}preprocess_params.c","w+")
    #myFile.write(f"#include \"AI_main.h\"\n")
    myFile.write(f"#include \"PPParams.h\"\n")

    stri = createMatrix('float', 'pca_components', pp.pca_components, 'N_FEATURE', 'N_ORIG_FEATURE')
    myFile.write(stri)
    myFile.write(f"\n")

    if cfg.normalization!=None and cfg.algo.lower() != 'dt':
        if cfg.normalization.lower()=='standard':
            myFile.write(f"#define STANDARD_NORMALIZATION\n\n")
            stri = createArray('float', "s_x", np.reshape(pp.s_x, (pp.s_x.size, )), 'N_ORIG_FEATURE')
            myFile.write(stri)
            stri = createArray('float', "u_x", np.reshape(pp.u_x, (pp.u_x.size, )), 'N_ORIG_FEATURE')
            myFile.write(stri)
        elif cfg.normalization.lower()=='minmax':
            myFile.write(f"#define MINMAX_NORMALIZATION\n\n")
            stri = createArray('float', "s_x", np.reshape(pp.s_x, (pp.s_x.size, )), 'N_ORIG_FEATURE')
            myFile.write(stri)
    myFile.close()

def saveSVMParams(w, bias):
    outdir = checkCreateDSDir()

    myFile = open(f"{outdir}SVM_params.h","w+")
    myFile.write(f"#ifndef SVM_PARAMS_H\n")
    myFile.write(f"#define SVM_PARAMS_H\n\n")
    if cfg.regr == False:
        myFile.write(f"#define N_CLASS {cfg.nClass}\n\n")
        myFile.write(f"#define WEIGTH_DIM {w.shape[0]}\n\n")
        myFile.write(f"#ifndef N_FEATURE\n")
        myFile.write(f"#define N_FEATURE {w.shape[1]}\n")
        myFile.write(f"#endif\n\n")
    else:
        myFile.write(f"#define N_CLASS 0\n\n")
        myFile.write(f"#define WEIGTH_DIM 1\n\n")
        myFile.write(f"#ifndef N_FEATURE\n")
        myFile.write(f"#define N_FEATURE {w.size}\n")
        myFile.write(f"#endif\n\n")

    if cfg.regr == False:
        myFile.write(f"extern float support_vectors[WEIGTH_DIM][N_FEATURE];\n")
    else: #Same as above as a declaration
        myFile.write(f"extern float support_vectors[WEIGTH_DIM][N_FEATURE];\n")
    
    myFile.write(f"extern float bias[WEIGTH_DIM];\n")
    myFile.write(f"\n#endif")
    myFile.close()
    
    outdirG = checkCreateGeneralIncludeDir()
    from shutil import copyfile
    copyfile(f"{outdir}SVM_params.h", f"{outdirG}SVM_params.h")

    outdirS = checkCreateGeneralSourceDir()
    myFile = open(f"{outdirS}SVM_params.c","w+")
    #myFile.write(f"#include \"AI_main.h\"\n")
    myFile.write(f"#include \"SVM_params.h\"\n")

    if cfg.regr == False:
        stri = createMatrix('float', 'support_vectors', w, 'WEIGTH_DIM', 'N_FEATURE')
    else:
        stri = createMatrix("float", "support_vectors", np.reshape(w, (1, w.size)), 'WEIGTH_DIM', 'N_FEATURE')   
    myFile.write(stri)
    stri = createArray("float", "bias", bias, 'WEIGTH_DIM')
    myFile.write(stri)
    myFile.close()


def saveDTParams(n_nodes, children_left, children_right, feature, threshold, values, target_classes):
    outdir = checkCreateDSDir()

    myFile = open(f"{outdir}DT_params.h","w+")
    myFile.write(f"#ifndef DT_PARAMS_H\n")
    myFile.write(f"#define DT_PARAMS_H\n\n")
    if cfg.regr == False:
        myFile.write(f"#define N_CLASS {cfg.nClass}\n\n")
    else:
        myFile.write(f"#define N_CLASS 0\n\n")
    myFile.write(f"#define N_NODES {n_nodes}\n\n")
    myFile.write(f"#define VALUES_DIM {values.shape[2]}\n\n")

    myFile.write(f"extern int children_left[N_NODES];\n")
    myFile.write(f"extern int children_right[N_NODES];\n")
    myFile.write(f"extern int feature[N_NODES];\n")
    myFile.write(f"extern float threshold[N_NODES];\n")
    myFile.write(f"extern int values[N_NODES][VALUES_DIM];\n")
    if cfg.regr == False:
        myFile.write(f"extern int target_classes[N_CLASS];\n")
    myFile.close()

    outdirG = checkCreateGeneralIncludeDir()
    from shutil import copyfile
    copyfile(f"{outdir}DT_params.h", f"{outdirG}DT_params.h")

    outdirS = checkCreateGeneralSourceDir()
    myFile = open(f"{outdirS}DT_params.c","w+")
    #myFile.write(f"#include \"AI_main.h\"\n")
    myFile.write(f"#include \"DT_params.h\"\n")

    stri = createArray("int", "children_left", children_left, 'N_NODES')
    myFile.write(stri)
    stri = createArray("int", "children_right", children_right, 'N_NODES')
    myFile.write(stri)
    stri = createArray("int", "feature", feature, 'N_NODES')
    myFile.write(stri)
    stri = createArray("float", "threshold", threshold, 'N_NODES')
    myFile.write(stri)
    stri = createMatrix2('int', 'values', values, 'N_NODES', 'VALUES_DIM') 
    myFile.write(stri)
    if cfg.regr == False:
        stri = createArray("int", "target_classes", target_classes, 'N_CLASS')
        myFile.write(stri)
    myFile.write(f"\n#endif")
    myFile.close()


def saveKNNParams(k):
    outdir = checkCreateDSDir()

    myFile = open(f"{outdir}KNN_params.h","w+")
    myFile.write(f"#ifndef KNN_PARAMS_H\n")
    myFile.write(f"#define KNN_PARAMS_H\n\n")
    if cfg.regr == False:
        myFile.write(f"#define N_CLASS {cfg.nClass}\n\n")
    else:
        myFile.write(f"#define N_CLASS 0\n\n")
    myFile.write(f"#define K {k}\n\n")
    myFile.write(f"#endif")
    myFile.close()
    outdirG = checkCreateGeneralIncludeDir()
    from shutil import copyfile
    copyfile(f"{outdir}KNN_params.h", f"{outdirG}KNN_params.h")

def saveTestNormalization(myFile):
    #myFile.write(f"\n\n#ifndef TEST_NORMALIZATION\n")
    #myFile.write(f"#define TEST_NORMALIZATION\n\n")
    if cfg.normalization.lower()=='standard':
        #myFile.write(f"#ifndef STANDARD_NORMALIZATION\n")
        #myFile.write(f"#define STANDARD_NORMALIZATION\n\n")
        myFile.write(f"#define S_Y {pp.s_y[0]}\n")
        myFile.write(f"#define U_Y {pp.u_y[0]}\n\n")
        #myFile.write(f"#endif\n")
    elif cfg.normalization.lower()=='minmax':
        #myFile.write(f"#ifndef MINMAX_NORMALIZATION\n")
        #myFile.write(f"#define MINMAX_NORMALIZATION\n\n")
        myFile.write(f"#define S_Y {pp.s_y[0]}\n\n")
        #myFile.write(f"#endif\n")
    #myFile.write(f"#endif\n")

def checkCreateDSDir():
    outdir = './out/'
    if os.path.isdir(outdir) == False:
        os.mkdir(outdir)
    outdir = './out/' + cfg.ds_name
    if os.path.isdir(outdir) == False:
        os.mkdir(outdir)
    outdir = './out/' + cfg.ds_name + '/include/'
    if os.path.isdir(outdir) == False:
        os.mkdir(outdir)
    outdir = './out/' + cfg.ds_name + '/include/' + cfg.algo.lower() + '/'
    if os.path.isdir(outdir) == False:
        os.mkdir(outdir)
    return outdir

def checkCreateGeneralIncludeDir():
    outdir = './out/'
    if os.path.isdir(outdir) == False:
        os.mkdir(outdir)
    outdir = './out/include/'
    if os.path.isdir(outdir) == False:
        os.mkdir(outdir)
    return outdir

def checkCreateGeneralSourceDir():
    outdir = './out/'
    if os.path.isdir(outdir) == False:
        os.mkdir(outdir)
    outdir = './out/source/'
    if os.path.isdir(outdir) == False:
        os.mkdir(outdir)
    return outdir

def cleanSIDirs(path):
    import shutil
    shutil.rmtree(path+'/source/', ignore_errors=True)
    os.mkdir(path+'/source/')
    shutil.rmtree(path+'/include/', ignore_errors=True)
    os.mkdir(path+'/include/')