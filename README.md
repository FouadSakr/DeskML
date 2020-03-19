# DeskML
This program processes a dataset using one of the four supported algorithms (ANN, SVM, K-NN, and DT) and evaluates it using a set of predefined parameters to get the best model. 

ANN: it outputs the model in hdf5 file format, to be imported by CubeAI, together with some .h files that are useful for processing data on the microcontroller.

SVM, K-NN, and DT: it outputs some .c and .h files to be imported by STM32CubeIDE to perform that are useful for processing data on the microcontroller to perform ML inferences.

## Input
### Algorithm and Dataset choice
launch.json lists the algorithms supported by the framework. User can add the name of his dataset and choose the wanted algorithm (comment the other algorithms and datasets)

### Dataset
Expected csv file in ../datasets/

### Configuration: 
config.py lists the characteristics of the dataset to be processed and parameters for preprocessing

#### Dataset Configuration
ds_name: is the name of the dataset (ds_name does not include the .csv extension)

regr: default false, set to true when it is a regression dataset 

metrics: 'binary_crossentropy' (i.e. binary classification), 'sparse_categorical_crossentropy' (i.e. multiclass classification) or 'mean_squared_error' (i.e. regression)

loss: 'binary_accuracy' (i.e. binary classification), 'accuracy' (i.e. multiclass classification) or 'mean_squared_error' (i.e. regression)

delimiter: ',' or ';' (according to the dataset)

decimal: '.' or ',' (how decimal numbers are expressed in the dataset)

skiprows: skip the first row in the dataset that contains the feature names (if this row exists)

startColumn: and endColumn: up to the user what columns he wants to process

targetColumn: column number of the target whether classification or regression

nClass: number of classes if classification 

pca: PCA algorithm for feature selection (None, 'mle' algorithm for automatic guess of PCA, or any number of features chosen by the user)

normalization: normalization technique (e.g. 'Standard', 'MinMax', or None)

nTests: number of testing samples to be saved for further processing at the edge 

#### ANN Configuration:
epochs: number of epochs for ANN

batch_size: number of training examples in one iteration

n_folds: number of folds for cross

repeats: repeat the training until no further change

dropout: to prevent model grom overfitting (user can choose the percentage of neuron to be turned off e.g. 0.2)

./config/\<ds_name\>/activeFuncs.dat specifies the various activation functions that could be used in all the layers

./config/\<ds_name\>/layerShape.dat specifies the possible shapes of the layers of the ANN.

#### SVM Configuration:
svm_param_grid: regularization parameter C (the algorithm tries all inputs and outputs the best C)

#### K-NN Configuration: 
cvl_knn and cvu_knn min and max number of neighbors to find the best K

training_set_cap: K-NN brings the training set to the target, which consumes memory (None if the user wants to save the whole training set, otherwize any number of samples e.g. 100, 200...)

#### DT Configuration:
DT.py lists the needed parameters for the algorithm and the user can change them (e.g. criterion, max_depth...)

Datasets are available in: https://github.com/FranzGH/Edge-AI-datasets 

## Output
### ANN: 
1- ./out/ds_name/include/ann/(the model name and configuration).h5: this output model is loaded in STM32CubeIDE to generate the C code to be deployed in microcontrollers. To know how to generate code visit this link: 

https://www.st.com/content/ccc/resource/technical/document/user_manual/group1/69/bb/ec/5d/78/16/43/ce/DM00570145/files/DM00570145.pdf/jcr:content/translations/en.DM00570145.pdf


2- ./out/ds_name/source: "minimal_testing_set.c" and "PPparams.c". 

3- ./out/ds_name/include: "minimal_testing_set.h" and "PPparams.h" 
These files represent the needed preprocessing parameters and the testing set

### SVM/K-NN/DT:
1- generated C files are in the following directory: ./out/source

2- generated h files are in the following directory: ./out/include

3- Follow the instruction in the README file in "Edge-CAI" using the following link:
https://github.com/FranzGH/Edge-CAI

## Log
./\<ds_name\>.log, log file for each dataset 

## Data type
float 32 data are used

## Run
main.py - Create code and data for the STM32 Nucleo

gpus.py - Get names of the enabled GPUs 

## Version
Currently working with Python 3.6, Keras 2.2.4 and Tensorflow 1.8.0
