# Edge-ANN-Benchmarking
This program processes one dataset and evaluates the best ANN among a set of predefined models (specified in configuration). 

It outputs the model in hdf5 file format, to be imported by CubeAI, together with some .h files that could be useful for processing data on the microcontroller.

## Input
### Dataset
Expected csv file in ../datasets/

\<ds_name\> is the name of the dataset (ds_name does not include the .csv extension)

Datasets are available in: https://github.com/FranzGH/Edge-AI-datasets 

### Configuration files
config.py lists the characteristics of the dataset to be processed and parameters for preprocessing (e.g., 'mle' algorithm for automatic guess of PCA), training and cross-validation

./config/\<ds_name\>/activeFuncs.dat specifies the various activation functions that could be used in all the layers

./config/\<ds_name\>/layerShape.dat specifies the possible shapes of the layers of the ANN.

## Output
./out/model.h5

./out/\<ds_name\>/include/configNN.h: contains: (1) components of the PCA transform; (2) Normalization parameters u (mean) and s (stdev): z = (x-u)/s

./out/\<ds_name\>/include/testNN.h: test data and targets that may be used on the microcontroller (directly, without PCA and normalization, as they have already been applied)

## Log
./\<ds_name\>.log, log file for each dataset 

## Data type
float 32 data are used

## Run
main.py - Create code and data for the STM32 Nucleo

gpus.py - Get names of the enabled GPUs 

## Version
Currently working with Python 3.6, Keras 2.2.4 and Tensorflow 1.8.0
