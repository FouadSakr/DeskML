# Desk-LM
Desk-LM is a python environment for training machine learning models. It currently implements the following ML algorithms:

- Linear SVM
- Decision Tree
- K-NN
- ANN 

We are extending the library to other algorithms, also unsupervised. Your voluntary contribution is welcome.

The user can specify a .csv dataset, an algorithm and a set of parameters, so to train and select the best model and export it for use on edge devices, by exploiting the twin tool Micro-LM.

For ANNs, Desk-LM outputs the model in hdf5 file format, to be imported by STM32 CubeAI, together with some .h files that could be useful for testing the whole dataset performance on the microcontroller (STM32 Nucleo boards only).

For all the other algorithms, Desk-ML produces .c and .h that will be used as source files in a Edge-LM project for optimzed memory footprint on edge devices. They contain the parameters of the selected ML model.

We are working so that Desk-ML will output .json files so to allow dynamic usage by microcontrollers.

## Reference article for more infomation
F., Sakr, F., Bellotti, R., Berta, A., De Gloria, "Machine Learning on Mainstream Microcontrollers," Sensors 2020, 20, 2638.
https://www.mdpi.com/1424-8220/20/9/2638

## Usage

### Input
#### Dataset
Expected csv file in ../ELM-datasets/

\<ds_name\> is the name of the dataset (ds_name does not include the .csv extension)

#### Configuration files
config.py exposes the characteristics the dataset to be processed and the parameters to be analyzed for a selected algorithm (e.g., 'mle' algorithm for automatic PCA), training and cross-validation

For Linear SVM:
- SVMConfig.py: configuration file for Linear SVM contains the parameters to be defined by the user.

For K-NN:
- SVMConfig.py: configuration file for K-NN contains the parameters to be defined by the user.

For Decison Tree:
- SVMConfig.py: configuration file for Decision Tree contains the parameters to be defined by the user.

For ANN only:
- ANNConfig.py: configuration file for ANN contains the parameters to be defined by the user.
- ./ANNconfig/ActivationFunction.dat specifies the various activation functions that could be used in all the layers
- /ANNconfig/LayerShape.dat specifies the possible shapes of the layers of the ANN.

### Output

#### Linear SVM / DT / K-NN 
In './out/source/' and in './out/include/', the .c and .h files are generated, that contain the selected model parameters, that need to be compiled in a Edge-LM project.

The same output is also provided under:
'./out/' + cfg.ds_name + '/include/' + cfg.algo.lower() + '/'

#### ANN
In './out/source/', the preprocess_params.c file is saved
In './out/include/', the preprocess_params.h file is saved, together with files for dataset testing and the ANN model in hdf5 format

The same output is also provided under:
'./out/' + cfg.ds_name + '/include/' + cfg.algo.lower() + '/', the ANN model in hdf5 format is saved

### Log
./\<ds_name\>.log, log file for each dataset 

## Data type
float 32 data are used

## Run
python main.py

python gpus.py - Get names of the enabled GPUs 

## Needed packages:
$pip install freeze
$pip freeze > requirements.txt 
$pip install -r requirements.txt
