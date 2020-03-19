# Name of the dataset
ds_name = ''

# Algorithm
algo = ''

# Regression
regr = False

# Percentage for the test set
test_size = 0.3

# Evaluation metrics
metrics = []

# Loss function
loss = ''

# Training
epochs = 10
batch_size = 10
dropout = None

# Cross validation
n_folds = 5

# Controlling for model variance
# https://machinelearningmastery.com/evaluate-skill-deep-learning-models/
repeats = 1

# Principal component analysis
pca = '' #'mle' 

# csv file
delimiter = ''
decimal = ','
skiprows = 0
endColumn = 0
targetColumn = 0
startColumn = 0
nClass = 0

# Number of test items to be exported
nTests = None

# K-NN Params
cvl_knn = 1 # Cross Validation lower limit
cvu_knn = 15 # Cross Validation lower limit
training_set_cap = 100 # K-NN brings the training set to the target, which consumes memory # None

# SVM Params
svm_param_grid = {'C':[0.01, 0.1, 1, 10, 100]}

# Normalization
normalization = 'Standard'#'MinMax' #'Standard' #None

export_dir = 'E:/2009 Articles/Applepies/2019/ML/Journal/Edge-CAI'
ds_test = True

def config(ds, alg):
    global ds_name
    global algo
    global regr
    global test_size
    global metrics
    global loss
    global epochs
    global batch_size
    global dropout
    global n_folds
    global repeats

    global pca

    global delimiter, decimal, skiprows, endColumn, targetColumn, startColumn, nClass

    global nTests

    global cvl_knn
    global cvu_knn
    global training_set_cap

    global svm_param_grid

    global normalization

    global export_dir
    global ds_test

    algo = alg

    if ds == 'energydata_complete':
        # Regression energy
        ds_name = 'energydata_complete'
        regr = True
        metrics = ['mean_squared_error']
        loss = 'mean_squared_error'
        epochs=1 #10
        batch_size=100 #10
        n_folds = 2 #5
        repeats = 1

        delimiter = ';'
        pca='mle'

        skiprows = 1
        endColumn = 30
        targetColumn=2
        startColumn=3

        nTests = 10

        # K-NN
        cvl_knn = 1
        cvu_knn = 25

        # SVR
        svm_param_grid = {'C':[0.01]} #svm_param_grid = {'C':[0.01, 0.1]}

    elif ds == 'peugeot_207_01':
        # Classify car/road type multiclass
        ds_name = 'peugeot_207_01'
        metrics = ['accuracy'] #['categorical_accuracy']
        loss='sparse_categorical_crossentropy'
        epochs=100 #150
        batch_size=10
        n_folds = 2 #10
        repeats = 1 #5

        delimiter = ';'
        pca = 'mle' # None

        skiprows = 1
        endColumn = 14
        targetColumn=None
        startColumn=0
        nClass = 3

        nTests = 10

        # K-NN
        cvl_knn = 1
        cvu_knn = 10

        # SVM
        svm_param_grid = {'C':[0.1, 1, 10]}

    elif ds == 'heart':
        # Classify heart (binary)
        ds_name = 'heart'
        metrics = ['binary_accuracy']
        loss='binary_crossentropy'
        epochs=1#20 #50
        batch_size=100#10
        #dropout = 0.1
        n_folds = 3#10 #5
        repeats = 1#3 #1 #5

        delimiter = ','
        pca=4 #None #5

        skiprows = 1
        endColumn = 13
        targetColumn=None
        startColumn=0
        nClass = 2

        nTests = 10#'full'#None#10

        # K-NN
        cvl_knn = 1
        cvu_knn = 15

        #SVM
        svm_param_grid = {'C':[0.001, 0.01, 0.1, 1, 10, 100]}