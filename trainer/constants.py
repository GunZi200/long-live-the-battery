from os.path import join

# Feature dimensions
STEPS = 1000  # number of steps in detail level features, e.g. Qdlin and Tdlin
INPUT_DIM = 1  # dimensions of detail level features, e.g. Qdlin and Tdlin

# Feature names - use these for matching features in dataset with model inputs
INTERNAL_RESISTANCE_NAME = 'IR'
QD_NAME = 'QD'
DISCHARGE_TIME_NAME = 'Discharge_time'
TDLIN_NAME = 'Tdlin'
QDLIN_NAME = 'Qdlin'
VDLIN_NAME = 'Vdlin'
REMAINING_CYCLES_NAME = 'Remaining_cycles'
CURRENT_CYCLE_NAME = 'Current_cycle'

# File paths
TRAIN_TEST_SPLIT = 'train_test_split.pkl'  # file location for train/test split definition
PROCESSED_DATA = join('data', 'processed_data.pkl')  # file location for processed data
DATASETS_DIR = join('data', 'tfrecords')  # base directory to write tfrecord files in
TENSORBOARD_DIR = 'Graph'  # base directory to write tensorboard logs in
SAVED_MODELS_DIR_LOCAL = 'saved_models'  # base directory to save trained model in
BASE_DIR = './'  # home directory
TRAIN_SET = join(DATASETS_DIR, 'train', '*tfrecord')  # regexp files for the training set
TEST_SET = join(DATASETS_DIR, 'test', '*tfrecord')  # regexp for the test set
SECONDARY_TEST_SET = join(DATASETS_DIR, 'secondary_test', '*tfrecord')  # regexp for the secondary test set
BIG_TRAIN_SET = join(DATASETS_DIR, 'train_big', '*tfrecord')  	 # regexp for the combined training set (train + 1st test sets)
SCALING_FACTORS_DIR = join(DATASETS_DIR, 'scaling_factors.csv')  # location for scaling factors for tfrecords files
DATA_DIR = 'data'
MODEL_DIR = "trainer/saved_model"
RESULTS_DIR = join('data','tfrecords')
SAMPLES_DIR = join('static','samples')

# Hyperparameter names
CONV_KERNEL = 'conv_kernel'
CONV_FILTERS = 'conv_filters'
CONV_STRIDE = 'conv_stride'
CONV_ACTIVATION = 'conv_activation'
LSTM_NUM_UNITS = 'lstm_num_units'
LSTM_ACTIVATION = 'lstm_activation'
DENSE_NUM_UNITS = 'dense_num_units'
DENSE_ACTIVATION = 'dense_activation'
OUTPUT_ACTIVATION = 'output_activation'
LEARNING_RATE = 'learning_rate'
DROPOUT_RATE_CNN = 'dropout_cnn'
DROPOUT_RATE_LSTM = 'dropout_lstm'

# Loading data parameters
WINDOW_SIZE = 20
BATCH_SIZE = 32
NUM_EPOCHS = 3
SHIFT = 5
STRIDE = 1
NUM_SAMPLES = 6

# unique full_cnn_model parameters
CONV_KERNEL_2D = 'conv_kernel_2d'
CONV_STRIDE_2D = 'conv_stride_2d'