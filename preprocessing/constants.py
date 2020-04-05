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
PROCESSED_DATA = join('data', 'processed_data.pkl')  # file location for processed data