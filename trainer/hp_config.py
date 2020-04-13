import trainer.constants as cst
from tensorboard.plugins.hparams import api as hp

"""
Hyperparameter configurations for every available model.
Comment out any lines you don't want to include in the 
hyperparameter optimization - they will default to the
scalar values defined in the matching model file.

Only hp.Discrete() works with hpo gridsearch,
avoid hp.RealInterval and hp.IntInterval.
"""

# Hérna stillir maður hvaða parametra maður vill leita af og bera saman í Hparams.

full_cnn_model_hparams = [
   hp.HParam(cst.CONV_FILTERS, hp.Discrete([8, 32])),
   hp.HParam(cst.CONV_KERNEL, hp.Discrete([9])),
   hp.HParam(cst.CONV_KERNEL_2D, hp.Discrete([3])),
   hp.HParam(cst.CONV_KERNEL_2D_X_1, hp.Discrete([1])),
   hp.HParam(cst.CONV_KERNEL_2D_Y_1, hp.Discrete([3, 9])),
   hp.HParam(cst.CONV_KERNEL_2D_X_2, hp.Discrete([1])),
   hp.HParam(cst.CONV_KERNEL_2D_Y_2, hp.Discrete([3, 9])),
   hp.HParam(cst.CONV_KERNEL_2D_X_3, hp.Discrete([1])),
   hp.HParam(cst.CONV_KERNEL_2D_Y_3, hp.Discrete([3,9])),
   hp.HParam(cst.PADDING, hp.Discrete(['same', 'valid'])),
   hp.HParam(cst.CONV_STRIDE, hp.Discrete([1])),
   hp.HParam(cst.CONV_STRIDE_2D, hp.Discrete([1])),
   hp.HParam(cst.CONV_STRIDE_2D_X, hp.Discrete([1])),
   hp.HParam(cst.CONV_STRIDE_2D_Y, hp.Discrete([3])),
   hp.HParam(cst.CONV_ACTIVATION, hp.Discrete(['relu'])),
   hp.HParam(cst.DENSE_NUM_UNITS, hp.Discrete([32])),
   hp.HParam(cst.DENSE_NUM_UNITS_1, hp.Discrete([64])),
   hp.HParam(cst.DENSE_NUM_UNITS_2, hp.Discrete([256])),
   hp.HParam(cst.DENSE_ACTIVATION, hp.Discrete(['relu'])),
   hp.HParam(cst.LEARNING_RATE, hp.Discrete([0.0001, 0.00001])),
   hp.HParam(cst.DROPOUT_RATE_CNN, hp.Discrete([0.3, 0.4])),
   ]


split_model_hparams = [
   #hp.HParam(cst.CONV_KERNEL, hp.Discrete([9, 27])),
   #hp.HParam(cst.CONV_FILTERS, hp.Discrete([8, 32])),
   #hp.HParam(cst.LSTM_NUM_UNITS, hp.Discrete([64, 128])),
   hp.HParam(cst.DENSE_NUM_UNITS, hp.Discrete([32])),
   hp.HParam(cst.OUTPUT_ACTIVATION, hp.Discrete(['relu', 'sigmoid'])),
   hp.HParam(cst.LEARNING_RATE, hp.Discrete([0.0001, 0.00001, 0.000005])),
   hp.HParam(cst.DROPOUT_RATE_CNN, hp.Discrete([0.2, 0.3, 0.4])),
   # hp.HParam(cst.DROPOUT_RATE_LSTM, hp.Discrete([0.3, 0.4])),
   hp.HParam(cst.CONV_KERNEL_2D, hp.Discrete([3, 9])),
   hp.HParam(cst.CONV_STRIDE_2D, hp.Discrete([1, 3])),
   ]
