import itertools
from os.path import join
import datetime

import tensorflow as tf
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except: 
    print("Error in task.py. Could not set memory growth for GPU.")
from tensorboard.plugins.hparams import api as hp

import trainer.constants as cst
import trainer.task as task
from trainer.hp_config import split_model_hparams, full_cnn_model_hparams



        
        
def get_hyperparameter_grid(hyperparameters):
    keys = [param.name for param in hyperparameters]
    values = [param.domain.values for param in hyperparameters]
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

def run(run_dir, hparams, args):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        mae_current, mae_remaining = task.train_and_evaluate(args, run_dir, hparams)
        tf.summary.scalar('current_mae', mae_current, step=1)
        tf.summary.scalar('remaining_mae', mae_remaining, step=1)

def grid_search(args):    
    run_timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.tboard_dir is None:
        tboard_dir = join(cst.TENSORBOARD_DIR, "gridsearches", run_timestr + "_gridsearch")
    else:
        tboard_dir = join(args.tboard_dir + "_gridsearch")
        
    # to pick parameters that are iterated over, edit hp_config.py
    hyperparameters = full_cnn_model_hparams
    
    session_num = 0
    for hparams in get_hyperparameter_grid(hyperparameters):
        print("RUN TIMESTR: {}".format(run_timestr))
        run_name = "run-{}_{}".format(session_num, run_timestr)
        print('--- Starting trial: {}'.format(run_name))
        print({h: hparams[h] for h in hparams})

        # RUN()
        run(join(tboard_dir, run_name), hparams, args)
        session_num += 1
            
            
if __name__ == "__main__":                
    args = task.get_args()
    grid_search(args)
    