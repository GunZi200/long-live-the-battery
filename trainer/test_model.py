import trainer.constants as cst
from trainer.evaluation import get_predictions_results
import tensorflow as tf
from trainer.data_pipeline import create_dataset
import server.clippy as clippy
from trainer.custom_metrics_losses import mae_current_cycle, mae_remaining_cycles
import pandas as pd

## START CODE HERE
dataset = create_dataset(cst.SECONDARY_TEST_SET,
                         window_size=20,
                         shift=1,
                         stride=1,
                         batch_size=1)
dateset = dataset.take(1)
scaling_factors = pd.read_csv(cst.SCALING_FACTORS_DIR)
scaling_factors_dict = scaling_factors.to_dict('records')			 
dependencies = {
	'clippy': clippy.Clippy(clippy.clipped_relu),
    'mae_current_cycle': mae_current_cycle,
    'mae_remaining_cycles': mae_remaining_cycles}

print('Loading Keras Model')
model = tf.keras.models.load_model(cst.MODEL_DIR, custom_objects=dependencies)		

print('Getting results')
results_df = get_predictions_results(model,dataset,scaling_factors_dict)
print('Writing results')		
results_df.to_csv(cst.RESULTS_DIR,'results.csv')		 