import json
import os
import tensorflow as tf
import trainer.constants as cst
import numpy as np
import server.clippy as clippy
from trainer.custom_metrics_losses import mae_current_cycle, mae_remaining_cycles

samples_fullpath = os.path.join('trainer',cst.SAMPLES_DIR)
num_samples = cst.NUM_SAMPLES
dependencies = {
		'clippy': clippy.Clippy(clippy.clipped_relu),
		'mae_current_cycle': mae_current_cycle,
		'mae_remaining_cycles': mae_remaining_cycles,
		'swish': tf.keras.activations.swish
				}
model = tf.keras.models.load_model(cst.MODEL_DIR, custom_objects=dependencies)

mae_curr = 0
mae_remain = 0

for i in range(num_samples):
	with open(os.path.join(samples_fullpath, 'sample_input_{}.json'.format(i+1)), 'r+') as infile:
		example = json.load(infile)
	with open(os.path.join(samples_fullpath, 'target_output_{}.json'.format(i+1)),'r+') as infile:
		target = json.load(infile)
	cycles = { 'Qdlin': np.array(json.loads(example['Qdlin'])),
				'IR': np.array(json.loads(example['IR'])),
				'Discharge_time': np.array(json.loads(example['Discharge_time'])),
				'QD': np.array(json.loads(example['QD']))
			}
	targets = np.array(json.loads(target))
	predictions = model.predict(cycles)
	mae_curr_temp = mae_current_cycle(targets,predictions)
	mae_remain_temp = mae_remaining_cycles(targets,predictions)
	mae_curr = mae_curr + mae_curr_temp
	mae_remain = mae_remain + mae_remain_temp
	#print('test_mae_current_cycles = {}'.format(mae_curr_temp))
	#print('test_mae_remaining_cycles = {}'.format(mae_remain_temp))

mae_curr = mae_curr/num_samples
mae_remain = mae_remain/num_samples

print('test_mae_current_cycles = {}'.format(mae_curr))
print('test_mae_remaining_cycles = {}'.format(mae_remain))