import json
import os
import tensorflow as tf
import trainer.constants as cst
from trainer.data_pipeline import create_dataset

samples_fullpath = os.path.join('trainer',cst.SAMPLES_DIR)

if not os.path.exists(samples_fullpath):
	os.makedirs(samples_fullpath)
	
dataset = create_dataset(cst.SECONDARY_TEST_SET,
							window_size=cst.WINDOW_SIZE,
							shift=cst.SHIFT,
							stride=cst.STRIDE,
							batch_size=cst.BATCH_SIZE)
rows = dataset.take(cst.NUM_SAMPLES)


i = 0
for example,target in rows.as_numpy_iterator():
	sample = {key: str(value.tolist()) for key, value in example.items()}
	targets = str(target.tolist())
	with open(os.path.join(samples_fullpath, 'sample_input_{}.json'.format(i+1)), 'w') as outfile:
		json.dump(sample, outfile)
	with open(os.path.join(samples_fullpath, 'target_output_{}.json'.format(i+1)),'w') as outfile:
		json.dump(targets,outfile)
	i = i+1
	print('Created sample file nr. {}'.format(i))

print("Created {} sample files in trainer/static/samples".format(cst.NUM_SAMPLES))