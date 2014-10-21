"""

"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import cPickle as pickle
import os
import sys
import subprocess
import pprint
    
data_dir = sys.argv[1]
repo_dir = '/home/yuncong/BrainSaliencyDetection'

external_nlevel = len(data_dir.split('/'))
for path, folders, files in os.walk(data_dir):

	print path
	os.chdir(path)

	all_segments = path.split('/')
	internal_nlevel = len(all_segments) - external_nlevel
	if internal_nlevel == 4 and all_segments[-1] == 'labelings':
		for labeling_fn in files:

 			if 'preview' in labeling_fn or 'names' in labeling_fn: # not a labeling file
 				continue

			print labeling_fn

			stack_name, resolution, slice_id, username, logout_time = labeling_fn[:-4].split('_')

			labeling = pickle.load(open(labeling_fn, 'r'))

			instance_name = '_'.join([stack_name, resolution, slice_id, 'redNissl'])

			segmentation_fn = os.path.join(data_dir, stack_name, resolution, slice_id, 'redNissl_pipelineResults', instance_name + '_segmentation.npy')
			segmentation = np.load(segmentation_fn)

			if 'init_labellist' not in labeling: # already converted
				continue

			init_labellist = labeling['init_labellist']
			if init_labellist is not None:
				init_labelmap = init_labellist[segmentation].astype(np.int8)
			else:
				init_labelmap = None

			final_labellist = labeling['final_labellist']
			final_labelmap = final_labellist[segmentation].astype(np.int8)

			new_labeling = labeling.copy()
			new_labeling.pop('init_labellist', None)
			new_labeling.pop('final_labellist', None)
			new_labeling.pop('history', None)

			new_labeling['init_labelmap'] = init_labelmap
			new_labeling['final_labelmap'] = final_labelmap
			new_labeling['params_id'] = None

			new_labeling_fn = '_'.join([stack_name, resolution, slice_id, username, logout_time + '.pkl'])

			pickle.dump(new_labeling, open(new_labeling_fn, 'w'))
