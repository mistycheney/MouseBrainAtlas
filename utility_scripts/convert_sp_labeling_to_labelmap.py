"""

"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import label2rgb, rgb2gray
import utilities
from joblib import Parallel, delayed
import cPickle as pickle
import os
import sys
import subprocess
import pprint
    
data_dir = '/home/yuncong/BrainLocal/DavidData'
repo_dir = '/home/yuncong/BrainSaliencyDetection'

stack_name = 'RS141'
resolution = 'x5'
slice_id = '0001'
params_name = 'redNissl'

results_dir = os.path.join(data_dir, stack_name, resolution, slice_id, params_name, 'pipelineResults')
labelings_dir = os.path.join(data_dir, stack_name, resolution, slice_id, params_name, 'labelings')

instance_name = '_'.join([stack_name, resolution, slice_id, params_name])

def load_data(suffix, ext):
    if ext == 'tif' or ext == 'png':
        return utilities.load_image(suffix, instance_name=instance_name, results_dir=results_dir)
    else:
        return utilities.load_array(suffix, instance_name=instance_name, results_dir=results_dir)

def save_data(data, suffix, ext):
    if ext == 'tif' or ext == 'png':
        utilities.save_image(img, suffix, instance_name=instance_name, results_dir=results_dir, overwrite=True)
    else:
        utilities.save_array(arr, suffix, instance_name=instance_name, results_dir=results_dir)


labeling_fn = sys.argv[1]

with open(labeling_fn, 'r') as f:
    labeling = pickle.load(f)

segmentation = load_data('segmentation', 'npy')

init_labellist = labeling['init_labellist']
init_labelmap = init_labellist[segmentation]
final_labellist = labeling['final_labellist']
final_labelmap = final_labellist[segmentation]

new_labeling.pop('init_labellist', None)
new_labeling['init_labelmap'] = init_labelmap
new_labeling['final_labelmap'] = final_labelmap

