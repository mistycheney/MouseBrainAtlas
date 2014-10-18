# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

# <codecell>

import numpy as np
import cv2

import matplotlib.pyplot as plt

import random, itertools, sys, os
from multiprocessing import Pool
import json

from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops
from skimage.util import img_as_ubyte
from skimage.color import hsv2rgb, label2rgb, gray2rgb, rgb2gray
from skimage.morphology import disk
from skimage.filter.rank import gradient
from skimage.filter import gabor_kernel
from skimage.transform import rescale, resize

from scipy.ndimage import gaussian_filter, measurements
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, squareform, euclidean, cdist
from scipy.signal import fftconvolve

from IPython.display import FileLink, Image, FileLinks

import utilities
from utilities import chi2

from joblib import Parallel, delayed

import cPickle as pickle

import glob, re, os, sys, subprocess, argparse
import pprint

def load_array(suffix):
    return utilities.load_array(suffix, instance_name=instance_name, results_dir=results_dir)

def save_array(arr, suffix):
    utilities.save_array(arr, suffix, instance_name=instance_name, results_dir=results_dir)
        
def save_image(img, suffix):
    utilities.save_image(img, suffix, instance_name=instance_name, results_dir=results_dir, overwrite=True)

def load_image(suffix):
    return utilities.load_image(suffix, instance_name=instance_name, results_dir=results_dir)

    
data_dir = '/home/yuncong/BrainLocal/DavidData'
repo_dir = '/home/yuncong/BrainSaliencyDetection'
params_dir = os.path.join(repo_dir, 'params')


# stack_name, resolution, slice_id, params_name, username, logout_time = os.path.basename(args.labeling_fn)[:-4].split('_')

# stack_name = args.stack_name
# resolution = args.resolution
# slice_id = args.slice_num
# params_name = args.params_name

stack_name = 'RS141'
resolution = 'x5'
slice_id = '0001'
params_name = 'redNissl'

results_dir = os.path.join(data_dir, stack_name, resolution, slice_id, params_name, 'pipelineResults')
labelings_dir = os.path.join(data_dir, stack_name, resolution, slice_id, params_name, 'labelings')

instance_name = '_'.join([stack_name, resolution, slice_id, params_name])
# parent_labeling_name = username + '_' + logout_time
parent_labeling_name = None

def full_object_name(obj_name, ext):
    return os.path.join(data_dir, stack_name, resolution, slice_id, params_name, 'pipelineResults', instance_name + '_' + obj_name + '.' + ext)

segmentation = np.load(full_object_name('segmentation', 'npy'))
n_superpixels = np.max(segmentation) + 1

# load parameter settings
params_dir = os.path.realpath(params_dir)
param_file = os.path.join(params_dir, 'param_%s.json'%params_name)
param_default_file = os.path.join(params_dir, 'param_default.json')
param = json.load(open(param_file, 'r'))
param_default = json.load(open(param_default_file, 'r'))

for k, v in param_default.iteritems():
    if not isinstance(param[k], basestring):
        if np.isnan(param[k]):
            param[k] = v

pprint.pprint(param)

# <codecell>

# Over-segment the image into superpixels using SLIC (http://ivrg.epfl.ch/research/superpixels)

print '=== over-segment the image into superpixels based on color information ==='

img_rgb = load_image('cropImg')
img = rgb2gray(img_rgb)

try:
    raise IOError
#     segmentation = load_array('segmentation')
    
except IOError:
    segmentation = slic(img_rgb, n_segments=int(param['n_superpixels']), 
                        max_iter=10, 
                        compactness=float(param['slic_compactness']), 
                        sigma=float(param['slic_sigma']), 
                        enforce_connectivity=True)
    print 'segmentation computed'
    
    save_array(segmentation.astype(np.int16), 'segmentation')
    
sp_props = regionprops(segmentation+1, intensity_image=img, cache=True)

def obtain_props_worker(i):
    return sp_props[i].centroid, sp_props[i].area, sp_props[i].mean_intensity

r = Parallel(n_jobs=16)(delayed(obtain_props_worker)(i) for i in range(len(sp_props)))

sp_centroids = np.array([s[0] for s in r])
sp_areas = np.array([s[1] for s in r])
sp_mean_intensity = np.array([s[2] for s in r])

n_superpixels = len(np.unique(segmentation))

img_superpixelized = mark_boundaries(img_rgb, segmentation)
sptext = img_as_ubyte(img_superpixelized)
for s in range(n_superpixels):
    sptext = cv2.putText(sptext, str(s), 
                      tuple(np.floor(sp_centroids[s][::-1]).astype(np.int)), 
                      cv2.FONT_HERSHEY_COMPLEX_SMALL,
                      .5, ((255,0,255)), 1)
save_img(sptext, 'segmentation')

