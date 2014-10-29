# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import cv2

import matplotlib.pyplot as plt

import random, itertools, sys, os
from multiprocessing import Pool
import json

import utilities
from utilities import chi2

from joblib import Parallel, delayed

import glob, re, os, sys, subprocess, argparse
import pprint
import cPickle as pickle

from skimage.color import hsv2rgb, label2rgb, gray2rgb

# <codecell>

# parser = argparse.ArgumentParser(
# formatter_class=argparse.RawDescriptionHelpFormatter,
# description='Semi-supervised Sigboost',
# epilog="""%s
# """%(os.path.basename(sys.argv[0]), ))

# parser.add_argument("stack_name", type=str, help="stack name")
# parser.add_argument("resolution", type=str, help="resolution string")
# parser.add_argument("slice_num", type=str, help="slice number, zero-padded to 4 digits")
# parser.add_argument("param_id", type=str, help="parameter identification name")
# args = parser.parse_args()

class args:
    stack_name = 'RS141'
    resolution = 'x5'
    slice_id = '0000'
    params_name = 'redNissl'

data_dir = '/home/yuncong/BrainLocal/DavidData_v3'
repo_dir = '/home/yuncong/BrainSaliencyDetection'
params_dir = os.path.join(repo_dir, 'params')

img_dir = os.path.join(data_dir, args.stack_name, args.resolution, args.slice_id)
results_dir = os.path.join(img_dir, args.params_name + '_pipelineResults')
labelings_dir = os.path.join(img_dir, 'labelings')

image_name = '_'.join([args.stack_name, args.resolution, args.slice_id])
instance_name = '_'.join([args.stack_name, args.resolution, args.slice_id, args.params_name])

im = cv2.imread(os.path.join(img_dir, image_name + '.tif'), 0)
img = utilities.regulate_img(im)

cropped_img = cv2.imread(full_object_name('cropImg', 'tif'), 0)
cropped_mask = np.load(full_object_name('cropMask', 'npy'), 0) > 0

cropped_masked_img = cropped_img.copy()
cropped_masked_img[~cropped_mask] = 0

def full_object_name(obj_name, ext):
    return os.path.join(data_dir, args.stack_name, args.resolution, args.slice_id, 
                        args.params_name+'_pipelineResults', instance_name + '_' + obj_name + '.' + ext)


# load parameter settings
params_dir = os.path.realpath(params_dir)
param_file = os.path.join(params_dir, 'param_%s.json'%args.params_name)
param_default_file = os.path.join(params_dir, 'param_default.json')
param = json.load(open(param_file, 'r'))
param_default = json.load(open(param_default_file, 'r'))

for k, v in param_default.iteritems():
    if not isinstance(param[k], basestring):
        if np.isnan(param[k]):
            param[k] = v

pprint.pprint(param)

# <codecell>

textonmap = np.load(full_object_name('texMap', 'npy'))
masked_textonmap = textonmap.copy()
masked_textonmap[cropped_mask] = -1

# <codecell>

texmap_vis = label2rgb(textonmap, bg_label=-1)

# <codecell>

plt.imshow(cropped_mask, cmap=plt.cm.Greys_r)
plt.show()

# <codecell>

plt.imshow(cropped_masked_img, cmap=plt.cm.Greys_r)
plt.show()

# <codecell>

plt.imshow(texmap_vis)
plt.show()

