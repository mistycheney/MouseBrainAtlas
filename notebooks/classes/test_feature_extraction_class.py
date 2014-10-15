# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

"""
This test uses FeatureExtractor class (defined in feature_extraction_class.py).

It shows the issue that in compute_texton() only a couple processes are utilized, 
thus making the computation very slow (20 sec vs. 1 sec).
"""

# <codecell>

import sys, os
import matplotlib.pyplot as plt

from IPython.display import FileLink

import cv2

import time

from utilities import *

from feature_extraction_class import *

class args(object):
    img_file = '../ParthaData/PMD1305_region0_reduce2/PMD1305_region0_reduce2_0244.tif'
    param_id = 'nissl324'
    output_dir ='/oasis/scratch/csd181/yuncong/output'
    params_dir = '/oasis/projects/nsf/csd181/yuncong/Brain/params'

# load parameter settings
params_dir = os.path.realpath(args.params_dir)
param_file = os.path.join(params_dir, 'param_%s.json'%args.param_id)
param_default_file = os.path.join(params_dir, 'param_default.json')
param = json.load(open(param_file, 'r'))
param_default = json.load(open(param_default_file, 'r'))

for k, v in param_default.iteritems():
    if not isinstance(param[k], basestring):
        if np.isnan(param[k]):
            param[k] = v

pprint.pprint(param)

# set image paths
img_file = os.path.realpath(args.img_file)
img_path, ext = os.path.splitext(img_file)
img_dir, img_name = os.path.split(img_path)

print img_file
img = cv2.imread(img_file, 0)
im_height, im_width = img.shape[:2]

# set output paths
output_dir = os.path.realpath(args.output_dir)

result_name = img_name + '_param_' + str(param['param_id'])
result_dir = os.path.join(output_dir, result_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# <codecell>

fe = FeatureExtractor(img, param)

# <codecell>

fe.compute_features()

# <codecell>

fe.compute_texton()

