#!/usr/bin/env python

import os 
import numpy as np
import sys
from skimage.io import imread, imsave
import time
import re
import cPickle as pickle
from skimage.transform import warp, AffineTransform

# sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], 'notebooks'))
# from utilities2014 import execute_command, create_if_not_exists

stack = sys.argv[1]

# moving_secind = int(sys.argv[2])
first_moving_secind = int(sys.argv[2])
last_moving_secind = int(sys.argv[3])

DATA_DIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_data'

input_dir = os.path.join(DATA_DIR, stack + '_thumbnail_padded')
output_dir = os.path.join(DATA_DIR, stack + '_thumbnail_output')

if not os.path.exists(output_dir):
	os.makedirs(output_dir)

n_sections = len(os.listdir(input_dir))

rg_param = os.environ['GORDON_REPO_DIR'] + "/elastix/parameters/Parameters_Rigid.txt"

for moving_secind in range(first_moving_secind, last_moving_secind+1):

	d = {'elastix_bin': os.environ['GORDON_ELASTIX'], 
		'rg_param': rg_param,
		'output_subdir': os.path.join(output_dir, 'output%dto%d'%(moving_secind, moving_secind-1)),
		'fixed_fn': os.path.join(input_dir, stack+'_%04d'%(moving_secind-1)+'_thumbnail_trimmed_padded.tif'),
		'moving_fn': os.path.join(input_dir, stack+'_%04d'%(moving_secind)+'_thumbnail_trimmed_padded.tif')
		}

	if not os.path.exists(d['output_subdir']):
		os.makedirs(d['output_subdir'])

	os.system('%(elastix_bin)s -f %(fixed_fn)s -m %(moving_fn)s -out %(output_subdir)s -p %(rg_param)s' % d)