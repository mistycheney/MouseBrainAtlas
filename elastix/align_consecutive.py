#!/usr/bin/env python

import os 
import numpy as np
import sys
from skimage.io import imread, imsave
import time
import re
import cPickle as pickle
from skimage.transform import warp, AffineTransform

stack = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]
first_moving_secind = int(sys.argv[4])
last_moving_secind = int(sys.argv[5])
suffix = 'thumbnail'

all_files = dict(sorted([(int(img_fn[:-4].split('_')[1]), img_fn) for img_fn in os.listdir(input_dir) if suffix in img_fn]))

if not os.path.exists(output_dir):
	os.makedirs(output_dir)

n_sections = len(os.listdir(input_dir))

rg_param = os.environ['REPO_DIR'] + "/elastix/parameters/Parameters_Rigid.txt"

for moving_secind in range(first_moving_secind, last_moving_secind+1):
	if moving_secind - 1 in all_files:

		d = {'elastix_bin': os.environ['GORDON_ELASTIX'], 
			'rg_param': rg_param,
			'output_subdir': os.path.join(output_dir, 'output%dto%d'%(moving_secind, moving_secind-1)),
			'fixed_fn': os.path.join(input_dir, all_files[moving_secind-1]),
			'moving_fn': os.path.join(input_dir, all_files[moving_secind])
			}

		if os.path.exists(d['output_subdir']):
			os.system('rm -r ' + d['output_subdir'])
		os.makedirs(d['output_subdir'])

		os.system('%(elastix_bin)s -f %(fixed_fn)s -m %(moving_fn)s -out %(output_subdir)s -p %(rg_param)s' % d)