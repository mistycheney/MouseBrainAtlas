#!/usr/bin/env python

import os
import numpy as np
import sys
from skimage.io import imread, imsave
import time
import re
import cPickle as pickle
from skimage.transform import warp, AffineTransform

sys.path.append(os.environ['REPO_DIR'] + '/utilities/')
from metadata import *

stack = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]
first_moving_secind = int(sys.argv[4])
last_moving_secind = int(sys.argv[5])
bad_sections = map(int, sys.argv[6].split('_'))
suffix = 'thumbnail'

all_files = dict(sorted([(int(img_fn[:-4].split('_')[1]), img_fn) for img_fn in os.listdir(input_dir) if suffix in img_fn]))

# This causes race condition that all processes will have error
# Current workaround is to create this folder manually e.g. ~/csd395/CSHL_data_processed/MD603_elastix_output
# if not os.path.exists(output_dir):
# 	os.makedirs(output_dir)

n_sections = len(os.listdir(input_dir))

parameter_dir = os.path.join(os.environ['REPO_DIR'], "preprocess/parameters")

rg_param = os.path.join(parameter_dir, "Parameters_Rigid.txt")

print rg_param

rg_param_mutualinfo = os.path.join(parameter_dir, "Parameters_Rigid_MutualInfo.txt")
rg_param_noNumberOfSamples = os.path.join(parameter_dir, "Parameters_Rigid_noNumberOfSpatialSamples.txt")
rg_param_requiredRatioOfValidSamples = os.path.join(parameter_dir, "Parameters_Rigid_RequiredRatioOfValidSamples.txt")

jump_aligned_sections = pickle.load(open(os.path.join(output_dir, 'jump_aligned_sections.pkl'), 'r'))

for moving_secind in range(first_moving_secind, last_moving_secind+1):

	if moving_secind in bad_sections:
		continue

	else:
		if moving_secind in jump_aligned_sections:
			last_good_section = jump_aligned_sections[moving_secind]
		else:
			last_good_section = moving_secind - 1

		print 'moving_secind', moving_secind, 'last_good_section', last_good_section

		if last_good_section not in all_files:
			continue

		# do map moving_secind to last_good_section

		if stack == 'MD598' and moving_secind == 347:
			param = rg_param_mutualinfo
		elif stack == 'MD581' and moving_secind == 33:
			param = rg_param_noNumberOfSamples
		elif stack == 'MD595' and moving_secind == 441:
			param = rg_param_requiredRatioOfValidSamples
		elif stack == 'MD635' and moving_secind == 416:
			param = rg_param_requiredRatioOfValidSamples
		else:
			param = rg_param

		d = {'elastix_bin': os.environ['ELASTIX_BIN'],
			'rg_param': param,
			# 'output_subdir': os.path.join(output_dir, 'output%dto%d'%(moving_secind, moving_secind-1)),
			# 'fixed_fn': os.path.join(input_dir, all_files[moving_secind-1]),
			# 'moving_fn': os.path.join(input_dir, all_files[moving_secind])
			'output_subdir': os.path.join(output_dir, 'output%dto%d'%(moving_secind, last_good_section)),
			'fixed_fn': os.path.join(input_dir, all_files[last_good_section]),
			'moving_fn': os.path.join(input_dir, all_files[moving_secind])
			}

		if os.path.exists(d['output_subdir']):
			os.system('rm -r ' + d['output_subdir'])
		os.makedirs(d['output_subdir'])

		os.system('%(elastix_bin)s -f %(fixed_fn)s -m %(moving_fn)s -out %(output_subdir)s -p %(rg_param)s' % d)
