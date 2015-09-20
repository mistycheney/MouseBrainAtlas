#! /usr/bin/env python

import sys
import os
import numpy as np
import cPickle as pickle

from joblib import Parallel, delayed

stack = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]
first_secind = int(sys.argv[4])
last_secind = int(sys.argv[5])
suffix = sys.argv[6]
x = sys.argv[7]
y = sys.argv[8]
w = sys.argv[9]
h = sys.argv[10]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

DATAPROC_DIR = '/home/yuncong/csd395/CSHL_data_processed'

all_files = dict(sorted([(int(img_fn[:-4].split('_')[1]), img_fn) for img_fn in os.listdir(input_dir) if suffix in img_fn]))

if suffix == 'thumbnail':
    scale_factor = 1
elif suffix == 'lossy':
    scale_factor = 16
elif suffix == 'lossless':
    scale_factor = 32
    
for secind in range(first_secind, last_secind+1):

	d = {
	'x': x * scale_factor,
	'y': y * scale_factor,
	'w': w * scale_factor,
	'h': h * scale_factor,
	'input_fn': os.path.join(input_dir, all_files[secind]),
	'output_fn': os.path.join(output_dir, all_files[secind][:-4] + '_cropped.tif')
	}

	os.system('convert %(input_fn)s -crop %(w)dx%(h)d+%(x)d+%(y)d %(output_fn)s'%d)





