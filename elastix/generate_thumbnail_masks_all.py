#!/usr/bin/env python

import sys
import os
import cPickle as pickle

from preprocess_utility import *
import time

stack = sys.argv[1]
first_sec, last_sec = map(int, sys.argv[2:4])
x,y,w,h = map(int, sys.argv[4:8])

# hostids = detect_responsive_nodes(exclude_nodes=[33,42])
hostids = detect_responsive_nodes()

print hostids

n_hosts = len(hostids)

DATA_DIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_data'
DATAPROC_DIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_processed'

script_dir = os.path.join(os.environ['GORDON_REPO_DIR'], 'elastix')

d = {'input_dir': os.path.join(DATAPROC_DIR, stack+'_thumbnail_aligned_cropped'),
	'mask_dir': os.path.join(DATAPROC_DIR, stack+'_thumbnail_aligned_cropped_mask'),
	 'masked_img_dir': os.path.join(DATAPROC_DIR, stack+'_thumbnail_aligned_cropped_masked')
	 }

t = time.time()
sys.stderr.write('generating mask ...')
run_distributed3(script_dir + '/generate_thumbnail_mask.py'%d, 
				[(stack, d['input_dir'], d['mask_dir'], d['masked_img_dir'], f, l) for f, l in first_last_tuples_distribute_over(first_sec, last_sec, n_hosts)],
				stdout=open('/tmp/log', 'ab+'))
sys.stderr.write('done in %f seconds\n' % (time.time() - t))
