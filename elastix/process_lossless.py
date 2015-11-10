#!/usr/bin/env python

import sys
import os
import cPickle as pickle

from preprocess_utility import *
import time

stack = sys.argv[1]
first_sec, last_sec = map(int, sys.argv[2:4])
x,y,w,h = map(int, sys.argv[4:8])

hostids = detect_responsive_nodes()
n_hosts = len(hostids)
# first_last_tuples = first_last_tuples_distribute_over(first_sec, last_sec, n_hosts)

DATA_DIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_data'
DATAPROC_DIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_processed'

tmp_dir = DATAPROC_DIR + '/' + 'tmp'

script_dir = os.path.join(os.environ['GORDON_REPO_DIR'], 'elastix')

# hostids = detect_responsive_nodes()
# n_host = len(hostids)
input_dir = os.path.join(DATA_DIR, stack)
filenames = os.listdir(input_dir)
tuple_sorted = sorted([(int(fn[:-4].split('_')[-1]), fn) for fn in filenames if fn.endswith('tif')])
indices_sorted, fn_sorted = zip(*tuple_sorted)
fn_correctInd_tuples = zip(fn_sorted, range(1, len(indices_sorted)+1))
filenames_per_node = [fn_correctInd_tuples[f-1:l] for f,l in first_last_tuples_distribute_over(1, len(indices_sorted), n_hosts)]

tmp_dir = DATAPROC_DIR + '/' + 'tmp'
if not os.path.exists(tmp_dir):
	os.makedirs(tmp_dir)

os.system('rm '+tmp_dir + '/'+stack+'_filename_map_*')

for i, args in enumerate(filenames_per_node):
	with open(tmp_dir + '/' + stack + '_filename_map_%d'%i, 'w') as f:
		f.write('\n'.join([fn + ' ' + str(ind) for fn, ind in args]))

t = time.time()
print 'renaming and decompressing lossless...',
run_distributed3(script_dir + '/rename.py',
                [(stack, os.path.join(DATA_DIR, stack), os.path.join(DATAPROC_DIR, stack + '_lossless_renamed'), tmp_dir+'/'+stack+'_filename_map_%d'%i, 'lossless') 
                    for i in range(n_hosts)],
                stdout=open('/tmp/log', 'ab+'))
print 'done in', time.time() - t, 'seconds'

t = time.time()
print 'warping and cropping...',
run_distributed3(script_dir + '/warp_crop_IM.py', 
                [(stack, os.path.join(DATAPROC_DIR, stack + '_lossless_renamed'), os.path.join(DATAPROC_DIR, stack + '_lossless_cropped'), f, l, 'lossless', x, y, w, h) 
                    for f, l in first_last_tuples_distribute_over(first_sec, last_sec, n_hosts)],
                stdout=open('/tmp/log', 'ab+'))
print 'done in', time.time() - t, 'seconds'