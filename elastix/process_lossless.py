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
first_last_tuples = first_last_tuples_distribute_over(first_sec, last_sec, n_hosts)


DATA_DIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_data'
DATAPROC_DIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_processed'

tmp_dir = DATAPROC_DIR + '/' + 'tmp'

script_dir = os.path.join(os.environ['GORDON_REPO_DIR'], 'elastix')

t = time.time()
print 'renaming and expanding lossless...',
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