#! /usr/bin/env python

import os
import sys
from preprocess_utility import *
from subprocess import check_output
import time
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Process after identifying the first and last sections in the stack that contain brainstem: 1) align thumbnails')

# DATA_DIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_data'
# DATAPROC_DIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_processed'
DATAPROC_DIR = os.environ['DATA_DIR']

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("first_sec", type=int, help="first section")
parser.add_argument("last_sec", type=int, help="last section")
args = parser.parse_args()

stack = args.stack_name
first_sec = args.first_sec
last_sec = args.last_sec

d = {
     'script_dir': os.path.join(os.environ['GORDON_REPO_DIR'], 'elastix'),
     'stack': stack,
     'first_sec': first_sec,
     'last_sec': last_sec,

     'input_dir': os.path.join(DATAPROC_DIR, stack+'_thumbnail_renamed'),
	 'elastix_output_dir': os.path.join(DATAPROC_DIR, stack+'_elastix_output'),
	 'aligned_dir': os.path.join(DATAPROC_DIR, stack+'_thumbnail_aligned'),
	 'suffix': 'thumbnail'
    }


hostids = detect_responsive_nodes()
n_hosts = len(hostids)
first_last_tuples = first_last_tuples_distribute_over(first_sec, last_sec, n_hosts)

# elastix has built-in parallelism
t = time.time()
print 'aligning...',
# run_distributed3('%(script_dir)s/align_consecutive.py'%d, 
# 				[(stack, d['input_dir'], d['elastix_output_dir'], f, l) for f, l in first_last_tuples],
# 				stdout=open('/tmp/log', 'ab+'))

run_distributed3('%(script_dir)s/align_consecutive.py %(stack)s %(input_dir)s %(elastix_output_dir)s %%(f)d %%(l)d'%d, 
                first_sec=first_sec,
                last_sec=last_sec,
                stdout=open('/tmp/log', 'ab+'),
                take_one_section=False)

print 'done in', time.time() - t, 'seconds'

from joblib import Parallel, delayed

def identify_shape(img_fn):
    return map(int, check_output("identify -format %%Wx%%H %s" % os.path.join(d['input_dir'], img_fn), shell=True).split('x'))

all_files = dict(sorted([(int(img_fn[:-4].split('_')[1]), img_fn) for img_fn in os.listdir(d['input_dir']) if d['suffix'] in img_fn]))
all_files = dict([(i, all_files[i]) for i in range(first_sec, last_sec+1)])
shapes = Parallel(n_jobs=16)(delayed(identify_shape)(img_fn) for img_fn in all_files.values())
img_shapes_map = dict(zip(all_files.keys(), shapes))
img_shapes_arr = np.array(img_shapes_map.values())
largest_sec = img_shapes_map.keys()[np.argmax(img_shapes_arr[:,0] * img_shapes_arr[:,1])]
print 'largest section is ', largest_sec

# no parallelism
t = time.time()
print 'composing transform...',
os.system("ssh gcn-20-33.sdsc.edu %(script_dir)s/compose_transform_thumbnail.py %(stack)s %(elastix_output_dir)s %(first_sec)d %(last_sec)d"%d + ' ' + str(largest_sec))
# os.system("ssh gcn-20-33.sdsc.edu %(script_dir)s/compose_transform_thumbnail.py %(stack)s %(elastix_output_dir)s %(first_sec)d %(last_sec)d"%d)
print 'done in', time.time() - t, 'seconds'

# no parallelism
t = time.time()
print 'warping...',
# run_distributed3('%(script_dir)s/warp_crop_IM.py'%d, 
# 				[(stack, d['input_dir'], d['aligned_dir'], f, l, d['suffix'], 0) for f, l in first_last_tuples],
# 				stdout=open('/tmp/log', 'ab+'))

run_distributed3('%(script_dir)s/warp_crop_IM.py %(stack)s %(input_dir)s %(aligned_dir)s %%(f)d %%(l)d %(suffix)s 0'%d, 
                first_sec=first_sec,
                last_sec=last_sec,
                take_one_section=False,
                stdout=open('/tmp/log', 'ab+'))
print 'done in', time.time() - t, 'seconds'