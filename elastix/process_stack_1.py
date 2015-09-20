#! /usr/bin/env python

import os
import sys
from preprocess_utility import run_distributed3	
import time

DATA_DIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_data'
DATAPROC_DIR = '/home/yuncong/csd395/CSHL_data_processed'

stack = sys.argv[1]
input_dir = sys.argv[2]
first_sec = int(sys.argv[3])
last_sec = int(sys.argv[4])
# x,y,w,h = map(int, sys.argv[5:9])

d = {'all_sections_str': ' '.join(map(str, range(first_sec, last_sec+1))),
     # 'all_sections_str2': ' '.join(map(str, range(first_sec+1, last_sec+1))),
     # 'all_servers_str': ','.join(['gcn-20-%d.sdsc.edu'%i for i in range(31,39)+range(41,49)]),
     'script_dir': os.path.join(os.environ['GORDON_REPO_DIR'], 'elastix'),
     'stack': stack,
     'first_sec': first_sec,
     'last_sec': last_sec,

     # 'trimmed_dir': os.path.join(DATA_DIR, stack+'_trimmed'),
     'input_dir': input_dir,
	 'padded_dir': os.path.join(DATAPROC_DIR, stack+'_thumbnail_padded'),
	 'elastix_output_dir': os.path.join(DATAPROC_DIR, stack+'_elastix_output'),
	 'warped_dir': os.path.join(DATAPROC_DIR, stack+'_thumbnail_warped'),
	 'suffix': 'thumbnail',
    }

n_host = 16
secs_per_job = (last_sec - first_sec + 1)/float(n_host)
first_last_tuples = [(int(first_sec+i*secs_per_job), int(first_sec+(i+1)*secs_per_job-1) if i != n_host - 1 else last_sec) for i in range(n_host)]

# with open(os.path.join(DATAPROC_DIR, 'tmp', stack + '_padOffsets.pkl'), 'w') as f:
#     offsets = pickle.load(f)


# joblib parallelism
# t = time.time()
# print 'padding...',
# os.system("ssh gcn-20-33.sdsc.edu %(script_dir)s/pad_IM.py %(stack)s %(input_dir)s %(padded_dir)s %(first_sec)d %(last_sec)d %(suffix)s"%d)
# print 'done in', time.time() - t, 'seconds'

# elastix has built-in parallelism
t = time.time()
print 'aligning...',
run_distributed3('%(script_dir)s/align_consecutive.py'%d, 
				[(stack, d['input_dir'], d['elastix_output_dir'], f, l) for f, l in first_last_tuples],
				stdout=open('/tmp/log', 'ab+'))
print 'done in', time.time() - t, 'seconds'

# # # # # GNU Parallel initialization very slow
# # # # # os.system("parallel -j 1 --filter-hosts -S %(all_servers_str)s %(script_dir)s/elastix_consecutive.py %(stack)s ::: %(all_sections_str2)s"%d)
# # # # # os.system("parallel -j 1 -N2 --filter-hosts -S %(all_servers_str)s %(script_dir)s/elastix_consecutive.py %(stack)s {1} {2} ::: "%d + ' '.join(map(lambda x: '%d %d'%x, first_last)))

# # no parallelism
t = time.time()
print 'composing transform...',
with open(os.path.join(DATAPROC_DIR, 'tmp', stack+'_largest_sec'), 'r') as f:
    largest_sec = int(f.readline())
print 'largest section is ', largest_sec
os.system("ssh gcn-20-33.sdsc.edu %(script_dir)s/compose_transform_thumbnail.py %(stack)s %(elastix_output_dir)s %(first_sec)d %(last_sec)d"%d + ' ' + str(largest_sec))
# os.system("ssh gcn-20-33.sdsc.edu %(script_dir)s/compose_transform_thumbnail.py %(stack)s %(elastix_output_dir)s %(first_sec)d %(last_sec)d"%d)
print 'done in', time.time() - t, 'seconds'

# no parallelism
t = time.time()
print 'warping...',
run_distributed3('%(script_dir)s/warp_crop_IM.py'%d, 
				[(stack, d['input_dir'], d['warped_dir'], f, l, d['suffix'], 0, 0, 2000, 1500) for f, l in first_last_tuples],
				stdout=open('/tmp/log', 'ab+'))
print 'done in', time.time() - t, 'seconds'


# t = time.time()
# print 'warping and cropping...',
# run_distributed3('%(script_dir)s/warp_crop_IM.py'%d, 
#                 [(stack, d['input_dir'], d['warped_dir'], f, l, d['suffix'], x, y, w, h) for f, l in first_last_tuples],
#                 # [(stack, d['input_dir'], d['warped_dir'], f, l, d['suffix'], 0, 0, 2000, 1500) for f, l in first_last_tuples],
#                 stdout=open('/tmp/log', 'ab+'))
# print 'done in', time.time() - t, 'seconds'


# GNU Parallel initialization very slow
# os.system("parallel -j 1 --filter-hosts -S %(all_servers_str)s %(script_dir)s/warp_thumbnails.py %(stack)s ::: %(all_sections_str)s"%d)
