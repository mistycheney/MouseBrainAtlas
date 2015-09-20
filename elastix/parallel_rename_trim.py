#!/usr/bin/env python

import sys
import os
import cPickle as pickle

from preprocess_utility import run_distributed3
import time


stack = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]

DATA_DIR = '/home/yuncong/csd395/CSHL_data'
DATAPROC_DIR = '/home/yuncong/csd395/CSHL_data_processed'

filenames = os.listdir(input_dir)
tuple_sorted = sorted([(int(fn[:-4].split('_')[-1]), fn) for fn in filenames if fn.endswith('tif')])
indices_sorted, fn_sorted = zip(*tuple_sorted)

fn_correctInd_tuples = zip(fn_sorted, range(1, len(indices_sorted)+1))

n_host = 16
last_sec = len(indices_sorted)
first_sec = 1
secs_per_job = (last_sec - first_sec + 1)/float(n_host)
first_last_tuples = [(int(first_sec+i*secs_per_job), int(first_sec+(i+1)*secs_per_job-1) if i != n_host - 1 else last_sec) for i in range(n_host)]
filenames_per_node = [fn_correctInd_tuples[f-1:l] for f,l in first_last_tuples]

tmp_dir = DATAPROC_DIR + '/' + stack + '_tmp'
if not os.path.exists(tmp_dir):
	os.makedirs(tmp_dir)

for i, args in enumerate(filenames_per_node):
	with open(tmp_dir + '/' + stack + '_tmp%d'%i, 'w') as f:
		f.write('\n'.join([fn + ' ' + str(ind) for fn, ind in args]))

d = {
    # 'args': ' '.join(tmp_dir+'/tmp%d'%i for i in range(n_host)),
    # 'input_dir': input_dir,
     'all_servers_str': ','.join(['gcn-20-%d.sdsc.edu'%i for i in range(31,39)+range(41,49)]),
     'script_dir': os.path.join(os.environ['GORDON_REPO_DIR'], 'elastix'),
     'stack': stack,
     'output_dir': output_dir,
     # 'tmp_dir': tmp_dir,
     'thumbnail_dir': os.path.join(DATAPROC_DIR, stack + '_thumbnail'),
     # 'trimmed_dir': os.path.join(DATA_DIR, stack + '_trimmed'),
    }

# rename raw images and expend jp2 to tif
# kdu_expand built-in parallelism

# os.system("parallel -j 1 --filter-hosts -S %(all_servers_str)s %(script_dir)s/rename.py %(stack)s %(input_dir)s %(output_dir)s ::: %(args)s"%d)
t = time.time()
print 'renaming and expanding...',
run_distributed3('%(script_dir)s/rename.py'%d,
                [(stack, input_dir, output_dir, tmp_dir+'/tmp%d'%i, 'thumbnail') for i in range(n_host)],
                stdout=open('/tmp/log', 'ab+'))
print 'done in', time.time() - t, 'seconds'

# find background colors
t = time.time()
print 'finding background colors...',
os.system("ssh gcn-20-33 %(script_dir)s/get_background_color.py %(stack)s %(output_dir)s "%d)
print 'done in', time.time() - t, 'seconds'


with open(os.path.join(tmp_dir, stack + '_bgColor.pkl'), 'r') as f:
	bg_map = pickle.load(f)

bg_map_items = bg_map.items()
fn_bg_tuples_per_node = [bg_map_items[f-1:l] for f,l in first_last_tuples]

for i, fn_bg_tuples in enumerate(fn_bg_tuples_per_node):
    with open(os.path.join(tmp_dir, stack + '_bgColor_argsfile_%d'%i), 'w') as f:
      for secind, (r,g,b) in fn_bg_tuples:
          f.write('%s %d %d %d\n'%(stack+'_%04d'%secind, r, g, b))

# with open(os.path.join(tmp_dir, stack + '_bgColor_argsfile'), 'w') as f:
# 	for secind, (r,g,b) in bg_map.iteritems():
# 		f.write('%s %d %d %d\n'%(stack+'_%04d'%secind, r, g, b))

# if not os.path.exists(d['trimmed_dir']):
# 	os.makedirs(d['trimmed_dir'])

# trim and fill images

t = time.time()
print 'trim and fill thumbnail...',
# os.system("cat " + os.path.join(tmp_dir, stack + '_bgColor_argsfile') + """| parallel -q --colsep ' ' -j 1 --filter-hosts -S %(all_servers_str)s convert %(output_dir)s/{1}_thumbnail.tif -trim -fuzz 4%% -fill "rgb({2},{3},{4})" -opaque white -compress lzw %(output_dir)s/{1}_thumbnail.tif"""%d)
run_distributed3('%(script_dir)s/trim_fill.py'%d,
                [(stack, output_dir, output_dir, tmp_dir+'/'+stack + '_bgColor_argsfile_%d'%i, 'thumbnail') for i in range(n_host)],
                stdout=open('/tmp/log', 'ab+'))
print 'done in', time.time() - t, 'seconds'

# print 'trim and fill lossy...',
# os.system("cat " + os.path.join(tmp_dir, stack + '_bgColor_argsfile') + """| parallel -q --colsep ' ' -j 1 --filter-hosts -S %(all_servers_str)s convert %(output_dir)s/{1}_lossy.tif -trim -fuzz 4%% -fill "rgb({2},{3},{4})" -opaque white -compress lzw %(output_dir)s/{1}_lossy.tif"""%d)
# print 'done in', time.time() - t, 'seconds'

# print 'trim and fill losssless...',
# os.system("cat " + os.path.join(tmp_dir, stack + '_bgColor_argsfile') + """| parallel -q --colsep ' ' -j 1 --filter-hosts -S %(all_servers_str)s convert %(output_dir)s/{1}_lossless.tif -trim -fuzz 4%% -fill "rgb({2},{3},{4})" -opaque white -compress lzw %(output_dir)s/{1}_lossless.tif"""%d)
# print 'done in', time.time() - t, 'seconds'

# make a directory for thumbnails only, for offline inspection.
t = time.time()
print 'copy out thumbnail...',
if not os.path.exists(d['thumbnail_dir']):
	os.makedirs(d['thumbnail_dir'])
os.system('cp %(output_dir)s/*thumbnail.tif %(thumbnail_dir)s'%d)
print 'done in', time.time() - t, 'seconds'
