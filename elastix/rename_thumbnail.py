#!/usr/bin/env python

import sys
import os
import cPickle as pickle

from preprocess_utility import *
import time
import argparse

# def first_last_tuples_distribute_over(first_sec, last_sec, n_host):
# 	secs_per_job = (last_sec - first_sec + 1)/float(n_host)
# 	first_last_tuples = [(int(first_sec+i*secs_per_job), int(first_sec+(i+1)*secs_per_job-1) if i != n_host - 1 else last_sec) for i in range(n_host)]
# 	return first_last_tuples

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Rename thumbnail images according to our naming format with consecutive section numbers')

DATA_DIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_data'

DATAPROC_DIR = os.environ['DATA_DIR']
# DATAPROC_DIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_processed'

stack = sys.argv[1]
input_dir = os.path.join(DATA_DIR, stack)
output_dir = os.path.join(DATAPROC_DIR, stack+'_thumbnail_renamed')

exclude_nodes = [33,41]
hostids = detect_responsive_nodes(exclude_nodes=exclude_nodes)
n_host = len(hostids)

filenames = os.listdir(input_dir)
tuple_sorted = sorted(sorted([fn for fn in filenames if fn.endswith('tif')], key=lambda fn: len(fn.split('-')[1])), key=lambda fn: int(fn[:-4].split('_')[-1]))

indices_sorted, fn_sorted = zip(*enumerate(tuple_sorted))
fn_correctInd_tuples = zip(fn_sorted, range(1, len(indices_sorted)+1))
filenames_per_node = [fn_correctInd_tuples[f-1:l] for f,l in first_last_tuples_distribute_over(1, len(indices_sorted), n_host)]

tmp_dir = DATAPROC_DIR + '/' + 'tmp'
if not os.path.exists(tmp_dir):
	os.makedirs(tmp_dir)

for i, args in enumerate(filenames_per_node):
	with open(tmp_dir + '/' + stack + '_filename_map_%d'%i, 'w') as f:
		f.write('\n'.join([fn + ' ' + str(ind) for fn, ind in args]))

t = time.time()
print 'renaming and expanding thumbnails...',
run_distributed3(os.path.join(os.environ['GORDON_REPO_DIR'], 'elastix') + '/rename.py', 1, len(tuple_sorted),
                [(stack, input_dir, output_dir, tmp_dir+'/'+stack+'_filename_map_%d'%i, 'thumbnail') for i in range(n_host)],
                stdout=open('/tmp/log', 'ab+'))
print 'done in', time.time() - t, 'seconds'