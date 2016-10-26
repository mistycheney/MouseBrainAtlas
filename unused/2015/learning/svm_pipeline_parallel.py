#! /usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Run pipeline for different instances on different servers")
parser.add_argument("task", type=str, help="task to perform (svm, interpolate, visualize)")
parser.add_argument("stack", help="stack name, e.g. MD593")
parser.add_argument("-b", type=int, help="beginning slide (default: first_detect_sec)", default=0)
parser.add_argument("-e", type=int, help="ending slide (default: last_detect_sec)", default=-1)

args = parser.parse_args()

from subprocess import check_output, call
import os
import time
import sys

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from preprocess_utility import *
from utilities2015 import *

first_detect_sec, last_detect_sec = detect_bbox_range_lookup[args.stack]

args.b = first_detect_sec if args.b == 0 else args.b
args.e = last_detect_sec if args.e == -1 else args.e

t = time.time()

exclude_nodes = [33] # 33 is yuncong's ipython notebook server

if args.task == 'svm':

	t = time.time()
	sys.stderr.write('running svm classifier ...')

	run_distributed4(command='%(script_path)s %(stack)s %%(first_sec)d %%(last_sec)d' % \
                {'script_path': os.path.join(os.environ['REPO_DIR'], 'learning') + '/svm_predict.py',
                'stack': stack},
                kwargs_list=dict(sections=range(first_sec, last_sec+1)),
                exclude_nodes=exclude_nodes,
                argument_type='partition')

	sys.stderr.write('done in %f seconds\n' % (time.time() - t)) # ~ 1000 seconds

elif args.task == 'interpolate':

	t = time.time()
	sys.stderr.write('interpolating scoremaps ...')

	run_distributed4(command='%(script_path)s %(stack)s %%(first_sec)d %%(last_sec)d' % \
                {'script_path': os.path.join(os.environ['REPO_DIR'], 'learning') + '/interpolate_scoremaps_v2.py',
                'stack': stack},
                kwargs_list=dict(sections=range(first_sec, last_sec+1)),
                exclude_nodes=exclude_nodes,
                argument_type='partition')

	sys.stderr.write('done in %f seconds\n' % (time.time() - t)) # ~240 seconds

elif args.task == 'visualize':

	add_annotation = False

	t = time.time()
	sys.stderr.write('visualize scoremaps ...')

	run_distributed4(command='%(script_path)s %(stack)s -b %%(first_sec)d -e %%(last_sec)d %(add_annotation)s' % \
                {'script_path': os.path.join(os.environ['REPO_DIR'], 'learning') + '/visualize_scoremaps_v2.py',
                'stack': stack,
                'add_annotation': '-a' if add_annotation else ''},
                kwargs_list=dict(sections=range(first_sec, last_sec+1)),
                exclude_nodes=exclude_nodes,
                argument_type='partition')

	sys.stderr.write('done in %f seconds\n' % (time.time() - t)) # ~ 40 seconds


print args.task, time.time() - t, 'seconds'
