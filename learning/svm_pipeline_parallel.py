#! /usr/bin/env python

import argparse
from subprocess import check_output, call
import os
import time
import sys

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from preprocess_utility import *
from utilities2015 import *

parser = argparse.ArgumentParser(description="Run pipeline for different instances on different servers")
parser.add_argument("task", type=str, help="task to perform (svm, interpolate, visualize)")
parser.add_argument("stack", help="stack name, e.g. MD593")
parser.add_argument("-b", type=int, help="beginning slide (default: %(default)s)", default=0)
parser.add_argument("-e", type=int, help="ending slide (default: %(default)s)", default=-1)

args = parser.parse_args()

first_detect_sec, last_detect_sec = detect_bbox_range_lookup[args.stack]

args.b = first_detect_sec if args.b == 0 else args.b
args.e = last_detect_sec if args.e == -1 else args.e

t = time.time()

exclude_nodes = [33]

if args.task == 'svm':

	t = time.time()
	sys.stderr.write('running svm classifier ...')

	run_distributed3(command='%(script_path)s %(stack)s %%(f)d %%(l)d'%\
	                            {'script_path': '/home/yuncong/Brain/learning/apply_svm.py',
	                            'stack': args.stack}, 
	                first_sec=args.b,
	                last_sec=args.e,
	                exclude_nodes=exclude_nodes,
	                take_one_section=False)

	sys.stderr.write('done in %f seconds\n' % (time.time() - t))
    
elif args.task == 'interpolate':

	t = time.time()
	sys.stderr.write('interpolating scoremaps ...')

	run_distributed3(command='%(script_path)s %(stack)s %%(f)d %%(l)d'%\
	                            {'script_path': '/home/yuncong/Brain/learning/interpolate_scoremaps.py',
	                            'stack': args.stack}, 
	                first_sec=args.b,
	                last_sec=args.e,
	                exclude_nodes=exclude_nodes,
	                take_one_section=False)

	sys.stderr.write('done in %f seconds\n' % (time.time() - t))

elif args.task == 'visualize':

	t = time.time()
	sys.stderr.write('visualize scoremaps ...')

	run_distributed3(command='%(script_path)s %(stack)s %%(f)d %%(l)d'%\
	                            {'script_path': '/home/yuncong/Brain/learning/visualize_scoremaps.py',
	                            'stack': args.stack}, 
	                first_sec=args.b,
	                last_sec=args.e,
	                exclude_nodes=exclude_nodes,
	                take_one_section=False)

	sys.stderr.write('done in %f seconds\n' % (time.time() - t)) #~40 seconds

    
print args.task, time.time() - t, 'seconds'
