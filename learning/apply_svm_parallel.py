#! /usr/bin/env python

import argparse
from subprocess import check_output, call
import os
import re
import time

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from preprocess_utility import *

parser = argparse.ArgumentParser(description="Run pipeline for different instances on different servers")
parser.add_argument("task", type=str, help="task to perform (svm)")
parser.add_argument("stack", help="stack name, e.g. MD593")
parser.add_argument("-b", type=int, help="beginning slide (default: %(default)s)", default=0)
parser.add_argument("-e", type=int, help="ending slide (default: %(default)s)", default=-1)

args = parser.parse_args()

t = time.time()

exclude_nodes = []

if args.task == 'svm':

	t = time.time()
	sys.stderr.write('running svm classifier ...')

	run_distributed3(command='%(script_path)s %(stack)s %%(f)d %%(l)d'%\
	                            {'script_path': '/home/yuncong/Brain/learning/apply_svm.py',
	                            'stack': args.stack}, 
	                first_sec=args.b,
	                last_sec=args.e,
	                exclude_nodes=[35],
	                take_one_section=False)

	sys.stderr.write('done in %f seconds\n' % (time.time() - t))

print args.task, time.time() - t, 'seconds'
