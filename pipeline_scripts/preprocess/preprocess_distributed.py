#!/usr/bin/python

import argparse
import os
import time

from preprocess_utility import *

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("stack", type=str, help="choose what stack of images to crop and resolution, ex: RS141")
	parser.add_argument("n_slides", type=int, help="number of slides")
	parser.add_argument("-j", "--jobs_per_node", type=int, help="jobs per node (default: %(default)d)", default=8)
	args = parser.parse_args()

	# t = time.time()

	# splitndpi_script = os.path.join(os.environ['GORDON_REPO_DIR'], 'pipeline_scripts', 'preprocess', 'split_ndpi.py')
	# splitndpi_args = [(args.stack, i, min(i + args.jobs_per_node - 1, args.n_slides)) for i in range(1, args.n_slides + 1, args.jobs_per_node)]

	# run_distributed(splitndpi_script, splitndpi_args)

	# print 'splitndpi', time.time() - t, 'seconds'

	# t = time.time()

	# genmask_script = os.path.join(os.environ['GORDON_REPO_DIR'], 'pipeline_scripts', 'preprocess', 'generate_mask_script.py')
	# genmask_args = [(args.stack, i, min(i + args.jobs_per_node - 1, args.n_slides)) for i in range(1, args.n_slides + 1, args.jobs_per_node)]

	# run_distributed3(genmask_script, genmask_args)

	# print 'genmask', time.time() - t, 'seconds'

	t = time.time()

	movedata_script = os.path.join(os.environ['GORDON_REPO_DIR'], 'pipeline_scripts', 'preprocess', 'preprocess_movedata_script.py')
	movedata_args = [(args.stack, i, min(i + args.jobs_per_node - 1, args.n_slides)) 
						for i in range(1, args.n_slides + 1, args.jobs_per_node)]

	run_distributed3(movedata_script, movedata_args)

	print 'movedata', time.time() - t, 'seconds'
