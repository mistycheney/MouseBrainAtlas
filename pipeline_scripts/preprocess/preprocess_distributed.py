#!/usr/bin/python

import argparse
import os
import time

from preprocess_utility import *

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("task", type=str, help="task to perform (must be one of splitndpi, genmask, movedata)")
	parser.add_argument("stack", type=str, help="the stack name")
	parser.add_argument("n_slides", type=int, help="number of slides")
	parser.add_argument("-j", "--slides_per_node", type=int, help="number of slides each node processes (default: %(default)d)", default=4)
	args = parser.parse_args()

	t = time.time()

	if task == 'splitndpi':

		splitndpi_script = os.path.join(os.environ['GORDON_REPO_DIR'], 'pipeline_scripts', 'preprocess', 'split_ndpi.py')
		splitndpi_args = [(args.stack, i, min(i + args.jobs_per_node - 1, args.n_slides)) for i in range(1, args.n_slides + 1, args.jobs_per_node)]

		run_distributed(splitndpi_script, splitndpi_args)

	elif task == 'genmask':

		genmask_script = os.path.join(os.environ['GORDON_REPO_DIR'], 'pipeline_scripts', 'preprocess', 'generate_mask_script.py')
		genmask_args = [(args.stack, i, min(i + args.jobs_per_node - 1, args.n_slides)) for i in range(1, args.n_slides + 1, args.jobs_per_node)]

		run_distributed3(genmask_script, genmask_args)

	elif task == 'movedata':

		movedata_script = os.path.join(os.environ['GORDON_REPO_DIR'], 'pipeline_scripts', 'preprocess', 'preprocess_movedata_script.py')
		movedata_args = [(args.stack, i, min(i + args.jobs_per_node - 1, args.n_slides)) 
							for i in range(1, args.n_slides + 1, args.jobs_per_node)]

		run_distributed3(movedata_script, movedata_args)

	else:
		raise Exception('task must be one of splitndpi, genmask, movedata.')

	print task, time.time() - t, 'seconds'