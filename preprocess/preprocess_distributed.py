#!/usr/bin/python

import argparse
import os
import time
import re

from subprocess import check_output

from preprocess_utility import *

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("task", type=str, help="task to perform (must be one of splitndpi, genmask, movedata)")
	parser.add_argument("stack", type=str, help="the stack name")
	parser.add_argument("n_slides", type=int, help="number of slides, use 0 for all slides")
	parser.add_argument("-j", "--slides_per_node", type=int, help="number of slides each node processes (default: %(default)d)", default=4)
	# parser.add_argument("-i", "--initsec", type=int, help="index of the first section on the first slide (default: %(default)d)", default=0)
	parser.add_argument("-s", "--stain", type=str, help="must use if task == genmask. stain is either fluorescent (fl) or nissl")
	
	args = parser.parse_args()

	task = args.task
	stack = args.stack
	n_slides = args.n_slides
	slides_per_node = args.slides_per_node

	t = time.time()

	s = check_output("ssh gordon.sdsc.edu ls %s" % os.path.join(os.environ['GORDON_NDPI_DIR'], stack), shell=True)
	print s
	slide_indices = [int(re.split("_|-", f[:-5])[1]) for f in s.split('\n') if len(f) > 0]

	if n_slides == 0:
		n_slides = max(slide_indices)
		print 'last slide index', n_slides

	if task == 'splitndpi':

		splitndpi_script = os.path.join(os.environ['GORDON_REPO_DIR'], 'pipeline_scripts', 'preprocess', 'split_ndpi_script.py')

		splitndpi_args = [(stack, i, min(i + slides_per_node - 1, n_slides)) 
							for i in range(1, n_slides + 1, slides_per_node)]

		run_distributed3(splitndpi_script, splitndpi_args)

	elif task == 'genmask':

		if 'stain' in args:
			stain = args.stain
		else:
			raise Exception('Please specify stain. It is either fluorescent or nissl.')

		if stain == 'fl':
			stain = 'fluorescent'

		genmask_script = os.path.join(os.environ['GORDON_REPO_DIR'], 'pipeline_scripts', 'preprocess', 'generate_mask_script.py')
		genmask_args = [(stack, i, min(i + slides_per_node - 1, n_slides), stain) 
						for i in range(1, n_slides + 1, slides_per_node)]

		run_distributed3(genmask_script, genmask_args)

	elif task == 'movedata':

		# initsec = args.initsec

		movedata_script = os.path.join(os.environ['GORDON_REPO_DIR'], 'pipeline_scripts', 'preprocess', 'preprocess_movedata_script.py')
		# movedata_args = [(stack, i, min(i + slides_per_node - 1, n_slides), initsec) 
		# 					for i in range(1, n_slides + 1, slides_per_node)]
		movedata_args = [(stack, i, min(i + slides_per_node - 1, n_slides)) 
							for i in range(1, n_slides + 1, slides_per_node)]

		run_distributed3(movedata_script, movedata_args)

	else:
		raise Exception('task must be one of splitndpi, genmask, movedata.')

	print task, time.time() - t, 'seconds'