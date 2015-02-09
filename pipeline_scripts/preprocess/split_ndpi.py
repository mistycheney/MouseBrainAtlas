#!/usr/bin/python

import os
import sys
import shutil
import glob
import argparse
import time
from joblib import Parallel, delayed

from preprocess_utility import *

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("stack", type=str, help="choose what stack of images to crop and resolution, ex: RS141")
	parser.add_argument("start_slide", type=int, help="first slide")
	parser.add_argument("end_slide", type=int, help="last slide")
	args = parser.parse_args()
	
	stack = args.stack

	slide_dir_stack = create_if_not_exists(os.path.join(os.environ['GORDON_SLIDEDATA_DIR'], stack))	# DavidData2015slides/CC35

	ndpi_stack_dir = os.path.join(os.environ['GORDON_NDPI_DIR'], stack)
	os.chdir(ndpi_stack_dir)

	def split_one_slide(ndpi_file):

		execute_command(os.environ['GORDON_NDPISPLIT_PROGRAM'] + ' ' + os.path.join(ndpi_stack_dir, ndpi_file))

		for level in ['macro', 'x0.078125', 'x0.3125', 'x1.25', 'x5', 'x20']:
			slide_dir_resol = create_if_not_exists(os.path.join(slide_dir_stack, level))			# DavidData2015slides/CC35/x0.3125
			for f in glob.glob('*_%s_z0.tif'%level):
				try:
					shutil.move(f, slide_dir_resol)
				except Exception as e:
					print e
					pass

		for level in ['macro', 'map']:
			map_dir = create_if_not_exists(os.path.join(slide_dir_stack, level))			# DavidData2015slides/CC35/map
			for f in glob.glob('*'+level+'*'):
				try:
					shutil.move(f, map_dir)
				except Exception as e:
					print e
					pass

	ndpi_files = [os.path.join(ndpi_stack_dir, '%s_%02d.ndpi'%(args.stack, slide_id)) for slide_id in range(args.start_slide, args.end_slide+1)]

	Parallel(n_jobs=16)(delayed(split_one_slide)(ndpi_file) for ndpi_file in ndpi_files)
