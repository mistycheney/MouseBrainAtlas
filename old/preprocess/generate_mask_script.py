#!/usr/bin/python

import os
import argparse
from joblib import Parallel, delayed
import numpy as np

from preprocess_utility import *
from generate_mask_simple import *
# from generate_mask import *

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("stack", type=str, help="choose what stack of images to crop and resolution, ex: RS141")
	parser.add_argument("start_slide", type=int, help="first slide")
	parser.add_argument("end_slide", type=int, help="last slide")
	parser.add_argument("stain", type=str, help="nissl or fluoresent (default: %(default)s)", default='nissl')

	args = parser.parse_args()

	stack = args.stack
	stain = args.stain

	resol = 'x0.3125'

	data_dir_stack = create_if_not_exists(os.path.join(os.environ['GORDON_SECTIONDATA_DIR'], stack))	# DavidData2015sections/CC35
	
	autogen_masked_img_dir = create_if_not_exists(os.path.join(data_dir_stack, 'autogen_maskedimg_'+resol))		# DavidData2015sections/CC35/autogen_masked_x0.3125
	autogen_bbox_dir = create_if_not_exists(os.path.join(data_dir_stack, 'autogen_bbox_'+resol))	# DavidData2015sections/CC35/autogen_bbox

	slide_dir_stack = create_if_not_exists(os.path.join(os.environ['GORDON_SLIDEDATA_DIR'], stack))	# DavidData2015slides/CC35
	slide_dir = os.path.join(slide_dir_stack, resol)		# DavidData2015slides/CC35/x0.3125

	options = [{'stain':stain, 'in_dir':slide_dir, 'slide_ind':i, 'stack':stack, 
					'out_dir':autogen_masked_img_dir, 'num_section_per_slide': 5, 'resol':resol} 
					for i in range(args.start_slide, args.end_slide+1)]

	bboxes = Parallel(n_jobs=16)(delayed(gen_mask)(**kwargs) for kwargs in options)

	for bboxes_slide, slide_ind in zip(bboxes, range(args.start_slide, args.end_slide+1)):
		if bboxes_slide is None: continue
		bbox_filepath = os.path.join(autogen_bbox_dir, 'bbox_%s_%s_%02d.txt'%(stack, resol, slide_ind))		# DavidData2015sections/CC35/autogen_bbox/bbox_CC35_03.txt
		np.savetxt(bbox_filepath, bboxes_slide, fmt='%d')