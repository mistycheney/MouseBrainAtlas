#!/usr/bin/python

from subprocess import call

def execute_command(cmd):
	try:
	    retcode = call(cmd, shell=True)
	    if retcode < 0:
	        print >>sys.stderr, "Child was terminated by signal", -retcode
	    else:
	        print >>sys.stderr, "Child returned", retcode
	except OSError as e:
	    print >>sys.stderr, "Execution failed:", e
	    raise e

import os
import sys
import shutil
import glob
import argparse
import json

import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray, rgb2hsv
from skimage.io import imread, imsave
from skimage.morphology import remove_small_objects
from skimage.measure import regionprops, label
from skimage.restoration import denoise_bilateral
from skimage.transform import rescale

from operator import itemgetter

from preprocess_utility import *

from PIL import Image

class NumpyAwareJSONEncoder(json.JSONEncoder):
	def default(self, obj):
	    if isinstance(obj, np.ndarray) and obj.ndim == 1:
	        return obj.tolist()
	    return json.JSONEncoder.default(self, obj)

def summarize_bbox_and_masks(bbox_dir, mask_dir):
	os.chdir(bbox_dir)

	section_counter = -1
	
	# summary_dict = {}

	all_bboxfiles = sorted([bbox_file for bbox_file in os.listdir('.') if bbox_file.endswith('txt')])

	for bbox_file in all_bboxfiles:

		# if section_counter > 3: break
		
		_, stack, _, slide_str = bbox_file[:-4].split('_')
		slide_ind = int(slide_str)

		bboxes_x03125 = np.loadtxt(bbox_file).astype(np.int)

		# slide_imagepath = os.path.join(os.environ['GORDON_SLIDEDATA_DIR'], stack, '', "CC35_%02d_x0.3125_z0.tif" % slide_ind)		# DavidData2015slides/CC35/x0.3125/CC35_45_x0.3125_z0.tif
		# slide_image = np.array(Image.open(slide_imagepath).convert('RGB'))
		# slide_width, slide_height = slide_image.shape[:2]

		# section_imagepath = os.path.join(os.environ['GORDON_SECTIONDATA_DIR'], stack, resol, 'autogen_maskedimg_x0.3125', '%s_%s_%04d.tif' % (stack, resol, section_counter))		# DavidData2015slides/CC35/x0.3125/CC35_45_x0.3125_z0.tif
		# mask_path = section_imagepath # TO CHANGE!!

		curr_resol_slide_image = {}

		for resol in ['x0.3125', 'x1.25', 'x5', 'x20']:
			curr_resol_slide_path = os.path.join(os.environ['GORDON_SLIDEDATA_DIR'], stack, resol, "%s_%02d_%s_z0.tif" % (stack, slide_ind, resol))
			curr_resol_slide_image[resol] = np.array(Image.open(curr_resol_slide_path).convert('RGB'))

		for section_ind, bbox in enumerate(bboxes_x03125):

			section_counter += 1

			if (bbox < 0).all(): continue

			masked_sectionimg_path = os.path.join(os.environ['GORDON_SECTIONDATA_DIR'], stack, 'autogen_maskedimg_x0.3125', "%s_x0.3125_%02d_%d.tif" % (stack, slide_ind, section_ind))
			# alpha = 255 * imread(masked_sectionimg_path)[..., -1].astype(np.uint8)

			alpha = imread(masked_sectionimg_path)[..., -1]

			# minr, minc, maxr, maxc = bbox
			# section_dict = {}

			for l, resol in enumerate(['x0.3125', 'x1.25', 'x5', 'x20']):
			# for l, resol in enumerate(['x0.3125']):
				if resol != 'x20': continue

				scaling = 4**l

				curr_resol_bbox = bbox * scaling

				# section_dict[resol] = {'slide_path': curr_resol_slide_path,
				# 				# 'bbox': bboxes_x03125.astype(np.float) / np.r_[slide_image.shape[:2], slide_image.shape[:2]] * np.r_[curr_resol_slide_shape, curr_resol_slide_shape],
				# 				'bbox': curr_resol_bbox,
				# 				# 'mask_path': os.path.join(os.environ['GORDON_SECTIONDATA_DIR'], stack, resol, 'masks', '%s_%s_%04d_mask.png' % (stack, resol, section_counter)),
				# 				'sectionimg_path': sectionimg_path
				# 				}

				minr, minc, maxr, maxc = curr_resol_bbox

				# print bbox_file, slide_ind, minr, minc, maxr, maxc

				cur_resol_sectionimg = curr_resol_slide_image[resol][minr:maxr+scaling, minc:maxc+scaling]
				cur_resol_alpha = 255 * (rescale(alpha, scaling, order=0) > 0).astype(np.uint8)
				# print bbox, curr_resol_bbox, cur_resol_sectionimg.shape, alpha.shape, cur_resol_alpha.shape

				cur_resol_masked_sectionimg = np.dstack([cur_resol_sectionimg, cur_resol_alpha])

				# sectionimg_path = os.path.join(os.environ['GORDON_SECTIONDATA_DIR'], stack, resol, 'images', '%s_%s_%04d.tif' % (stack, resol, section_counter))
				cur_resol_masked_sectionimg_path = os.path.join(os.environ['GORDON_SECTIONDATA_DIR'], stack, resol, '%s_%s_%04d.tif' % (stack, resol, section_counter))
				imsave(cur_resol_masked_sectionimg_path, cur_resol_masked_sectionimg)

				# execute_command('convert -resize 400\% %s %s' % (mask_path, section_dict[resol]['mask_path']))

			# summary_dict[section_counter] = section_dict

	# json.dump(summary_dict, open('/tmp/summary.json', 'w'), cls=NumpyAwareJSONEncoder)


if __name__ == '__main__':


	parser = argparse.ArgumentParser()
	parser.add_argument("stack", type=str, help="choose what stack of images to crop and resolution, ex: RS141")
	args = parser.parse_args()
	stack = args.stack
	
	data_dir_stack = create_if_not_exists(os.path.join(os.environ['GORDON_SECTIONDATA_DIR'], stack))	# DavidData2015sections/CC35

	autogen_masked_img_dir = create_if_not_exists(os.path.join(data_dir_stack, 'autogen_maskedimg_x0.3125'))		# DavidData2015sections/CC35/autogen_masked_x0.3125
	autogen_bbox_dir = create_if_not_exists(os.path.join(data_dir_stack, 'autogen_bbox_x0.3125'))		# DavidData2015sections/CC35/autogen_bbox

	for l, resol in enumerate(['x0.3125', 'x1.25', 'x5', 'x20']):
		data_dir_resol = create_if_not_exists(os.path.join(data_dir_stack, resol))		# DavidData2015sections/CC35/x0.3125

	summarize_bbox_and_masks(bbox_dir=autogen_bbox_dir, mask_dir=autogen_masked_img_dir)