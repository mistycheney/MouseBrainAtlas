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
import time

sys.path.append(os.environ['MSNAKES_PATH'])
import morphsnakes

import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray, rgb2hsv
from skimage.io import imread, imsave
from skimage.morphology import disk, remove_small_objects
from skimage.measure import regionprops, label
from skimage.restoration import denoise_bilateral
from skimage.filter.rank import median

from joblib import Parallel, delayed

from PIL import Image

ndpisplit = os.environ['GORDON_NDPISPLIT_PROGRAM']

def foreground_mask_morphsnakes_slide(img, levelset=None, max_iters=1000, num_section_per_slide=5, min_iters=300):

	img = denoise_bilateral(img, win_size=7, sigma_range=1, sigma_spatial=7, bins=10000, mode='constant', cval=0)

	gI = morphsnakes.gborders(img, alpha=15000, sigma=1)

	msnake = morphsnakes.MorphGAC(gI, smoothing=2, threshold=0.3, balloon=-3)

	if levelset is None:
		msnake.levelset = np.ones_like(img)
		msnake.levelset[:2,:] = 0
		msnake.levelset[-2:,:] = 0
		msnake.levelset[:,:2] = 0
		msnake.levelset[:,-2:] = 0
	else:
		msnake.levelset = levelset

	for i in xrange(max_iters):
		msnake.step()
		if i > 0:
			diff = np.count_nonzero(msnake.levelset - previous_levelset < 0)
			if diff < 40 and i > min_iters: # oscillate
				break

		previous_levelset = msnake.levelset

	blob_labels, n_labels = label(msnake.levelset, neighbors=4, return_num=True, background=0)

	blob_props = regionprops(blob_labels + 1)
	all_areas = np.array([p.area for p in blob_props])
	all_centers = np.array([p.centroid for p in blob_props])
	all_bboxes = np.array([p.bbox for p in blob_props])

	indices = np.argsort(all_areas)[::-1]
	largest_area = all_areas[indices[:2]].mean()

	valid = np.where(all_areas > largest_area * .45)[0]

	valid = valid[np.argsort(all_centers[valid, 1])]
	centers_x = all_centers[valid, 1]
	centers_y = all_centers[valid, 0]

	height, width = img.shape[:2]


	if len(valid) > num_section_per_slide:
		indices_close = np.where(np.diff(centers_x) < width * .1)[0]

		ups = []
		downs = []

		if len(indices_close) > 0:

			for i in range(len(valid)):
				if i-1 in indices_close:
					continue
				elif i in indices_close:
					if centers_y[i] > height * .5:
						ups.append(valid[i+1])
						downs.append(valid[i])
					else:
						ups.append(valid[i])
						downs.append(valid[i+1])
				else:
					if centers_y[i] > height * .5:
						ups.append(-1)
						downs.append(valid[i])
					else:
						ups.append(valid[i])
						downs.append(-1)

		arrangement = np.r_[ups, downs]

	elif len(valid) < num_section_per_slide:
		snap_to_columns = (np.round((centers_x / width + 0.1) * num_section_per_slide) - 1).astype(np.int)

		arrangement = -1 * np.ones((num_section_per_slide,), dtype=np.int)
		arrangement[snap_to_columns] = valid
	else:
		arrangement = valid

	bboxes = []
	masks = []

	for i, j in enumerate(arrangement):
		if j == -1: continue

		minr, minc, maxr, maxc = all_bboxes[j]
		bboxes.append(all_bboxes[j])

		mask = np.zeros_like(img, dtype=np.bool)
		mask[blob_labels == j] = 1

		section_mask = mask[minr:maxr+1, minc:maxc+1]

		masks.append(section_mask)

	return masks, bboxes, arrangement > -1

def gen_mask(slide_ind=None, in_dir=None, stack=None, use_hsv=True, out_dir=None, num_section_per_slide=5, mirror=None, rotate=None):

	imgcolor = Image.open(os.path.join(in_dir, '%s_%02d_x0.3125_z0.tif'%(stack, slide_ind)))
	imgcolor = imgcolor.convert('RGB')
	imgcolor = np.array(imgcolor)
	img_gray = rgb2gray(imgcolor)
	# slide_height, slide_width = img_gray.shape[:2]

	if use_hsv:
		imghsv = rgb2hsv(imgcolor)
		img = imghsv[...,1]
	else:
		img = img_gray

# utilize bounding box selected during scanning process
	a = img_gray < 0.98

	a = median(a.astype(np.float), disk(3))
	a = remove_small_objects(a.astype(np.bool), min_size=50, connectivity=2, in_place=False)
	column_labels, n_columns = label(a, neighbors=8, return_num=True, background=0)
	column_props = regionprops(column_labels+1)
	column_bboxes = np.array([p.bbox for p in column_props])
	indices = np.argsort(column_bboxes[:,1])

	mask = np.zeros_like(img_gray, dtype=np.bool)
	col_margin = 2
	row_margin = 2
	for i, b in enumerate(column_bboxes[indices, :]):
		minr, minc, maxr, maxc = b
		mask[minr+row_margin:maxr-row_margin, minc+col_margin:maxc-col_margin] = 1

	section_masks, bboxes, exists = foreground_mask_morphsnakes_slide(img, levelset=mask, 
				num_section_per_slide=num_section_per_slide, min_iters=400)

	ids = np.where(exists)[0]
	#     section_images = []
	for _, ((minr, minc, maxr, maxc), mask, section_id) in enumerate(zip(bboxes, section_masks, ids)):
		img = imgcolor[minr:maxr+1, minc:maxc+1].copy()
		img[~mask] = 0
		#         section_images.append(img)

		image_filepath = os.path.join(out_dir, '%s_x0.3125_%02d_%d.tif'%(stack, slide_ind, section_id))
		imsave(image_filepath, img)

	# bboxes_percent = np.array(bboxes, dtype=np.float)/np.r_[img_gray.shape[:2], img_gray.shape[:2]][np.newaxis, :]

	return bboxes

def create_if_not_exists(path):
	if not os.path.exists(path):
		os.makedirs(path)
	return path

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("stack", type=str, help="choose what stack of images to crop and resolution, ex: RS141")

	args = parser.parse_args()
	
	args_dict = {}

	stack = args.stack
	args_dict['stack'] = stack

	slide_dir_stack = create_if_not_exists(os.path.join(os.environ['GORDON_SLIDEDATA_DIR'], stack))	# DavidData2015slides/CC35

	######################## Split ndpi files ######################## 1 min 42 sec for CC35

	# split_t = time.time()

	# ndpi_stack_dir = os.path.join(os.environ['GORDON_NDPI_DIR'], stack)
	# os.chdir(ndpi_stack_dir)

	# def split_one_slide(ndpi_file):

	# 	execute_command(ndpisplit + ' ' + os.path.join(ndpi_stack_dir, ndpi_file))

	# 	for level in ['macro', 'x0.078125', 'x0.3125', 'x1.25', 'x5', 'x20']:
	# 		slide_dir_resol = create_if_not_exists(os.path.join(slide_dir_stack, level))			# DavidData2015slides/CC35/x0.3125
	# 		for f in glob.glob('*_%s_z0.tif'%level):
	# 			try:
	# 				shutil.move(f, slide_dir_resol)
	# 			except Exception as e:
	# 				print e
	# 				pass

	# 	for level in ['macro', 'map']:
	# 		map_dir = create_if_not_exists(os.path.join(slide_dir_stack, level))			# DavidData2015slides/CC35/map
	# 		for f in glob.glob('*'+level+'*'):
	# 			try:
	# 				shutil.move(f, map_dir)
	# 			except Exception as e:
	# 				print e
	# 				pass

	# ndpi_files = [ndpi_file for ndpi_file in os.listdir('.') if ndpi_file.endswith('ndpi')]

	# Parallel(n_jobs=16)(delayed(split_one_slide)(ndpi_file) for ndpi_file in ndpi_files)


	# print time.time() - split_t, 'seconds'

	####################### Section bounding box and mask generation ##########################

	split_t = time.time()


	data_dir_stack = create_if_not_exists(os.path.join(os.environ['GORDON_SECTIONDATA_DIR'], stack))	# DavidData2015sections/CC35

	autogen_masked_img_dir = create_if_not_exists(os.path.join(data_dir_stack, 'autogen_maskedimg_x0.3125'))		# DavidData2015sections/CC35/autogen_masked_x0.3125
	autogen_bbox_dir = create_if_not_exists(os.path.join(data_dir_stack, 'autogen_bbox_x0.3125'))		# DavidData2015sections/CC35/autogen_bbox

	slide_dir = os.path.join(slide_dir_stack, 'x0.3125')		# DavidData2015slides/CC35/x0.3125

	n_slides = len([f for f in os.listdir(slide_dir) if f.endswith('.tif')])
	# n_slides = 3

	options = [{'use_hsv':True, 'in_dir':slide_dir, 'slide_ind':i+1, 'stack':stack, 
					'out_dir':autogen_masked_img_dir, 'num_section_per_slide': 5} for i in range(n_slides)] # set to True for Nissl stains, False for Fluorescent stains

	# gen_mask(**options[22])

	bboxes = Parallel(n_jobs=16)(delayed(gen_mask)(**kwargs) for kwargs in options)

	for slide_ind, bboxes_slide in enumerate(bboxes):
		bbox_filepath = os.path.join(autogen_bbox_dir, 'bbox_%s_x0.3125_%02d.txt'%(stack, slide_ind+1))		# DavidData2015sections/CC35/autogen_bbox/bbox_CC35_03.txt
		np.savetxt(bbox_filepath, bboxes_slide, fmt='%d')

	print time.time() - split_t, 'seconds'


	####################### Manually inspect masked sections and bounding boxes; Modify (using Gimp) if necessary ##########################
	# ...