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

def foreground_mask_morphsnakes_slide(img, levelset=None, max_iters=1000, num_section_per_slide=5, min_iters=300):

	# img = denoise_bilateral(img, win_size=5, sigma_range=1, sigma_spatial=7, bins=10000, mode='constant', cval=0)

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
	for i, box in enumerate(column_bboxes[indices, :]):
		minr, minc, maxr, maxc = box

		# touch_top
		t = 0 if np.std(img[minr, minc:maxc+1]) > .06 else 5
		# touch_bottom
		b = 0 if np.std(img[maxr-1, minc:maxc+1]) > .06 else 5
		# touch_left
		l = 0 if np.std(img[minr:maxr+1, minc]) > .06 else 5
		# touch_right
		r = 0 if np.std(img[minr:maxr+1, maxc-1]) > .06 else 5

		mask[minr+t:maxr-b, minc+l:maxc-r] = 1

	section_masks, bboxes, exists = foreground_mask_morphsnakes_slide(img, levelset=mask, 
				num_section_per_slide=num_section_per_slide, min_iters=400)

	ids = np.where(exists)[0]
	#     section_images = []
	for _, ((minr, minc, maxr, maxc), mask, section_id) in enumerate(zip(bboxes, section_masks, ids)):
		img = imgcolor[minr:maxr+1, minc:maxc+1].copy()
		mask = 255 * mask.astype(np.uint8)
		masked_img = np.dstack([img, mask])

		# img[~mask] = 0
		#         section_images.append(img)

		image_filepath = os.path.join(out_dir, '%s_x0.3125_%02d_%d.tif'%(stack, slide_ind, section_id))
		imsave(image_filepath, masked_img)

	# bboxes_percent = np.array(bboxes, dtype=np.float)/np.r_[img_gray.shape[:2], img_gray.shape[:2]][np.newaxis, :]

	filled_bboxes = -1 * np.ones((len(exists), 4), dtype=np.int)
	filled_bboxes[ids, :] = bboxes

	return filled_bboxes