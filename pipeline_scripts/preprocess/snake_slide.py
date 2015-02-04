import sys

sys.path.append('morphsnakes')
import morphsnakes

import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray, rgb2hsv
from skimage.filter.rank import noise_filter, median
from skimage.morphology import disk, remove_small_objects
from skimage.measure import regionprops, label

from matplotlib import pyplot as plt
import Image

from collections import defaultdict
import json


def foreground_mask_morphsnakes(img, num_iters=1000):

	gI = morphsnakes.gborders(img, alpha=20000, sigma=1)

	msnake = morphsnakes.MorphGAC(gI, smoothing=2, threshold=0.3, balloon=-3)

	msnake.levelset = np.ones_like(img)
	msnake.levelset[:3,:] = 0
	msnake.levelset[-3:,:] = 0
	msnake.levelset[:,:3] = 0
	msnake.levelset[:,-3:] = 0

	for i in xrange(num_iters):
		msnake.step()

		if i > 0:
			if i > 1:
				previous_diff = diff
			diff = np.count_nonzero(msnake.levelset - previous_levelset)
			# print i, diff
			if i > 1:
				if diff == previous_diff and diff < 40 and i > 500: # oscillate
					break

		previous_levelset = msnake.levelset

	# plt.figure()
	# morphsnakes.evolve_visual(msnake, num_iters=600, background=img)

	blob_labels, n_labels = label(msnake.levelset, neighbors=4, return_num=True, background=0)

	blob_props = regionprops(blob_labels + 1)
	all_areas = [p.area for p in blob_props]
	indices = np.argsort(all_areas)[::-1]
	largest_blob = indices[0]

	n_sections = 1

	if len(all_areas) > 1:	
		second_blob = indices[1]
		if all_areas[second_blob] / all_areas[largest_blob] > 0.6:
			print 'two sections in this image'
			n_sections = 2

	masks = []
	for sec_i in range(n_sections):

		mask = np.zeros_like(msnake.levelset, dtype=np.bool)
		mask[blob_labels == indices[sec_i]] = 1

		min_size = 40
		mask = remove_small_objects(mask, min_size=min_size, connectivity=1, in_place=False)

		masks.append(mask)

	return masks

stack = 'CC35'
bboxes_json = defaultdict(list)

use_hsv = True	# set to True for Nissl stains, False for Fluorescent stains

for slide_ind in range(3, 55):

	print 'slide', slide_ind

	# slide_ind = 23
	imgcolor = Image.open("/home/yuncong/DavidData2015tifFlat/x0.3125_slide/CC35_%02d_x0.3125_z0.tif" % slide_ind)
	imgcolor = imgcolor.convert('RGB')
	imgcolor = np.array(imgcolor)
	img_gray = rgb2gray(imgcolor)
	slide_height, slide_width = img_gray.shape[:2]

	if use_hsv:
		imghsv = rgb2hsv(imgcolor)
		img = imghsv[...,1]
	else:
		img = img_gray

	# plt.imshow(imghsv[...,0], cmap=plt.cm.gray)
	# plt.show()
	# plt.imshow(imghsv[...,1], cmap=plt.cm.gray)
	# plt.show()
	# plt.imshow(imghsv[...,2], cmap=plt.cm.gray)
	# plt.show()

	a = img_gray < 0.98

	a = median(a.astype(np.float), disk(3))
	a = remove_small_objects(a.astype(np.bool), min_size=50, connectivity=2, in_place=False)

	# plt.imshow(a, cmap=plt.cm.gray)
	# plt.show()

	column_labels, n_columns = label(a, neighbors=8, return_num=True, background=0)
	print n_columns, 'columns detected'

	props = regionprops(column_labels+1)

	masks = []
	images = []
	centers = []
	bboxes = []

	for i, p in enumerate(props):

		print 'column', i
		min_row, min_col, max_row, max_col = p.bbox

		column_im = img[min_row:max_row, min_col:max_col]

		# plt.imshow(column_sat, cmap=plt.cm.gray)
		# plt.show()

		col_masks = foreground_mask_morphsnakes(column_im, num_iters=800)
		print '%d sections found' % len(col_masks)
		
		for sec_i, mask in enumerate(col_masks):
			
			sec_prop = regionprops(mask)[0]

			min_sec_row, min_sec_col, max_sec_row, max_sec_col = sec_prop.bbox

			sec_mask = mask[min_sec_row:max_sec_row, min_sec_col:max_sec_col]
			masks.append(sec_mask)

			center = np.array([min_row, min_col]) + sec_prop.centroid
			center_perc = center/imgcolor.shape[:2]
			centers.append(center_perc)

			min_sec_row_perc = (min_row + min_sec_row) / float(slide_height)
			max_sec_row_perc = (min_row + max_sec_row) / float(slide_height)
			min_sec_col_perc = (min_col + min_sec_col) / float(slide_width)
			max_sec_col_perc = (min_col + max_sec_col) / float(slide_width)
			bbox_perc = (min_sec_row_perc, min_sec_col_perc, max_sec_row_perc, max_sec_col_perc)

			bboxes.append(bbox_perc)

			sec_rgb = imgcolor[min_row + min_sec_row : min_row + max_sec_row, min_col + min_sec_col : min_col + max_sec_col].copy()
			sec_rgb[~sec_mask] = 0

			images.append(sec_rgb)

			# plt.imshow(sec_rgb, cmap=plt.cm.gray)
			# plt.show()


	n_sections = len(images)

	bboxes_json[slide_ind] = bboxes

	# ax1 = plt.subplot2grid((2,n_sections), (0,0), colspan=n_sections)

	# # plt.figure()
	# ax1.imshow(imgcolor)
	# ax1.set_title("slide %d" % slide_ind)
	# ax1.axis('off')

	# # fig, axes = plt.subplots(1, n_sections)
	# if n_columns < 8:
	# 	for i in range(n_sections):
	# 		print 'slide', slide_ind, 'section', i, bboxes[i], centers[i]
			
	# 		ax = plt.subplot2grid((2,n_sections), (1,i), colspan=1)
	# 		ax.imshow(images[i])
	# 		ax.set_title('section %d' % i)
	# 		ax.axis('off')
	# else:
	# 	raise

	# plt.show()

 	bbox_dir = os.path.join(os.environ['LOCAL_REPO_DIR'], 'pipeline_scripts', 'preprocess', 'bbox')
	bbox_filepath = os.path.join(bbox_dir, stack + '_bbox.json')
	json.dump(bboxes_json, open(bbox_filepath, 'w'))
