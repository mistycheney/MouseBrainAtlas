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


def foreground_mask_morphsnakes_slide(img, num_iters=1000):

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
				if diff == previous_diff and diff < 40 and i > 300: # oscillate
					break

		previous_levelset = msnake.levelset

	# plt.figure()
	# morphsnakes.evolve_visual(msnake, num_iters=num_iters, background=img)

	blob_labels, n_labels = label(msnake.levelset, neighbors=4, return_num=True, background=0)

	blob_props = regionprops(blob_labels + 1)
	all_areas = [p.area for p in blob_props]

	indices = np.argsort(all_areas)[::-1]
	largest = indices[0]

	largest_area = all_areas[largest]

	valid = np.where(all_areas > largest_area * .6)[0]

	blob_props_good = [blob_props[i] for i in valid]

	bboxes = [p.bbox for p in blob_props_good]

	masks = []

	for i, (minr, minc, maxr, maxc) in enumerate(bboxes):
		mask = np.zeros_like(img, dtype=np.bool)
		mask[blob_labels == i] = 1
		
		section_mask = mask[minr:maxr+1, minc:maxc+1]
		
		masks.append(section_mask)

	return masks, bboxes

stack = 'CC35'
bboxes_json = defaultdict(list)

use_hsv = True	# set to True for Nissl stains, False for Fluorescent stains

for slide_ind in range(4, 55):

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

	section_masks, bboxes = foreground_mask_morphsnakes_slide(img, num_iters=400)
	
	section_images = []
	for i, ((minr, minc, maxr, maxc), mask) in enumerate(zip(bboxes, section_masks)):
		img = imgcolor[minr:maxr+1, minc:maxc+1].copy()
		img[~mask] = 0
		section_images.append(img)

	for im in section_images:
		plt.imshow(im)
		plt.show()

 # 	bbox_dir = os.path.join(os.environ['LOCAL_REPO_DIR'], 'pipeline_scripts', 'preprocess', 'bbox')
	# bbox_filepath = os.path.join(bbox_dir, stack + '_bbox.json')
	# json.dump(bboxes_json, open(bbox_filepath, 'w'))
