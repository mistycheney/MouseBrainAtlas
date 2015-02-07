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
from PIL import Image

from collections import defaultdict
import json

from itertools import chain
import cPickle as pickle

def foreground_mask_morphsnakes_slide(img, num_iters=1000, num_section_per_slide=5):

	# ls = pickle.load(open('/tmp/levelset.pkl', 'rb'))

	gI = morphsnakes.gborders(img, alpha=20000, sigma=1)

	msnake = morphsnakes.MorphGAC(gI, smoothing=2, threshold=0.3, balloon=-3)

	msnake.levelset = np.ones_like(img)
	msnake.levelset[:3,:] = 0
	msnake.levelset[-3:,:] = 0
	msnake.levelset[:,:3] = 0
	msnake.levelset[:,-3:] = 0

	for i in xrange(num_iters):
		# print i
		msnake.step()
		if i > 0:
			diff = np.count_nonzero(msnake.levelset - previous_levelset < 0)
			# print i, diff
			if diff < 40 and i > 300: # oscillate
				break

		previous_levelset = msnake.levelset

	# plt.figure()
	# morphsnakes.evolve_visual(msnake, num_iters=num_iters, background=img)
	
	blob_labels, n_labels = label(msnake.levelset, neighbors=4, return_num=True, background=0)

	# pickle.dump(msnake.levelset, open('/tmp/levelset.pkl', 'wb'))

	# blob_labels, n_labels = label(ls, neighbors=4, return_num=True, background=0)

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

	print 'valid', valid

	height, width = img.shape[:2]


	if len(valid) > num_section_per_slide:

		indices_close = np.where(np.diff(centers_x) < width * .1)[0]
		print indices_close

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

		print ups, downs
		arrangement = np.r_[ups, downs]

	elif len(valid) < num_section_per_slide:
		snap_to_columns = (np.round((centers_x / width + 0.1) * num_section_per_slide) - 1).astype(np.int)
		print 'snap_to_columns', snap_to_columns

		arrangement = -1 * np.ones((num_section_per_slide,), dtype=np.int)
		arrangement[snap_to_columns] = valid
	else:
		arrangement = valid

	print 'arrangement', arrangement

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

stack = 'CC35'
bboxes_json = defaultdict(list)

use_hsv = True	# set to True for Nissl stains, False for Fluorescent stains

# for slide_ind in range(24, 55):

	# print 'slide', slide_ind

slide_ind = 54
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

num_section_per_slide = 5
section_masks, bboxes, exists = foreground_mask_morphsnakes_slide(img, num_iters=1000, num_section_per_slide=5)
print 'exists', exists

section_images = []
for (minr, minc, maxr, maxc), mask in zip(bboxes, section_masks):
	img = imgcolor[minr:maxr+1, minc:maxc+1].copy()
	img[~mask] = 0
	section_images.append(img)

ids = np.arange(2*num_section_per_slide)[exists]
for im, i in zip(section_images, ids):
	plt.imshow(im)
	plt.title('section %d' % i)
	plt.show()