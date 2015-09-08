import sys
import os

sys.path.append('morphsnakes')
import morphsnakes

import numpy as np
from skimage.io import imread, imsave
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

from skimage.restoration import denoise_bilateral

def foreground_mask_morphsnakes_slide(img, levelset=None, max_iters=1000, num_section_per_slide=5, min_iters=300):

	# ls = pickle.load(open('/tmp/levelset.pkl', 'rb'))

	# img = denoise_bilateral(img, win_size=5, sigma_range=1, sigma_spatial=3, bins=10000, mode='constant', cval=0)

	# plt.imshow(img)
	# plt.show()

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

	# for i in xrange(max_iters):
	# 	msnake.step()
	# 	if i > 0:
	# 		diff = np.count_nonzero(msnake.levelset - previous_levelset < 0)
	# 		print i, diff
	# 		if diff < 40 and i > min_iters: # oscillate
	# 			break
	# 	previous_levelset = msnake.levelset

	plt.figure()
	morphsnakes.evolve_visual(msnake, num_iters=max_iters, background=img)
	
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


if __name__ == '__main__':

	stack = sys.argv[1]
	bboxes_json = defaultdict(list)

	use_hsv = True	# set to True for Nissl stains, False for Fluorescent stains

	# for slide_ind in range(24, 55):

		# print 'slide', slide_ind

	slide_ind = int(sys.argv[2])

	
	imgcolor = Image.open("/home/yuncong/DavidData2015slides/%s/x0.3125/%s_%02d_x0.3125_z0.tif" % (stack, stack, slide_ind))
	# imgcolor = Image.open("/home/yuncong/DavidData2015slides/%s/x5/%s_%02d_x5_z0.tif" % (stack, stack, slide_ind))
	imgcolor = imgcolor.convert('RGB')
	imgcolor = np.array(imgcolor)
	img_gray = rgb2gray(imgcolor)
	slide_height, slide_width = img_gray.shape[:2]

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
	print n_columns, 'columns detected'
	column_props = regionprops(column_labels+1)
	column_bboxes = np.array([p.bbox for p in column_props])
	indices = np.argsort(column_bboxes[:,1])

	mask = np.zeros_like(img_gray, dtype=np.bool)
	for i, box in enumerate(column_bboxes[indices, :]):
		minr, minc, maxr, maxc = box

		# # touch_top
		# t = 0 if len(np.unique(img[minr, minc:maxc+1])) > 100 else 5
		# # touch_bottom
		# b = 0 if len(np.unique(img[maxr-1, minc:maxc+1])) > 100 else 5
		# # touch_left
		# l = 0 if len(np.unique(img[minr:maxr+1, minc])) > 100 else 5
		# # touch_right
		# r = 0 if len(np.unique(img[minr:maxr+1, maxc-1])) > 100 else 5

		# touch_top
		t = 0 if np.std(img[minr, minc:maxc+1]) > .06 else 5
		# touch_bottom
		b = 0 if np.std(img[maxr-1, minc:maxc+1]) > .06 else 5
		# touch_left
		l = 0 if np.std(img[minr:maxr+1, minc]) > .06 else 5
		# touch_right
		r = 0 if np.std(img[minr:maxr+1, maxc-1]) > .06 else 5

		# print img[maxr-1, minc:maxc+1]

		print np.std(img[minr, minc:maxc+1]),  np.std(img[maxr-1, minc:maxc+1]), np.std(img[minr:maxr+1, minc]),  np.std(img[minr:maxr+1, maxc-1])

		mask[minr+t:maxr-b, minc+l:maxc-r] = 1

	# plt.imshow(a, cmap=plt.cm.gray)
	# plt.show()

	# plt.imshow(imghsv[...,0], cmap=plt.cm.gray)
	# plt.show()
	plt.imshow(imghsv[...,1], cmap=plt.cm.gray)
	plt.show()
	# plt.imshow(imghsv[...,2], cmap=plt.cm.gray)
	# plt.show()

	# exit()

	num_section_per_slide = 5
	section_masks, bboxes, exists = foreground_mask_morphsnakes_slide(img, levelset=mask, 
						max_iters=1000, min_iters=400, num_section_per_slide=5)
	print 'exists', exists

	if not os.path.exists('curr_session'):
		os.makedirs('curr_session')

	section_images = []
	for (minr, minc, maxr, maxc), mask in zip(bboxes, section_masks):
		img = imgcolor[minr:maxr+1, minc:maxc+1].copy()
		img[~mask] = 0
		section_images.append(img)

	ids = np.where(exists)[0]
	for im, i in zip(section_images, ids):
		plt.imshow(im, aspect='equal')
		plt.title('section %d' % i)
		plt.show()

		# imsave('curr_session/'+str(i)+'.tif', im)

	# bboxes_percent = np.array(bboxes, dtype=np.float)/np.r_[img_gray.shape[:2], img_gray.shape[:2]][np.newaxis, :]
	# print bboxes_percent
	# np.savetxt('curr_session/bbox_curr_session.txt', bboxes_percent, fmt='%.3f')


