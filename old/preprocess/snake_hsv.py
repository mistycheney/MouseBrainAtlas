import sys

sys.path.append('morphsnakes')
import morphsnakes

import numpy as np
from skimage.io import imread
from matplotlib import pyplot as plt
from skimage.color import rgb2gray, rgb2hsv
from skimage.restoration import denoise_bilateral
from skimage.util import img_as_float

from skimage.morphology import watershed

import Image

# imgcolor = Image.open("/home/yuncong/DavidData2015tifFlat/CC35_x0.3125/CC35_x0.3125_0050.tif")

# imgcolor = Image.open("/home/yuncong/DavidData2015tifFlat/x0.3125_slide/CC35_08_x0.3125_z0.tif")
# imgcolor = Image.open("/home/yuncong/DavidData2015tifFlat/DAPIcy3NTgreenGlyc_27_x0.3125_z0.tif")
# imgcolor = imgcolor.convert('RGB')
# imgcolor = np.array(imgcolor)

imgcolor = imread('/home/yuncong/cur_resol_sectionimg.tif')
# img_gray = rgb2gray(imgcolor)

# img_gray = denoise_bilateral(img_gray, win_size=7, sigma_range=1, 
# 		sigma_spatial=7, bins=10000, mode='constant', cval=0)

imghsv = rgb2hsv(imgcolor)
img_gray = (imghsv[...,1] < .08).astype(np.float)

markers = np.zeros_like(img_gray)
markers[5,5] = 1
markers[-5,5] = 1
markers[5,-5] = 1
markers[-5,-5] = 1

mask = imread('/home/yuncong/cur_resol_alpha.tif') > 0

watershed(img_gray, markers, connectivity=4, offset=None, mask=mask)

plt.imshow(img_gray)
plt.colorbar()
plt.show()

exit()

# gI = morphsnakes.gborders(img_gray, alpha=1000, sigma=3)

# Morphological GAC. Initialization of the level-set.
# mgac = morphsnakes.MorphGAC(gI, smoothing=1, threshold=.8, balloon=-1)

# mgac = morphsnakes.MorphGAC(gI, smoothing=1, threshold=.8, balloon=-1)



mgac.levelset = mask

diffs = []

import time

 # Iterate.
# num_iters = 1000
# for i in xrange(1000):

# 	print i

# 	b = time.time()

# 	mgac.step()

# 	print time.time() - b

	# if i > 0:
	#     if i > 10:
	#         diff = np.count_nonzero(mgac.levelset - prev_levelset != 0)
	#         diffs.append(diff)
	#         if np.abs(np.mean(diffs[-5:]) - np.mean(diffs[-10:-5])) < 2:
	#         	break
	#     prev_levelset = mgac.levelset

	# if i >= 100:
	#     plt.plot(diffs)
	#     plt.show()


# mgac.levelset = np.ones_like(img_gray)
# mgac.levelset[:2,:] = 0
# mgac.levelset[-2:,:] = 0
# mgac.levelset[:,:2] = 0
# mgac.levelset[:,-2:] = 0

# Visual evolution.
plt.figure()
morphsnakes.evolve_visual(mgac, num_iters=500, background=img_gray)