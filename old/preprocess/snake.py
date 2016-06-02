import sys

sys.path.append('/home/yuncong/morphsnakes')
import morphsnakes

import numpy as np
from skimage.io import imread
from matplotlib import pyplot as plt
from skimage.color import rgb2gray, rgb2hsv

imgcolor = imread("/home/yuncong/DavidData2015tifFlat/CC35_x0.3125/CC35_x0.3125_0281.tif")/255.
img = rgb2gray(imgcolor)

gI = morphsnakes.gborders(img, alpha=20000, sigma=1)

# Morphological GAC. Initialization of the level-set.
mgac = morphsnakes.MorphGAC(gI, smoothing=2, threshold=0.3, balloon=-3)
mgac.levelset = np.ones_like(img)
mgac.levelset[:3,:] = 0
mgac.levelset[-3:,:] = 0
mgac.levelset[:,:3] = 0
mgac.levelset[:,-3:] = 0

# Visual evolution.
plt.figure()
morphsnakes.evolve_visual(mgac, num_iters=500, background=imgcolor)