import sys

sys.path.append('/home/yuncong/morphsnakes')
import morphsnakes

import numpy as np
from skimage.io import imread
from matplotlib import pyplot as plt
from skimage.color import rgb2gray, rgb2hsv

import Image

# imgcolor = Image.open("/home/yuncong/DavidData2015tifFlat/CC35_x0.3125/CC35_x0.3125_0050.tif")

# imgcolor = Image.open("/home/yuncong/DavidData2015tifFlat/x0.3125_slide/CC35_08_x0.3125_z0.tif")
imgcolor = Image.open("/home/yuncong/DavidData2015tifFlat/DAPIcy3NTgreenGlyc_27_x0.3125_z0.tif")
imgcolor = imgcolor.convert('RGB')

imgcolor = np.array(imgcolor)
img_gray = rgb2gray(imgcolor)

# imghsv = rgb2hsv(imgcolor)
# img_sat = imghsv[...,1]

gI = morphsnakes.gborders(img_gray, alpha=20000, sigma=1)

# Morphological GAC. Initialization of the level-set.
mgac = morphsnakes.MorphGAC(gI, smoothing=2, threshold=0.3, balloon=-3)

mgac.levelset = np.ones_like(img_gray)
mgac.levelset[:3,:] = 0
mgac.levelset[-3:,:] = 0
mgac.levelset[:,:3] = 0
mgac.levelset[:,-3:] = 0

# Visual evolution.
plt.figure()
morphsnakes.evolve_visual(mgac, num_iters=500, background=img_gray)