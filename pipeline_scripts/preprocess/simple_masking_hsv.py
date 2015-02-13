import sys

sys.path.append('morphsnakes')
import morphsnakes

import numpy as np
from skimage.io import imread
from matplotlib import pyplot as plt
from skimage.color import rgb2gray, rgb2hsv
from skimage.restoration import denoise_bilateral
from skimage.util import img_as_float, img_as_ubyte
from skimage.measure import label, regionprops
from skimage.morphology import watershed, remove_small_objects

import Image

imgcolor = imread('/home/yuncong/cur_resol_sectionimg.tif')

imghsv = rgb2hsv(imgcolor)
img_gray = imghsv[...,1]
img_gray = img_gray < 0.08
img_gray = remove_small_objects(img_gray, min_size=img_gray.size * .1, connectivity=8, in_place=False)

# plt.imshow(img_gray)
# plt.colorbar()
# plt.show()

labels, n_labels = label(img_gray, neighbors=8, return_num=True, background=0)

regions = regionprops(labels+1)
border_regions = [r for r in regions if (r.bbox == np.r_[0, 0, img_gray.shape[0]-1, img_gray.shape[1]-1]).any()]

img_alpha = np.ones_like(img_gray)
for r in border_regions:

	# plt.imshow(r.image)
	# plt.show()

	img_alpha[r.image] = 0

img_rgba = np.dstack([img_as_float(imgcolor), img_alpha])

plt.imshow(img_rgba)
plt.show()

exit()

plt.imshow(img_gray)
plt.colorbar()
plt.show()

markers = np.zeros_like(img_gray)
markers[5,5] = 1
markers[-5,5] = 1
markers[5,-5] = 1
markers[-5,-5] = 1

mask = imread('/home/yuncong/cur_resol_alpha.tif') > 0

out = watershed(img_gray, markers, mask=mask)

plt.imshow(out)
plt.colorbar()
plt.show()

exit()