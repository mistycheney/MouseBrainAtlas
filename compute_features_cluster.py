# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sys, os
from scipy.ndimage import measurements
from skimage.filter import threshold_otsu, threshold_adaptive, denoise_bilateral, gaussian_filter
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

from IPython.display import FileLink

import cv2

import time
from multiprocessing import Pool

import skimage
from skimage.color import color_dict

from skimage.segmentation import clear_border
from skimage.filter import threshold_otsu, threshold_adaptive, gaussian_filter
from skimage.morphology import binary_dilation, binary_erosion
from skimage.morphology import watershed

from skimage.morphology import label
from skimage.segmentation import slic, mark_boundaries
from skimage.color import gray2rgb, label2rgb
from skimage.measure import regionprops
from scipy.spatial.distance import pdist, squareform
from skimage.util import img_as_ubyte
from IPython.display import FileLink, Image

# <codecell>

CACHE_DIR = '/home/yuncong/my_csd181_scratch/'
IMG_DIR = '/home/yuncong/my_csd181/ParthaData/PMD1305_reduce2/region3/'
img_name_fmt = 'PMD1305_%d.reduce2.region3'
img_id = 4
img_name = img_name_fmt%img_id
img = cv2.imread(IMG_DIR + img_name + '.tif', 0)

# <codecell>

mask = foreground_mask(img)

# <codecell>

from skimage.filter import gabor_kernel

kernels = []
kernel_freqs = []
kernel_thetas = []

theta_interval = 3

n_freqs = 4
# freq_max = np.sqrt(2)/2**2/(2*np.pi)
freq_max = 1./5.
frequencies = np.array([freq_max/2**i for i in range(n_freqs)])

for freq in frequencies:
    for theta in np.arange(0, np.pi, np.deg2rad(theta_interval)):
        kernel = gabor_kernel(freq, theta=theta,
                              bandwidth=1.)
        kernels.append(kernel)
        kernel_freqs.append(freq)
        kernel_thetas.append(theta)

kernels = map(np.real, kernels)
kernel_thetas = np.array(kernel_thetas)
kernel_freqs = np.array(kernel_freqs)

print '# of kernels = %d' % (len(kernels))
print 'frequencies:', frequencies
print '# of pixels per cycle:', 1/frequencies
print 'kernel size:', [kern.shape[0] for kern in kernels]

def _convolve_per_proc(i):
    return fftconvolve(img, kernels[i], 'same')

def _gaussian_filter_per_proc((sigma, filtered)):
    alpha = 0.25
    nonlinear = np.tanh(alpha * filtered)
    return nonlinear

# <codecell>

filtered = filtered[100:-100, 100:-100, :]
img = img[100:-100, 100:-100]
mask = mask[100:-100, 100:-100]
dirmap = dirmap[100:-100, 100:-100]

# <codecell>

img_rgb = gray2rgb(img)
segmentation = slic(img_rgb, n_segments=2000, max_iter=10, compactness=5, sigma=10, enforce_connectivity=True)

# <codecell>

from sklearn.cluster import MiniBatchKMeans 
from skimage.color import label2rgb
num_textons = 20
kmeans_model = MiniBatchKMeans(num_textons)
kmeans_model.fit(features[mask,:].reshape(-1, n_feat))

# <codecell>

n_superpixels = len(np.unique(segmentation))

img_superpixelized = mark_boundaries(img_rgb, segmentation)
superpixel_boundaries = find_boundaries(segmentation)
img_superpixelized_patches = label2rgb(segmentation, img_rgb)

sp_props = regionprops(segmentation+1, intensity_image=img, cache=True)
sp_centroids = np.array([s.centroid for s in sp_props])
sp_areas = np.array([s.area for s in sp_props])
# sp_wcentroids = np.array([s.weighted_centroid for s in sp_props])
sp_centroid_dist = pdist(sp_centroids)
sp_centroid_dist_matrix = squareform(sp_centroid_dist)

sp_mean_intensity = np.array([s.mean_intensity for s in sp_props])

# cv2.imwrite(img_name+'_superpixelized.jpg', img_as_ubyte(img_superpixelized))
# Image(img_name+'_superpixelized.jpg')

# colors = np.random.random((n_superpixels,3))
# a = label2rgb(segmentation, img_rgb, colors=colors)

img_superpixelized_text = img_as_ubyte(img_superpixelized)
for s in range(n_superpixels):
    img_superpixelized_text = cv2.putText(img_superpixelized_text, str(s), 
                                          tuple(np.floor(sp_centroids[s][::-1]).astype(np.int)), 
                                          cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                          .5, ((255,0,255)), 1)
    
img_superpixelized_text = img_superpixelized_text/255.

img_superpixelized_patches = label2rgb(segmentation, img)

# <codecell>

from matplotlib.backends.backend_pdf import PdfPages

pdf = PdfPages('region2_collection.pdf')

img_name = img_name_fmt%img_id

plt.imshow(cv2.imread('output/'+img_name+'_features_%d.png'%img_id))
plt.title('features, image %d'% img_id, fontsize=5)
plt.axis('off')
pdf.savefig(bbox_inches='tight', dpi=300)
plt.close();

fig, axes = plt.subplots(nrows=2, ncols=2)

#         img_name = 'PMD1305_%d.reduce2.region1'%img_id    
#         im = cv2.imread(IMG_DIR + img_name + '.tif')

fig.suptitle('image %d' % img_id, fontsize=5)

axes[0,0].imshow(cv2.imread('output/'+img_name+'_superpixelized_%d.png'%img_id))
axes[0,0].set_title('superpixelized, compactness=%d, presmooth=%.2f'%(compactness,presmooth), fontsize=5)
axes[0,0].axis('off')

axes[0,1].imshow(cv2.imread('output/'+img_name+'_textonmap_%d.png'%img_id))
axes[0,1].set_title('texton map, num_textons=%d'%num_textons, fontsize=5)
axes[0,1].axis('off')

axes[1,0].imshow(cv2.imread('output/'+img_name+'_saliencymap_%d.png'%img_id))
axes[1,0].set_title('saliency map, neighbor_term_weight=%.2f'%neighbor_term_weight, fontsize=5)
axes[1,0].axis('off')

axes[1,1].imshow(cv2.imread('output/'+img_name+'_salientclusters_%d.png'%img_id))
axes[1,1].set_title('10 most salient clusters, cluster_grow_thresh=%.2f'%dist_thresh, fontsize=5)
axes[1,1].axis('off')

pdf.savefig(bbox_inches='tight', dpi=300)
plt.close();

