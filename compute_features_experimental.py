# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

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

%load_ext autoreload
%autoreload 2

# <codecell>

import sys, os
from scipy.ndimage import measurements
from skimage.filter import threshold_otsu, threshold_adaptive, gaussian_filter
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

from IPython.display import FileLink

import cv2

import time
from multiprocessing import Pool

from utilities import *

# CACHE_DIR = '/home/yuncong/my_csd181_scratch/'
# IMG_DIR = '/home/yuncong/my_csd181/ParthaData/PMD1305_reduce2/region3/'
# img_name_fmt = 'PMD1305_%d.reduce2.region3'
# img_id = 4
# img_name = img_name_fmt%img_id
# img = cv2.imread(IMG_DIR + img_name + '.tif', 0)


mask = foreground_mask(img)
cv2.imwrite(img_name+'_img.jpg', 
            img_as_ubyte(img*mask))
FileLink(img_name+'_img.jpg')
# plt.imshow(img., cmap=plt.cm.Greys_r)
# plt.show()

# <codecell>

def crop_image(img):
    blurred = gaussian_filter(img, 20)
    thresholded = blurred < threshold_otsu(blurred)
    # plot_figure(thresholded, title='', figsize=(10,10), cbar=False, ticks=None, cmap=cm.Greys_r)
    slc = measurements.find_objects(thresholded)[0]

    margin = 100
    xstart = max(slc[0].start - margin, 0)
    xstop = min(slc[0].stop + margin, img.shape[0])
    ystart = max(slc[1].start - margin, 0)
    ystop = min(slc[1].stop + margin, img.shape[1])

    cutout = img[xstart:xstop, ystart:ystop]
    return cutout
    # plot_figure(cutout, title='', figsize=(10,10), cbar=False, ticks=None, cmap=cm.Greys_r)

img = crop_image(img)

# <codecell>

from skimage.filter import gabor_kernel

kernels = []
kernel_freqs = []
kernel_thetas = []

theta_interval = 10

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

%%time
global_pool = Pool()
filtered = np.dstack(global_pool.map(_convolve_per_proc, range(len(kernels))))

# <codecell>

from IPython import parallel
rc = parallel.Client()
dv = rc[:]

# <codecell>

%%time
def _convolve_per_proc(kern):
    from scipy.signal import fftconvolve
    return fftconvolve(img, kern, 'same')

dv['img'] = img
amr = dv.map(_convolve_per_proc, kernels)
filtered = np.dstack(amr.get())
print filtered.shape

# <codecell>

for i in range(36,40):
    plt.imshow(filtered[...,i])
    plt.axis('off')
    plt.show()

# <markdowncell>

# generate directionality map

# <codecell>

f = np.reshape(filtered, (filtered.shape[0], filtered.shape[1], n_freqs, filtered.shape[2]/n_freqs))
dirmap = np.argmax(np.max(f, axis=2), axis=-1)

# colors = [(1,0,0),(0,1,0),(0,0,1),(.5,.5,.0),(0,.5,.5),(.5,0,.5)]

dirmap_rgb = label2rgb(dirmap, image=None, colors=None, alpha=0.3, image_alpha=1)
# cv2.imwrite(img_name+'_dirmap_rgb.jpg', img_as_ubyte(.6*dirmap_rgb + .4*gray2rgb(img/255.)))
dirmap_rgb[~mask] = -1
cv2.imwrite(img_name+'_dirmap_rgb.jpg', img_as_ubyte(dirmap_rgb))
FileLink(img_name+'_dirmap_rgb.jpg')

# <codecell>

np.save('/home/yuncong/my_csd181_scratch/'+img_name + '_dirmap', dirmap)

# <codecell>

a = img_as_ubyte(filtered[:,:,12]/filtered[:,:,12].max())
a[~mask] = 0
cv2.imwrite(img_name+'_a.jpg', a)
FileLink(img_name+'_a.jpg')

# <codecell>

cv2.imwrite(img_name+'_img.jpg', img_as_ubyte(gray2rgb(img/255.)))
FileLink(img_name+'_img.jpg')

# <codecell>

np.unique(dirmap)

# <codecell>

dirmap[23,662]

# <codecell>

f[23,662,:]

# <codecell>

# q = np.sort(filtered, axis=2)

# <codecell>

# strongest_response_id = filtered.argmax(axis=2)

# from skimage.color import label2rgb
# from skimage.util import img_as_ubyte
# img = img[x:-x, y:-y]

# strongest_response_labelmap = label2rgb(strongest_response_id, image=None, colors=None, 
#                                    alpha=0.5,
#                                    image_alpha=0.6)
# cv2.imwrite(img_name+'_strongest_response_labelmap.jpg', img_as_ubyte(strongest_response_labelmap))
# FileLink(img_name+'_strongest_response_labelmap.jpg')

# <codecell>

# unique_ratio = q[...,-1]/q[...,-2]
# plt.matshow(np.log(unique_ratio))

# <codecell>

%%time
r2 = 0.95

energies = filtered.sum(axis=0).sum(axis=0)
order = np.argsort(energies)[::-1]
energies_sorted = energies[order]
r2s = np.cumsum(energies_sorted) / energies_sorted.sum()
k = np.searchsorted(r2s, r2)

selected_kernel_indices = order[:k]

# np.save('/home/yuncong/my_csd181_scratch/'+img_name+'_selected_kernel_indices', selected_kernel_indices)

# selected_kernel_indices = np.load('/home/yuncong/my_csd181_scratch/Resized_6_region_selected_kernel_indices')

filtered_reduced = filtered[..., selected_kernel_indices]

num_feat_reduced = len(selected_kernel_indices)
freqs_reduced = kernel_freqs[selected_kernel_indices]

proportion = 1.5
alpha = 0.25
nonlinear = np.tanh(alpha * filtered_reduced)
# features = nonlinear
sigmas = proportion * (1. / freqs_reduced)
features = np.dstack(gaussian_filter(nonlinear[..., i], sigmas[i]) 
                     for i in range(num_feat_reduced))
# features = np.dstack(gaussian_filter(filtered_reduced[..., i], sigmas[i]) for i in range(num_feat_reduced))
print features.shape

np.save('/home/yuncong/my_csd181_scratch/'+img_name+'_features', features)

# <codecell>

selected_kernel_indices

# <codecell>

plt.matshow(kernels[selected_kernel_indices[i]])

# <codecell>

print gaussian_filter(q, sigmas[0], mode='constant')[-1]
print gaussian_filter(q, sigmas[0], mode='constant')[0]

# <codecell>

from sklearn.cluster import MiniBatchKMeans 
kmeans_model = MiniBatchKMeans(6)
kmeans_model.fit(features.reshape(-1, features.shape[-1]))

# <codecell>

skimage.io.imsave('cutout.tif', cutout)
FileLink('cutout.tif')

# <codecell>

skimage.io.use_plugin('pil')
skimage.io.imsave('labelmap.tif', labeled)
FileLink('labelmap.tif')


# <codecell>

from sklearn.decomposition import PCA
k = 4
pca = PCA(k)
X = features.reshape(-1, n_selected)
X_pca = pca.fit_transform(X)
# kmeans_model.fit(X_pca)

# <codecell>

print X_pca.shape

# <codecell>

def image_grid(images, cols=10, scale=1, suptitle='', titles=None):
    """ display a grid of images
        cols: number of columns = number of images in each row
        scale: 1 to fill screen
    """
    n = len(images)
    H, W = images[0].shape[:2]
    rows = int(ceil((n+0.0)/cols))
    fig = plt.figure(figsize=[scale*20.0/H*W,scale*20.0/cols*rows],dpi=300)
    fig.suptitle(suptitle)
    for i in range(n):
        ax = fig.add_subplot(rows,cols,i+1)
        ax.imshow(images[i], cmap=cm.Greys_r)
        if titles is not None:
            ax.set_title(titles[i])
        ax.axis('off')
    plt.show()

# <codecell>

for theta in range(18):
    for sigma in range(4):
        plt.imshow(kernels[theta*4+sigma])
        plt.show()

# <codecell>

# plt.figure(figsize=(5,5))
# plt.scatter(X_pca[:200,0], X_pca[:200,1])
# plt.axis('equal');

# print pca.components_
for clr, component in enumerate(pca.components_):
    blended = np.zeros((), dtype=np.float)
    plt.bar(selected_kernel_indices, component, color='rbgw'[clr])
    
    blended += kernels[selected_kernel_indices]*
    
    plt.show()

    # print pca.explained_variance_ratio_
# plt.plot(pca.explained_variance_ratio_);

# <codecell>

from sklearn.cluster import MiniBatchKMeans
kmeans_model = MiniBatchKMeans(6, init='kmeans++')
kmeans_model.fit(X_pca)

