# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import cv2
import matplotlib.pyplot as plt

import random, itertools, sys, os
from multiprocessing import Pool
import json

# from sklearn.cluster import MiniBatchKMeans 

from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops
from skimage.util import img_as_ubyte
from skimage.color import hsv2rgb, label2rgb, gray2rgb
from skimage.morphology import disk
from skimage.filter.rank import gradient
from skimage.filter import gabor_kernel
from skimage.transform import rescale, resize

from scipy.ndimage import gaussian_filter, measurements
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, squareform, euclidean, cdist
from scipy.signal import fftconvolve

from IPython.display import FileLink, Image, FileLinks

from utilities import *

# <codecell>

import glob, re, os, sys, subprocess, argparse

parser = argparse.ArgumentParser()
parser.add_argument("param_file", type=str, help="parameter file name")
parser.add_argument("-f", "--output_feature", type=bool, action='store_true', help="whether to output feature array")
parser.add_argument("-t", "--output_textonmap", type=bool, action='store_true', help="whether to output textonmap")
args = parser.parse_args()

# <codecell>


# params = json.load(open(args.param_file))

CACHE_DIR = 'scratch'
IMG_DIR = '/home/yuncong/ParthaData/PMD1305_reduce0_region0/'
img_name_fmt = 'PMD1305_%d_reduce0_region0'
img_id = 244
img_name = img_name_fmt%img_id

params = {
'param_id': 3285,
'theta_interval': 10,
'n_freq': 4,
'max_freq': 0.2,
'n_texton': 20,
'cache_dir': CACHE_DIR,
'img_dir': IMG_DIR
}

theta_interval = params['theta_interval'] #10
n_angle = 180/theta_interval
n_freq = params['n_freq']
freq_max = params['max_freq'] #1./5.
frequencies = freq_max/2**np.arange(n_freq)

kernels = [gabor_kernel(f, theta=t, bandwidth=1.) for f in frequencies 
          for t in np.arange(0, np.pi, np.deg2rad(theta_interval))]
kernels = map(np.real, kernels)
n_kernel = len(kernels)

print 'num. of kernels: %d' % (n_kernel)
print 'frequencies:', frequencies
print 'wavelength (pixels):', 1/frequencies

max_kern_size = np.max([kern.shape[0] for kern in kernels])
print 'max kernel matrix size:', max_kern_size

# <codecell>

import json
json.dump(params, open('params/params%d.json'%params['param_id'], 'w'))

# <codecell>

%%time

img = cv2.imread(os.path.join(params['img_dir'], img_name + '.tif'), 0)
im_height, im_width = img.shape[:2]

feature_file = os.path.join(params['cache_dir'], 
                            '%s_param%d.npy'%(img_name, 
                                              params['param_id']))

def convolve_per_proc(i):
    return fftconvolve(img, kernels[i], 'same').astype(np.half)

if os.path.exists(feature_file):
    filtered = np.load(feature_file)
    print 'load features from %s' % feature_file
else:
    pool = Pool(processes=8)
    filtered = pool.map(convolve_per_proc, range(n_kernel))

    features = np.empty((im_height, im_width, n_kernel), dtype=np.half)
    for i in range(n_kernel):
        features[...,i] = filtered[i]
    
    n_feature = features.shape[-1]
    
#     if args.output_feature:
    if True:
        np.save(feature_file, features)
        print 'features saved to %s' % feature_file

# <codecell>

%%time

print 'finding foreground mask'
mask = foreground_mask(rescale(img, .5**3), min_size=100)
mask = resize(mask, img.shape) > .5

features = features[max_kern_size:-max_kern_size, max_kern_size:-max_kern_size, :]
img = img[max_kern_size:-max_kern_size, max_kern_size:-max_kern_size]
mask = mask[max_kern_size:-max_kern_size, max_kern_size:-max_kern_size]

# <codecell>

def compute_dist_per_proc((X_partial, c_all_rot)):
    D = cdist(X_partial, c_all_rot, 'sqeuclidean')
    ci, ri = np.unravel_index(D.argmin(axis=1), (n_texton, n_angle))
    return np.column_stack((ci, ri))

# <codecell>

%%prun

n_texton = params['n_texton']
# X = features[mask, :]
X = features.reshape(-1, n_feature)
n_data = X.shape[0]

n_splits = 1000
n_sample = 10000
data = random.sample(X, n_sample)
centroids = data[:n_texton]

# <codecell>

%%time

n_iter = 5
pool = Pool(processes=16)

for iteration in range(n_iter):
    print 'iteration', iteration
    centroid_all_rotations = np.vstack([np.concatenate(np.roll(np.split(c, n_freq), i)) 
                            for c,i in itertools.product(centroids, range(n_angle))])


    pool = Pool(processes=16)
    res = np.vstack(pool.map(compute_dist_per_proc, 
                             zip(np.array_split(data, n_splits, axis=0), 
                                 itertools.repeat(centroid_all_rotations, n_splits))))
    
    
    labels = res[:,0]
    rotations = res[:,1]

    centroids_new = np.zeros((n_texton, n_feature))
    for d, l, r in itertools.izip(data, labels, rotations):
        rot = np.concatenate(np.roll(np.split(d, n_freq), i))
        centroids_new[l] += rot

    counts = np.bincount(labels)
    centroids_new /= counts[:, np.newaxis]
    print np.sqrt(np.sum((centroids - centroids_new)**2, axis=1)).mean()

    centroids = centroids_new

print 'kmeans completes'
centroid_all_rotations = np.vstack([np.concatenate(np.roll(np.split(c, n_freq), i)) 
                            for c,i in itertools.product(centroids, range(n_angle))])

res = np.vstack(pool.map(compute_dist_per_proc, 
                         zip(np.array_split(X, n_splits, axis=0), itertools.repeat(centroid_all_rotations, n_splits))))
labels = res[:,0]
rotations = res[:,1]

pool.close()
pool.join()
del pool

textonmap = -1*np.ones((features.shape[:2]))
textonmap = labels.reshape(features.shape[:2])
textonmap[~mask] = -1

# <codecell>

# if args.output_textonmap:

textonmap_rgb = label2rgb(textonmap, image=None, colors=None, alpha=0.3, image_alpha=1)
textonmap_file = os.path.join(params['cache_dir'], 
                            '%s_param%d_textonmap.png'%(img_name, params['param_id']))
cv2.imwrite(textonmap_file, img_as_ubyte(textonmap_rgb))
FileLink(textonmap_file)

