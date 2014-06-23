# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import cv2
import matplotlib.pyplot as plt

import random, itertools, sys, os
from multiprocessing import Pool
import json

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
import manager_utilities

from joblib import Parallel, delayed

import glob, re, os, sys, subprocess, argparse

# <codecell>

parser = argparse.ArgumentParser()
parser.add_argument("param_file", type=str, help="parameter file name")
parser.add_argument("img_file", type=str, help="path to image file")
parser.add_argument("-c", "--cache_dir", default='scratch', help="directory to store outputs")
args = parser.parse_args()

# <codecell>

def load_array(suffix):
    return manager_utilities.load_array(suffix, img_name, 
                                 params['param_id'], args.cache_dir)

def save_array(arr, suffix):
    manager_utilities.save_array(arr, suffix, img_name, 
                                 params['param_id'], args.cache_dir)
        
def save_img(img, suffix):
    manager_utilities.save_img(img, suffix, img_name, params['param_id'], 
                               args.cache_dir, overwrite=True)

def get_img_filename(suffix, ext='tif'):
    return manager_utilities.get_img_filename(suffix, img_name, params['param_id'], args.cache_dir, ext=ext)

# <codecell>

params = json.load(open(args.param_file))
p, ext = os.path.splitext(args.img_file)
img_dir, img_name = os.path.split(p)
img = cv2.imread(os.path.join(args.img_file), 0)
im_height, im_width = img.shape[:2]

output_dir = os.path.join(args.cache_dir, img_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print '=== finding foreground mask ==='
mask = foreground_mask(rescale(img, .5**3), min_size=100)
mask = resize(mask, img.shape) > .5

# <codecell>

theta_interval = params['theta_interval'] #10
n_angle = 180/theta_interval
# n_freq = params['n_freq']
freq_step = params['freq_step']
freq_max = 1./params['min_wavelen'] #1./5.
freq_min = 1./params['max_wavelen']
n_freq = int(np.log(freq_max/freq_min)/np.log(freq_step)) +1
frequencies = freq_max/freq_step**np.arange(n_freq)

kernels = [gabor_kernel(f, theta=t, bandwidth=1.) for f in frequencies 
          for t in np.arange(0, np.pi, np.deg2rad(theta_interval))]
kernels = map(np.real, kernels)
n_kernel = len(kernels)

print '=== filter using Gabor filters ==='
print 'num. of kernels: %d' % (n_kernel)
print 'frequencies:', frequencies
print 'wavelength (pixels):', 1/frequencies

max_kern_size = np.max([kern.shape[0] for kern in kernels])
print 'max kernel matrix size:', max_kern_size

# <codecell>

try:
    features = load_array('features')
except IOError:
    def convolve_per_proc(i):
        return fftconvolve(img, kernels[i], 'same').astype(np.half)

    filtered = Parallel(n_jobs=16)(delayed(convolve_per_proc)(i) 
                            for i in range(n_kernel))
    features = np.empty((im_height, im_width, n_kernel), dtype=np.half)
    for i in range(n_kernel):
        features[...,i] = filtered[i]

    del filtered
    
    save_array(features, 'features')

n_feature = features.shape[-1]

# <codecell>

print 'crop border where filters show border effects'
features = features[max_kern_size:-max_kern_size, max_kern_size:-max_kern_size, :]
img = img[max_kern_size:-max_kern_size, max_kern_size:-max_kern_size]
mask = mask[max_kern_size:-max_kern_size, max_kern_size:-max_kern_size]
im_height, im_width = img.shape[:2]

# <codecell>

print '=== compute rotation-invariant texton map using K-Means ==='

n_texton = params['n_texton']

try: 
    textonmap = load_array('textonmap')
except IOError:
    
    X = features.reshape(-1, n_feature)
    n_data = X.shape[0]
    n_splits = 1000
    n_sample = 10000
    centroids = random.sample(X, n_texton)
    
    n_iter = 5

    def compute_dist_per_proc(X_partial, c_all_rot):
        D = cdist(X_partial, c_all_rot, 'sqeuclidean')
        ci, ri = np.unravel_index(D.argmin(axis=1), (n_texton, n_angle))
        return np.c_[ci, ri]
    
    for iteration in range(n_iter):
        
        data = random.sample(X, n_sample)
        
        print 'iteration', iteration
        centroid_all_rotations = np.vstack([np.concatenate(np.roll(np.split(c, n_freq), i)) 
                                for c,i in itertools.product(centroids, range(n_angle))])

        r = Parallel(n_jobs=16)(delayed(compute_dist_per_proc)(x,c) 
                        for x, c in zip(np.array_split(data, n_splits, axis=0), 
                                        itertools.repeat(centroid_all_rotations, n_splits)))
        res = np.vstack(r)        

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

    r = Parallel(n_jobs=16)(delayed(compute_dist_per_proc)(x,c) 
                            for x, c in zip(np.array_split(X, n_splits, axis=0), 
                                            itertools.repeat(centroid_all_rotations, n_splits)))
    res = np.vstack(r)
    
    labels = res[:,0]
    rotations = res[:,1]

    textonmap = labels.reshape(features.shape[:2])
    textonmap[~mask] = -1
    
    save_array(textonmap, 'textonmap')
    
textonmap_rgb = label2rgb(textonmap, image=None, colors=None, alpha=0.3, image_alpha=1)
save_img(textonmap_rgb, 'textonmap')

# <codecell>

print '=== over-segment the image into superpixels based on color information ==='

img_rgb = gray2rgb(img)

try:
    segmentation = load_array('segmentation')
    
except IOError:
    segmentation = slic(img_rgb, n_segments=params['n_superpixels'], max_iter=10, 
                        compactness=params['slic_compactness'], 
                        sigma=params['slic_sigma'], enforce_connectivity=True)
    print 'segmentation computed'
    
    save_array(segmentation, 'segmentation')

n_superpixels = len(np.unique(segmentation))


img_superpixelized = mark_boundaries(img_rgb, segmentation)
sptext = img_as_ubyte(img_superpixelized)
# for s in range(n_superpixels):
#     sptext = cv2.putText(sptext, str(s), 
#                       tuple(np.floor(sp_centroids[s][::-1]).astype(np.int)), 
#                       cv2.FONT_HERSHEY_COMPLEX_SMALL,
#                       .5, ((255,0,255)), 1)
save_img(sptext, 'segmentation')

# <codecell>

sp_props = regionprops(segmentation+1, intensity_image=img, cache=True)
    
def foo2(i):
    return sp_props[i].centroid, sp_props[i].area, sp_props[i].mean_intensity

r = Parallel(n_jobs=16)(delayed(foo2)(i) for i in range(len(sp_props)))
sp_centroids_areas_meanintensitys = np.array(r)

sp_centroids = sp_centroids_areas_meanintensitys[:,0]
sp_areas = sp_centroids_areas_meanintensitys[:,1]
sp_mean_intensity = sp_centroids_areas_meanintensitys[:,2]

# <codecell>

def foo(i):
    return np.count_nonzero(mask[segmentation==i])

r = Parallel(n_jobs=16)(delayed(foo)(i) for i in range(n_superpixels))
superpixels_fg_count = np.array(r)
bg_superpixels = np.nonzero((superpixels_fg_count/sp_areas) < 0.3)[0]
print '%d background superpixels'%len(bg_superpixels)

# <codecell>

print '=== compute texton and directionality histogram of each superpixel ==='

try:
    raise IOError
    sp_texton_hist_normalized = load_array('sp_texton_hist_normalized')
except IOError:
    def bar(i):
        return np.bincount(textonmap[(segmentation == i)&(textonmap != -1)], minlength=n_texton)

    r = Parallel(n_jobs=16)(delayed(bar)(i) for i in range(n_superpixels))
    sp_texton_hist = np.array(r)
    sp_texton_hist_normalized = sp_texton_hist.astype(np.float) / sp_texton_hist.sum(axis=1)[:, np.newaxis]
    save_array(sp_texton_hist_normalized, 'sp_texton_hist_normalized')

# <codecell>

try:
    sp_dir_hist_normalized = load_array('sp_dir_hist_normalized')
except IOError:
    f = np.reshape(features, (features.shape[0], features.shape[1], n_freq, n_angle))
    dir_energy = np.sum(abs(f), axis=2)

    def bar2(i):
        segment_dir_energies = dir_energy[segmentation == i].astype(np.float_).sum(axis=0)
        return segment_dir_energies/segment_dir_energies.sum()    

    r = Parallel(n_jobs=16)(delayed(bar2)(i) for i in range(n_superpixels))
    sp_dir_hist_normalized = np.vstack(r)
    save_array(sp_dir_hist_normalized, 'sp_dir_hist_normalized')

# <codecell>

def chi2(u,v):
    return np.sum(np.where(u+v!=0, (u-v)**2/(u+v), 0))

print '=== compute significance of each superpixel ==='

overall_texton_hist = np.bincount(textonmap[mask].flat)
overall_texton_hist_normalized = overall_texton_hist.astype(np.float) / overall_texton_hist.sum()
individual_texton_saliency_score = np.array([chi2(sp_hist, overall_texton_hist_normalized) 
                                             if sp_hist not in bg_superpixels else 0 
                                             for sp_hist in sp_texton_hist_normalized])

texton_saliency_score = np.zeros((n_superpixels,))
for i, sp_hist in enumerate(sp_texton_hist_normalized):
    if i not in bg_superpixels:
        texton_saliency_score[i] = individual_texton_saliency_score[i]
        
texton_saliency_map = texton_saliency_score[segmentation]

save_img(texton_saliency_map, 'texton_saliencymap')

