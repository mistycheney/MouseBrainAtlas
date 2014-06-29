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

import utilities

from joblib import Parallel, delayed

import glob, re, os, sys, subprocess, argparse, csv

# <codecell>

parser = argparse.ArgumentParser(
formatter_class=argparse.RawDescriptionHelpFormatter,
description='Execute feature extraction pipeline',
epilog="""
The following command processes image PMD1305_region0_reduce2_0244.tif using the parameter setting number 10.
python %s ../data/PMD1305_region0_reduce2/PMD1305_region0_reduce2_0244.tif 10

This script loads the parameters from ../params.csv. 
The meanings of all parameters are explained in GitHub README.

The results are stored in a sub-directory under the output directory. 
The sub-directory is named <dataset name>_reduce<reduce level>_<image index>_param<parameter id>.
The content of this sub-directory are the .npy files or image files with different _<suffix>. See GitHub README for details of these files.

* GitHub README *
https://github.com/mistycheney/BrainSaliencyDetection/blob/master/README.md
"""%(os.path.basename(sys.argv[0]), ))

parser.add_argument("img_file", type=str, help="path to image file")
parser.add_argument("param_id", type=str, help="parameter id")
parser.add_argument("-o", "--output_dir", type=str, help="output directory (default: %(default)s)", default='/oasis/scratch/csd181/yuncong/output')
parser.add_argument("-p", "--params_file", type=str, help="csv file of all parameter settings (default: %(default)s)", default='/oasis/projects/nsf/csd181/yuncong/Brain/params.csv')
args = parser.parse_args()

# <codecell>

def load_array(suffix):
    return utilities.load_array(suffix, img_name, param['param_id'], args.output_dir)

def save_array(arr, suffix):
    utilities.save_array(arr, suffix, img_name, param['param_id'], args.output_dir)
        
def save_img(img, suffix):
    utilities.save_img(img, suffix, img_name, param['param_id'], args.output_dir, overwrite=True)

def get_img_filename(suffix, ext='tif'):
    return utilities.get_img_filename(suffix, img_name, param['param_id'], args.output_dir, ext=ext)

# <codecell>

params_file = os.path.realpath(args.params_file)
parameters = utilities.load_parameters(params_file)
param = parameters[int(args.param_id)]
        
img_file = os.path.realpath(args.img_file)
img_path, ext = os.path.splitext(img_file)
img_dir, img_name = os.path.split(img_path)

img = cv2.imread(img_file, 0)
im_height, im_width = img.shape[:2]
print 'read %s' % img_file

output_dir = os.path.realpath(args.output_dir)

result_name = img_name + '_param' + str(param['param_id'])
result_dir = os.path.join(output_dir, result_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# <codecell>

print '=== finding foreground mask ==='
mask = utilities.foreground_mask(rescale(img, .5**3), min_size=100)
mask = resize(mask, img.shape) > .5

# <codecell>

theta_interval = param['theta_interval']
n_angle = 180/theta_interval
freq_step = param['freq_step']
freq_max = 1./param['min_wavelen']
freq_min = 1./param['max_wavelen']
bandwidth = param['bandwidth']
n_freq = int(np.log(freq_max/freq_min)/np.log(freq_step)) + 1
frequencies = freq_max/freq_step**np.arange(n_freq)

kernels = [gabor_kernel(f, theta=t, bandwidth=bandwidth) for f in frequencies 
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
features = features[max_kern_size/2:-max_kern_size/2, max_kern_size/2:-max_kern_size/2, :]
img = img[max_kern_size/2:-max_kern_size/2, max_kern_size/2:-max_kern_size/2]
mask = mask[max_kern_size/2:-max_kern_size/2, max_kern_size/2:-max_kern_size/2]
im_height, im_width = img.shape[:2]

# <codecell>

print '=== compute rotation-invariant texton map using K-Means ==='

n_texton = param['n_texton']

try: 
    textonmap = load_array('textonmap')
except IOError:
    
    X = features.reshape(-1, n_feature)
    n_data = X.shape[0]
    n_splits = 1000
    n_sample = param['n_sample']
    centroids = np.array(random.sample(X, n_texton))
    
    n_iter = param['n_iter']

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

        counts = np.bincount(labels, minlength=n_texton)
        centroids_new /= counts[:, np.newaxis]
        centroids_new[counts==0] = centroids[counts==0]
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
    segmentation = slic(img_rgb, n_segments=param['n_superpixels'], max_iter=10, 
                        compactness=param['slic_compactness'], 
                        sigma=param['slic_sigma'], enforce_connectivity=True)
    print 'segmentation computed'
    
    save_array(segmentation, 'segmentation')
    
sp_props = regionprops(segmentation+1, intensity_image=img, cache=True)

def foo2(i):
    return sp_props[i].centroid, sp_props[i].area, sp_props[i].mean_intensity

r = Parallel(n_jobs=16)(delayed(foo2)(i) for i in range(len(sp_props)))
sp_centroids = np.array([s[0] for s in r])
sp_areas = np.array([s[1] for s in r])
sp_mean_intensity = np.array([s[2] for s in r])

n_superpixels = len(np.unique(segmentation))

img_superpixelized = mark_boundaries(img_rgb, segmentation)
sptext = img_as_ubyte(img_superpixelized)
for s in range(n_superpixels):
    sptext = cv2.putText(sptext, str(s), 
                      tuple(np.floor(sp_centroids[s][::-1]).astype(np.int)), 
                      cv2.FONT_HERSHEY_COMPLEX_SMALL,
                      .5, ((255,0,255)), 1)
save_img(sptext, 'segmentation')

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

