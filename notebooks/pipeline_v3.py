# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import cv2

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

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
from utilities import chi2

from joblib import Parallel, delayed

import cPickle as pickle

import glob, re, os, sys, subprocess, argparse
import pprint

# <codecell>

# # parse arguments
# parser = argparse.ArgumentParser(
# formatter_class=argparse.RawDescriptionHelpFormatter,
# description='Execute feature extraction pipeline',
# epilog="""
# The following command processes image PMD1305_region0_reduce2_0244.tif using the parameter id nissl324.
# python %s ../ParthaData/PMD1305_region0_reduce2/PMD1305_region0_reduce2_0244.tif nissl324

# This script loads the parameters in params_dir. 
# Results are stored in a sub-directory named <result name>_param_<parameter id>, under output_dir.
# Details are in the GitHub README (https://github.com/mistycheney/BrainSaliencyDetection/blob/master/README.md)
# """%(os.path.basename(sys.argv[0]), ))

# parser.add_argument("img_file", type=str, help="path to image file")
# parser.add_argument("param_id", type=str, help="parameter identification name")
# # parser.add_argument("-o", "--output_dir", type=str, help="output directory (default: %(default)s)", default='/oasis/projects/nsf/csd181/yuncong/DavidData2014v2')
# # parser.add_argument("-p", "--params_dir", type=str, help="directory containing csv parameter files %(default)s)", default='/oasis/projects/nsf/csd181/yuncong/Brain/params')
# args = parser.parse_args()

parser = argparse.ArgumentParser(
formatter_class=argparse.RawDescriptionHelpFormatter,
description='Execute feature processing pipeline',
epilog="""%s
"""%(os.path.basename(sys.argv[0]), ))

parser.add_argument("img_file", type=str, help="path to image file")
parser.add_argument("param_id", type=str, help="parameter identification name")
parser.add_argument("textons_file", type=str, help="pre-computed textons", default=None)
args = parser.parse_args()

data_dir = '/oasis/projects/nsf/csd181/yuncong/DavidData2014v2'
repo_dir = '/oasis/projects/nsf/csd181/yuncong/Brain/'
params_dir = os.path.join(repo_dir, 'params')

# class args:
#     img_file = os.path.join(data_dir, 'RS141', 'x5', '0001', 'RS141_x5_0001.tif')
#     param_id = 'redNissl'

# <codecell>

# load parameter settings
# params_dir = os.path.realpath(params_dir)

param_file = os.path.join(params_dir, 'param_%s.json'%args.param_id)
param_default_file = os.path.join(params_dir, 'param_default.json')
param = json.load(open(param_file, 'r'))
param_default = json.load(open(param_default_file, 'r'))

for k, v in param_default.iteritems():
    if not isinstance(param[k], basestring):
        if np.isnan(param[k]):
            param[k] = v

pprint.pprint(param)

# set image paths
img_file = os.path.realpath(args.img_file)
img_path, ext = os.path.splitext(img_file)
img_dir, img_name = os.path.split(img_path)

stack, resol, slice = img_name.split('_')

img = cv2.imread(img_file, 0)
print img_file
im_height, im_width = img.shape[:2]

# set output paths
# data_dir = os.path.realpath(data_dir)

instance_name = img_name + '_' + str(param['param_id'])

results_dir = os.path.join(data_dir, stack, resol, slice, args.param_id+'_pipelineResults')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
textons_file = os.path.realpath(args.textons_file)

# <codecell>

# def load_array(suffix):
#     return utilities.load_array(suffix, img_name, param['param_id'], args.output_dir)

# def save_array(arr, suffix):
#     utilities.save_array(arr, suffix, img_name, param['param_id'], args.output_dir)
        
# def save_img(img, suffix):
#     utilities.save_img(img, suffix, img_name, param['param_id'], args.output_dir, overwrite=True)

# def get_img_filename(suffix, ext='png'):
#     return utilities.get_img_filename(suffix, img_name, param['param_id'], args.output_dir, ext=ext)

def load_array(suffix):
    return utilities.load_array(suffix, instance_name=instance_name, results_dir=results_dir)

def save_array(arr, suffix):
    utilities.save_array(arr, suffix, instance_name=instance_name, results_dir=results_dir)
        
def save_image(img, suffix):
    utilities.save_image(img, suffix, instance_name=instance_name, results_dir=results_dir, overwrite=True)

def load_image(suffix):
    return utilities.load_array(suffix, instance_name=instance_name, results_dir=results_dir)

    
# def get_img_filename(suffix, ext='png'):
#     return utilities.get_img_filename(suffix, img_name, param['param_id'], args.output_dir, ext=ext)

# <codecell>

# Find foreground mask

print '=== finding foreground mask ==='

try:
    
    mask_fn = os.path.join(img_dir, '_'.join([stack, resol, slice, '_mask.png']))
    mask = cv2.imread(mask_fn, 0) > 0
    print 'loaded mask from', mask_fn
    
#     mask = load_array('uncropMask')

except:

    mask = utilities.foreground_mask(img, min_size=2500)
    mask = mask > .5
    # plt.imshow(mask, cmap=plt.cm.Greys_r);

    save_array(mask, 'uncropMask')

# <codecell>

# Generate Gabor filter kernels

theta_interval = param['theta_interval']
n_angle = int(180/theta_interval)
freq_step = param['freq_step']
freq_max = 1./param['min_wavelen']
freq_min = 1./param['max_wavelen']
bandwidth = param['bandwidth']
n_freq = int(np.log(freq_max/freq_min)/np.log(freq_step)) + 1
frequencies = freq_max/freq_step**np.arange(n_freq)

kernels = [gabor_kernel(f, theta=t, bandwidth=bandwidth) for f in frequencies 
          for t in np.arange(0, n_angle)*np.deg2rad(theta_interval)]
kernels = map(np.real, kernels)

n_kernel = len(kernels)

print '=== filter using Gabor filters ==='
print 'num. of kernels: %d' % (n_kernel)
print 'frequencies:', frequencies
print 'wavelength (pixels):', 1/frequencies

max_kern_size = np.max([kern.shape[0] for kern in kernels])
print 'max kernel matrix size:', max_kern_size

# <codecell>

# Process the image using Gabor filters

try:
#     raise IOError
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

# Crop image border where filters show border effects

features = features[max_kern_size/2:-max_kern_size/2, max_kern_size/2:-max_kern_size/2, :]
img = img[max_kern_size/2:-max_kern_size/2, max_kern_size/2:-max_kern_size/2]
mask = mask[max_kern_size/2:-max_kern_size/2, max_kern_size/2:-max_kern_size/2]
im_height, im_width = img.shape[:2]

save_image(img, 'cropImg')

save_array(mask, 'cropMask')

# <codecell>

# Compute rotation-invariant texton map using K-Means

print '=== compute rotation-invariant texton map using K-Means ==='

n_texton = int(param['n_texton'])

def compute_dist_per_proc(X_partial, c_all_rot):
    D = cdist(X_partial, c_all_rot, 'sqeuclidean')
    ci, ri = np.unravel_index(D.argmin(axis=1), (n_texton, n_angle))
    return np.c_[ci, ri]

try: 
#     raise IOError
    textonmap = load_array('texMap')
except IOError:
    
    X = features.reshape(-1, n_feature)
    n_data = X.shape[0]
    n_splits = 1000
    n_sample = int(param['n_sample'])
    
    try:
#         centroids = load_array('centroids')

        if textons_file is None:
            raise IOError

        centroids = np.load(textons_file)
        print 'loading textons from', textons_file
    
    except IOError:
        
        centroids = np.array(random.sample(X, n_texton))

        n_iter = int(param['n_iter'])

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
            centroids_new /= counts[:, np.newaxis] # denominator might be zero
            centroids_new[counts==0] = centroids[counts==0]
            print np.sqrt(np.sum((centroids - centroids_new)**2, axis=1)).mean()

            centroids = centroids_new

        print centroids.shape
        save_array(centroids, 'centroids')
    
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
    
#     save_array(textonmap, 'texMap')
    save_array(textonmap.astype(np.int16), 'texMap')
    
textonmap_rgb = label2rgb(textonmap, image=None, colors=None, alpha=0.3, image_alpha=1)
save_image(textonmap_rgb, 'texMap')

# <codecell>

# Over-segment the image into superpixels using SLIC (http://ivrg.epfl.ch/research/superpixels)

print '=== over-segment the image into superpixels based on color information ==='

img_rgb = gray2rgb(img)

try:
#     raise IOError
    segmentation = load_array('segmentation')
    
except IOError:
    segmentation = slic(img_rgb, n_segments=int(param['n_superpixels']), 
                        max_iter=10, 
                        compactness=float(param['slic_compactness']), 
                        sigma=float(param['slic_sigma']), 
                        enforce_connectivity=True)
    print 'segmentation computed'
    
    save_array(segmentation.astype(np.int16), 'segmentation')
    
sp_props = regionprops(segmentation+1, intensity_image=img, cache=True)

def obtain_props_worker(i):
    return sp_props[i].centroid, sp_props[i].area, sp_props[i].mean_intensity

r = Parallel(n_jobs=16)(delayed(obtain_props_worker)(i) for i in range(len(sp_props)))
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
save_image(sptext, 'segmentation')

# <codecell>

# Determine which superpixels are mostly background

def count_fg_worker(i):
    return np.count_nonzero(mask[segmentation==i])

r = Parallel(n_jobs=16)(delayed(count_fg_worker)(i) for i in range(n_superpixels))
superpixels_fg_count = np.array(r)
bg_superpixels = np.nonzero((superpixels_fg_count/sp_areas) < 0.3)[0]
# bg_superpixels = np.array(list(set(bg_superpixels.tolist()
#                           +[0,1,2,3,4,5,6,7,119,78,135,82,89,187,174,242,289]
#                           +[50,51,56,57,58,59,60,61,62,63,64,65,115,73,88,109,99,91,122,110,151,192,165,158,254,207,236,306]
#                           )))
fg_superpixels = np.array([i for i in range(n_superpixels) if i not in bg_superpixels])
print '%d background superpixels'%len(bg_superpixels)

save_array(fg_superpixels, 'fg')
save_array(bg_superpixels, 'bg')

# a = np.zeros((n_superpixels,), dtype=np.bool)
# a[fg_superpixels] = True
# plt.imshow(a[segmentation], cmap=plt.cm.Greys_r)
# plt.show()

# <codecell>

# Compute neighbor lists and connectivity matrix

from skimage.morphology import disk
from skimage.filter.rank import gradient
from scipy.sparse import coo_matrix

edge_map = gradient(segmentation.astype(np.uint8), disk(3))
neighbors = [set() for i in range(n_superpixels)]

for y,x in zip(*np.nonzero(edge_map)):
    neighbors[segmentation[y,x]] |= set(segmentation[y-2:y+2,x-2:x+2].ravel())

for i in range(n_superpixels):
    neighbors[i] -= set([i])
    
rows = np.hstack([s*np.ones((len(neighbors[s]),), dtype=np.int) for s in range(n_superpixels)])
cols = np.hstack([list(neighbors[s]) for s in range(n_superpixels)])
data = np.ones((cols.size, ), dtype=np.bool)
connectivity_matrix = coo_matrix((data, (rows, cols)), shape=(n_superpixels,n_superpixels))
connectivity_matrix = connectivity_matrix.transpose() * connectivity_matrix

save_array(neighbors, 'neighbors')

# <codecell>

# compute texton histogram of every superpixel
print '=== compute texton histogram of each superpixel ==='

try:
#     raise IOError
    sp_texton_hist_normalized = load_array('texHist')
except IOError:
    def texton_histogram_worker(i):
        return np.bincount(textonmap[(segmentation == i)&(textonmap != -1)], minlength=n_texton)

    r = Parallel(n_jobs=16)(delayed(texton_histogram_worker)(i) for i in range(n_superpixels))
    sp_texton_hist = np.array(r)
    sp_texton_hist_normalized = sp_texton_hist.astype(np.float) / sp_texton_hist.sum(axis=1)[:, np.newaxis] # denom might be invalid
    save_array(sp_texton_hist_normalized, 'texHist')

# compute the null texton histogram
overall_texton_hist = np.bincount(textonmap[mask].flat)
overall_texton_hist_normalized = overall_texton_hist.astype(np.float) / overall_texton_hist.sum()

# <codecell>

# compute directionality histogram of every superpixel
print '=== compute directionality histogram of each superpixel ==='

try:
#     raise IOError
    sp_dir_hist_normalized = load_array('dirHist')
except IOError:
    f = np.reshape(features, (features.shape[0], features.shape[1], n_freq, n_angle))
    dir_energy = np.sum(abs(f), axis=2)

    def dir_histogram_worker(i):
        segment_dir_energies = dir_energy[segmentation == i].astype(np.float_).mean(axis=0)
        return segment_dir_energies    

    r = Parallel(n_jobs=16)(delayed(dir_histogram_worker)(i) for i in range(n_superpixels))
    
    sp_dir_hist = np.vstack(r)
    sp_dir_hist_normalized = sp_dir_hist/sp_dir_hist.sum(axis=1)[:,np.newaxis]
    save_array(sp_dir_hist_normalized, 'dirHist')

# compute the null directionality histogram
overall_dir_hist = sp_dir_hist_normalized[fg_superpixels].mean(axis=0)
overall_dir_hist_normalized = overall_dir_hist.astype(np.float) / overall_dir_hist.sum()

