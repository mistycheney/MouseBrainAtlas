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

img_dir = '../data/PMD1305_reduce2_region2'
img_name_full = 'PMD1305_reduce2_region2_0001.tif'

img_path = os.path.join(img_dir, img_name_full)
img_name, ext = os.path.splitext(img_name_full)

param_id = 5

class args:
    param_file = '../params/param%d.json'%param_id
    img_file = img_path
    output_dir = '../output'

# <codecell>

def load_array(suffix):
    return manager_utilities.load_array(suffix, img_name, 
                                 params['param_id'], args.output_dir)

def save_array(arr, suffix):
    manager_utilities.save_array(arr, suffix, img_name, 
                                 params['param_id'], args.output_dir)
        
def save_img(img, suffix):
    manager_utilities.save_img(img, suffix, img_name, params['param_id'], 
                               args.output_dir, overwrite=True)

def get_img_filename(suffix, ext='tif'):
    return manager_utilities.get_img_filename(suffix, img_name, params['param_id'], args.output_dir, ext=ext)

# <codecell>

params = json.load(open(args.param_file))

p, ext = os.path.splitext(args.img_file)
img_dir, img_name = os.path.split(p)
img = cv2.imread(args.img_file, 0)
im_height, im_width = img.shape[:2]

print 'read %s' % args.img_file

result_name = img_name + '_param' + str(params['param_id'])
result_dir = os.path.join(args.output_dir, result_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# <codecell>

%%time
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
n_freq = int(np.log(freq_max/freq_min)/np.log(freq_step)) + 1
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

%%time

try:
    features = load_array('features')
except IOError:
    def convolve_per_proc(i):
        return fftconvolve(img, kernels[i], 'same').astype(np.half)

#     pool = Pool(processes=8)

    filtered = Parallel(n_jobs=16)(delayed(convolve_per_proc)(i) 
                            for i in range(n_kernel))
#     filtered = pool.map(convolve_per_proc, range(n_kernel))

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

%%time

print '=== compute rotation-invariant texton map using K-Means ==='

n_texton = params['n_texton']

try: 
#     raise IOError
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
    
#     pool = Pool(processes=16)

    for iteration in range(n_iter):
        
        data = random.sample(X, n_sample)
        
        print 'iteration', iteration
        centroid_all_rotations = np.vstack([np.concatenate(np.roll(np.split(c, n_freq), i)) 
                                for c,i in itertools.product(centroids, range(n_angle))])

        r = Parallel(n_jobs=16)(delayed(compute_dist_per_proc)(x,c) 
                        for x, c in zip(np.array_split(data, n_splits, axis=0), 
                                        itertools.repeat(centroid_all_rotations, n_splits)))
        res = np.vstack(r)        
#         res = np.vstack(pool.map(compute_dist_per_proc, 
#                                  zip(np.array_split(data, n_splits, axis=0), 
#                                      itertools.repeat(centroid_all_rotations, n_splits))))
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
    
#     res = np.vstack(pool.map(compute_dist_per_proc, 
#                              zip(np.array_split(X, n_splits, axis=0), itertools.repeat(centroid_all_rotations, n_splits))))
    labels = res[:,0]
    rotations = res[:,1]

#     pool.close()
#     pool.join()
#     del pool

    textonmap = labels.reshape(features.shape[:2])
    textonmap[~mask] = -1
    
    save_array(textonmap, 'textonmap')
    
    textonmap_rgb = label2rgb(textonmap, image=None, colors=None, alpha=0.3, image_alpha=1)
    save_img(textonmap_rgb, 'textonmap')

# <codecell>

# %%time

# print '=== compute directionality map ==='

# try:
#     dirmap = load_array('dirmap')
# except IOError:
#     f = np.reshape(features, (features.shape[0], features.shape[1], n_freq, n_angle))
#     dir_energy = np.sum(abs(f), axis=2)
#     total_energy = np.mean(dir_energy, axis=-1)
#     dirmap = dir_energy/total_energy[:, :, np.newaxis]
#     dirmap[~mask] = -1
#     save_array(dirmap, 'dirmap')
#     print 'dirmap computed'

# dirmap_rgb = label2rgb(dirmap, image=None, colors=None, alpha=0.3, image_alpha=1)
# save_img(dirmap_rgb, 'dirmap')

# <codecell>

%%time

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
    
    sp_props = regionprops(segmentation+1, intensity_image=img, cache=True)
    
    def foo2(i):
        return sp_props[i].centroid, sp_props[i].area, sp_props[i].mean_intensity

    r = Parallel(n_jobs=16)(delayed(foo2)(i) for i in range(len(sp_props)))
    sp_centroids_areas_meanintensitys = np.array(r)

    sp_centroids = sp_centroids_areas_meanintensitys[:,0]
    sp_areas = sp_centroids_areas_meanintensitys[:,1]
    sp_mean_intensity = sp_centroids_areas_meanintensitys[:,2]

    # sp_areas = np.array([s.area for s in sp_props])
    # sp_mean_intensity = np.array([s.mean_intensity for s in sp_props])

    # sp_centroids = np.array([s.centroid for s in sp_props])
    # sp_wcentroids = np.array([s.weighted_centroid for s in sp_props])
    # sp_centroid_dist = pdist(sp_centroids)
    # sp_centroid_dist_matrix = squareform(sp_centroid_dist)

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

FileLink(get_img_filename('segmentation', 'png')[3:])

# <codecell>

%%time
# superpixels_fg_count = [np.count_nonzero(mask[segmentation==i]) for i in range(n_superpixels)]

def foo(i):
    return np.count_nonzero(mask[segmentation==i])

r = Parallel(n_jobs=16)(delayed(foo)(i) for i in range(n_superpixels))
superpixels_fg_count = np.array(r)
bg_superpixels = np.nonzero((superpixels_fg_count/sp_areas) < 0.3)[0]
print '%d background superpixels'%len(bg_superpixels)

# pool = Pool(16)
# superpixels_fg_count = np.array(pool.map(foo, range(n_superpixels)))
# pool.close()
# pool.join()
# del pool

# <codecell>

%%time

print '=== compute texton and directionality histogram of each superpixel ==='

# sample_interval = 1
# gridy, gridx = np.mgrid[:img.shape[0]:sample_interval, :img.shape[1]:sample_interval]

# all_seg = segmentation[gridy.ravel(), gridx.ravel()]

try:
#     raise IOError
    sp_texton_hist_normalized = load_array('sp_texton_hist_normalized')
except IOError:
#     all_texton = textonmap[gridy.ravel(), gridx.ravel()]

    def bar(i):
        return np.bincount(textonmap[(segmentation == i)&(textonmap != -1)], minlength=n_texton)

    r = Parallel(n_jobs=16)(delayed(bar)(i) for i in range(n_superpixels))
    sp_texton_hist = np.array(r)
    # sp_texton_hist = np.array([np.bincount(textonmap[(segmentation == s)&(textonmap != -1)], minlength=n_texton) 
    #                  for s in range(n_superpixels)])
    sp_texton_hist_normalized = sp_texton_hist.astype(np.float) / sp_texton_hist.sum(axis=1)[:, np.newaxis]
    save_array(sp_texton_hist_normalized, 'sp_texton_hist_normalized')

# <codecell>

%%time

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

# pool = Pool(16)
# r = pool.map(bar2, range(n_superpixels))
# try:
#     sp_dir_hist_normalized = load_array('sp_dir_hist_normalized')
# except IOError:
# sp_dir_hist_normalized = np.empty((n_superpixels, n_angle))
# for i in range(n_superpixels):
#     segment_dir_energies = dir_energy[segmentation == i].astype(np.float_).sum(axis=0)
#     sp_dir_hist_normalized[i,:] = segment_dir_energies/segment_dir_energies.sum()    
# save_array(sp_dir_hist_normalized, 'sp_dir_hist_normalized')

# <codecell>

%%time

def chi2(u,v):
    return np.sum(np.where(u+v!=0, (u-v)**2/(u+v), 0))

print '=== compute significance of each superpixel ==='

overall_texton_hist = np.bincount(textonmap[mask].flat)
overall_texton_hist_normalized = overall_texton_hist.astype(np.float) / overall_texton_hist.sum()

# individual_texton_saliency_score = np.zeros((n_superpixels, ))
# for i, sp_hist in enumerate(sp_texton_hist_normalized):
#     individual_texton_saliency_score[i] = chi2(sp_hist, overall_texton_hist_normalized)

# individual_texton_saliency_score = cdist(sp_texton_hist_normalized, overall_texton_hist_normalized[np.newaxis,:], chi2)
# individual_texton_saliency_score[bg_superpixels] = 0
individual_texton_saliency_score = np.array([chi2(sp_hist, overall_texton_hist_normalized) 
                                             if sp_hist not in bg_superpixels else 0 
                                             for sp_hist in sp_texton_hist_normalized])

# texton_saliency_score = individual_texton_saliency_score

texton_saliency_score = np.zeros((n_superpixels,))
for i, sp_hist in enumerate(sp_texton_hist_normalized):
    if i not in bg_superpixels:
        texton_saliency_score[i] = individual_texton_saliency_score[i]
        
texton_saliency_map = texton_saliency_score[segmentation]

save_img(texton_saliency_map, 'texton_saliencymap')
# Image(get_img_filename('texton_saliencymap', 'png'))

# <codecell>

import h5py
f = h5py.File(os.path.join(args.cache_dir, 'myfile.hdf5'),'w', libver='latest')

# <codecell>

grp = f.create_group('segmentation')

# <codecell>

f['segmentation']['segmentation'][()]

# <codecell>

sp_dir_hist_normalized[120]

# <codecell>

a[120]

# <codecell>

%%time

from skimage.morphology import disk
from skimage.filter.rank import gradient
from scipy.sparse import coo_matrix

edge_map = gradient(segmentation.astype(np.uint8), disk(3))
neighbors = [set() for i in range(n_superpixels)]

def bar3()

r = Parallel(n_jobs=16)(delayed(bar3)(i) for i in range(n_superpixels))
sp_texton_hist = np.array(r)

for y,x in zip(*np.nonzero(edge_map)):
    neighbors[segmentation[y,x]] |= set(segmentation[y-2:y+2,x-2:x+2].ravel())

# connectivity_matrix = np.zeros((n_segmentation, n_segmentation), dtype=np.bool)
rows = np.hstack([s*np.ones((len(neighbors[s]),), dtype=np.int) for s in range(n_superpixels)])
cols = np.hstack([list(neighbors[s]) for s in range(n_superpixels)])
data = np.ones((cols.size, ), dtype=np.bool)
connectivity_matrix = coo_matrix((data, (rows, cols)), shape=(n_superpixels,n_superpixels))
# plt.matshow(connectivity_matrix.todense(), cmap=plt.cm.Greys_r)
# plt.show()

# from skimage.draw import line
# superpixel_connectivity_img = img_superpixelized.copy()
# for i in range(n_superpixels):
#     for neig in neighbors[i]:
#         rr, cc = line(int(sp_centroids[i,0]), int(sp_centroids[i,1]),
#                       int(sp_centroids[neig,0]), int(sp_centroids[neig,1]))
#         superpixel_connectivity_img[rr, cc] = (0,0,1)

# <codecell>

q = sp_dir_hist_normalized.argsort(axis=1)

a = np.zeros((n_superpixels, ))
for i in range(n_superpixels):
    a[i] = sp_dir_hist_normalized[i,q[i,-1]]/sp_dir_hist_normalized[i,q[i,-2]]
# maxdir_rgb = label2rgb(maxdir_map)

# <codecell>

w = [i for i in a.argsort() if i not in bg_superpixels]

# <codecell>

maxdir_map = np.ones_like(segmentation, dtype=np.float)
for i in w:
    maxdir_map[segmentation==i] = a[i]
# maxdir_map[~mask] = 1

# <codecell>

plt.matshow(maxdir_map);
plt.axis('off')
plt.colorbar();

# <codecell>

FileLink(get_img_filename('segmentation', 'png')[3:])

# <codecell>

Image(get_img_filename('textonmap', 'png'))

# <codecell>

plt.matshow(texton_saliency_map);
plt.axis('off')
plt.colorbar();

# <codecell>

D = pdist(sp_texton_hist_normalized)
D = squareform(D)

# <codecell>

bg_superpixels

# <codecell>

sp_texton_hist_normalized[47]

# <codecell>

D[74, 204]

# <codecell>


# <codecell>

sig_sp_sorted = np.argsort(texton_saliency_score)[::-1]

sig_sp_sorted

seed = 135
matched_sp = np.argsort(D[seed])[np.sort(D[seed]) < 0.08]
matched_scores = D[seed][matched_sp]
matched_map = np.ones_like(segmentation, dtype=np.float)
for i, s in zip(matched_sp, matched_scores):
    if i in bg_superpixels: continue
    print i, s
    matched_map[segmentation==i] = s
    
plt.matshow(matched_map);
plt.colorbar();

