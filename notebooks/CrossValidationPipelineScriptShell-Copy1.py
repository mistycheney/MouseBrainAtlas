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

import glob, re, os, sys, subprocess, argparse

# <codecell>

def save_array(arr, suffix):
    arr_file = os.path.join(args.cache_dir,
                        '%s_param%d_%s.npy'%(img_name, params['param_id'],
                                             suffix))
    if not os.path.exists(arr_file):
        np.save(arr_file, arr)
        print '%s saved to %s' % (suffix, arr_file)
    else:
        print '%s already exists' % (arr_file)
        
def save_img(img, suffix, is_ubyte=True):
    '''
    img is in uint8 type or float type
    '''
    img_fn = get_img_filename(suffix)
    if not os.path.exists(img_fn):
        if is_ubyte:        
            cv2.imwrite(img_fn, img)
        else:
            plt.matshow(saliency_map, cmap=cm.Greys_r)
            plt.colorbar()
            plt.savefig(img_fn, bbox_inches='tight')
            plt.close();
        print '%s saved to %s' % (suffix, img_fn)
    else:
        print '%s already exists' % (img_fn)
        
    return img_fn

def get_img_filename(suffix):
    img_fn = os.path.join(args.cache_dir,
                '%s_param%d_%s.png'%(img_name, params['param_id'], suffix))
    return img_fn

# <codecell>

# parser = argparse.ArgumentParser()
# parser.add_argument("param_file", type=str, help="parameter file name")
# parser.add_argument("img_file", type=str, help="path to image file")
# parser.add_argument("-of", "--output_feature", action='store_true', help="whether to output feature array")
# parser.add_argument("-ot", "--output_textonmap", action='store_true', help="whether to output textonmap file")
# parser.add_argument("-od", "--output_dirmap", action='store_true', help="whether to output dirmap file")
# parser.add_argument("-os", "--output_segmentation", action='store_true', help="whether to output superpixel segmentation file")
# parser.add_argument("-c", "--cache_dir", default='scratch', help="directory to store outputs")
# args = parser.parse_args()

cache_dir = 'scratch'
# img_dir = '~/ParthaData/PMD1305_reduce0_region0/'
img_dir = '/home/yuncong/ParthaData/PMD1305_reduce2/region1'
# img_name = 'PMD1305_244_reduce0_region0'
img_name = 'PMD1305_244.reduce2.region1'
img_path = os.path.join(img_dir, img_name+'.tif')

param_id = 3285
param_file = 'params/param%d.json'%param_id

class args:
    param_file = param_file
    img_file = img_path
    output_feature = True
    output_textonmap = True
    output_dirmap = True
    output_segmentation = True
    cache_dir = 'scratch'

# %run CrossValidationPipelineScriptShell.py -of -ot -od -os {param_file} {img_path}

# <codecell>

params = json.load(open(args.param_file))
p, ext = os.path.splitext(args.img_file)
img_dir, img_name = os.path.split(p)

# <codecell>

img_rgb = gray2rgb(img)

sp_file = os.path.join(args.cache_dir,
                        '%s_param%d_segmentation.npy'%(img_name, params['param_id']))
if os.path.exists(sp_file):
    segmentation = np.load(sp_file)
    print 'load superpixel segmentation from %s' % sp_file
    
else:
    segmentation = slic(img_rgb, n_segments=params['n_superpixels'], max_iter=10, 
                        compactness=params['slic_compactness'], 
                        sigma=params['slic_sigma'], enforce_connectivity=True)
    print 'segmentation computed'
    
    if args.output_segmentation:
        np.save(sp_file, segmentation)
        print 'segmentation saved to %s' % sp_file

        
n_superpixels = len(np.unique(segmentation))

sp_props = regionprops(segmentation+1, intensity_image=img, cache=True)
sp_centroids = np.array([s.centroid for s in sp_props])
sp_areas = np.array([s.area for s in sp_props])
# sp_wcentroids = np.array([s.weighted_centroid for s in sp_props])
sp_centroid_dist = pdist(sp_centroids)
sp_centroid_dist_matrix = squareform(sp_centroid_dist)
sp_mean_intensity = np.array([s.mean_intensity for s in sp_props])

img_superpixelized = mark_boundaries(img_rgb, segmentation)
sp_img = img_as_ubyte(img_superpixelized)
# sptext = img_as_ubyte(img_superpixelized)
# for s in range(n_superpixels):
#     sptext = cv2.putText(sptext, str(s), 
#                       tuple(np.floor(sp_centroids[s][::-1]).astype(np.int)), 
#                       cv2.FONT_HERSHEY_COMPLEX_SMALL,
#                       .5, ((255,0,255)), 1)

# save_img(sptext, 'sptext')
# Image(get_img_filename('sptext'))

save_img(sp_img, 'sp')
Image(get_img_filename('sp'))

# <codecell>

superpixels_bg_count = np.array([(~mask[segmentation==i]).sum() for i in range(n_superpixels)])
bg_superpixels = np.nonzero((superpixels_bg_count/sp_areas) > 0.7)[0]
print len(bg_superpixels), 'background superpixels'

# <codecell>

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

img = cv2.imread(os.path.join(args.img_file), 0)
im_height, im_width = img.shape[:2]

feature_file = os.path.join(args.cache_dir, 
                            '%s_param%d.npy'%(img_name, 
                                              params['param_id']))

def convolve_per_proc(i):
    return fftconvolve(img, kernels[i], 'same').astype(np.half)

if os.path.exists(feature_file):
    features = np.load(feature_file)
    n_feature = features.shape[-1]
    print 'load features from %s' % feature_file
else:
    pool = Pool(processes=8)
    filtered = pool.map(convolve_per_proc, range(n_kernel))

    features = np.empty((im_height, im_width, n_kernel), dtype=np.half)
    for i in range(n_kernel):
        features[...,i] = filtered[i]
    
    n_feature = features.shape[-1]
    
    if args.output_feature:
        np.save(feature_file, features)
        print 'features saved to %s' % feature_file

# <codecell>

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
    return np.c_[ci, ri]

n_texton = params['n_texton']

textonmap_file = os.path.join(args.cache_dir,
                        '%s_param%d_textonmap.npy'%(img_name, params['param_id']))

if os.path.exists(textonmap_file):
    textonmap = np.load(textonmap_file)
    print 'load textonmap from %s' % textonmap_file
else:

    X = features.reshape(-1, n_feature)
    n_data = X.shape[0]

    n_splits = 1000
    n_sample = 10000
    data = random.sample(X, n_sample)
    centroids = data[:n_texton]
    
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

    textonmap = labels.reshape(features.shape[:2])
    textonmap[~mask] = -1
    
    if args.output_textonmap:
        np.save(textonmap_file, textonmap)
        print 'textonmap saved to %s' % textonmap_file

    textonmap_rgb = label2rgb(textonmap, image=None, colors=None, alpha=0.3, image_alpha=1)
    save_img(textonmap_rgb, 'textonmap')    

Image(get_img_filename('textonmap'))

# <codecell>

dirmap_file = os.path.join(args.cache_dir,
                        '%s_param%d_dirmap.npy'%(img_name, params['param_id']))

if os.path.exists(dirmap_file):
    textonmap = np.load(textonmap_file)
    print 'load textonmap from %s' % textonmap_file
else:
    f = np.reshape(features, (features.shape[0], features.shape[1], n_freq, n_angle))
    dirmap = np.argmax(np.max(f, axis=2), axis=-1)
    dirmap[~mask] = -1
    print 'dirmap computed'

    if args.output_dirmap:
        save_array(dirmap, 'dirmap')

# colors = [(1,0,0),(0,1,0),(0,0,1),(.5,.5,.0),(0,.5,.5),(.5,0,.5)]

    dirmap_rgb = label2rgb(dirmap, image=None, colors=None, alpha=0.3, image_alpha=1)
    # cv2.imwrite(img_name+'_dirmap_rgb.jpg', img_as_ubyte(.6*dirmap_rgb + .4*gray2rgb(img/255.)))

    save_img(dirmap_rgb, 'dirmap')
    
Image(get_img_filename('dirmap'))

# <codecell>

sample_interval = 1
gridy, gridx = np.mgrid[:img.shape[0]:sample_interval, :img.shape[1]:sample_interval]

all_seg = segmentation[gridy.ravel(), gridx.ravel()]
all_texton = textonmap[gridy.ravel(), gridx.ravel()]
sp_texton_hist = np.array([np.bincount(all_texton[(all_seg == s)&(all_texton != -1)], minlength=n_texton) 
                 for s in range(n_superpixels)])

row_sums = sp_texton_hist.sum(axis=1)
sp_texton_hist_normalized = sp_texton_hist.astype(np.float) / row_sums[:, np.newaxis]

def chi2(u,v):
    return np.sum(np.where(u+v!=0, (u-v)**2/(u+v), 0))

eps = 0.001
def kl(u,v):
    return np.sum((u+eps)*np.log((u+eps)/(v+eps)))

def kl_no_eps(u,v):
    return np.sum(u*np.log(u/v))

# D = pdist(sp_texton_hist_normalized, chi2)
# hist_distance_matrix = squareform(D)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(hist_distance_matrix)
# fig.colorbar(cax)
# plt.show()

# if args.output_hist:
#     np.save('')

# <codecell>

overall_texton_hist = np.bincount(textonmap[mask].flat)
overall_texton_hist_normalized = overall_texton_hist.astype(np.float) / overall_texton_hist.sum()

individual_saliency_score = np.array([chi2(sp_hist, overall_texton_hist_normalized) for sp_hist in sp_texton_hist_normalized])

# sp_diameters = np.array([s.equivalent_diameter for s in sp_props])
# sp_diameter_mean = sp_diameters.mean()

saliency_score = np.zeros((n_superpixels,))
for i, sp_hist in enumerate(sp_texton_hist_normalized):
    if i not in bg_superpixels:
        saliency_score[i] = individual_saliency_score[i]
#         neighbor_term = 0
#         c = 0
#         for j in neighbors[i]:
#             if j!=i and j not in bg_superpixels:
#                 neighbor_term += np.exp(-hist_distance_matrix[i,j]) * individual_saliency_score[j]
#                 c += 1
#         saliency_score[i] += neighbor_term/c
        
saliency_map = saliency_score[segmentation]

save_img(saliency_map, 'saliencymap', is_ubyte=False)
Image(get_img_filename('saliencymap'))

