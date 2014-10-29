# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

# <codecell>

import numpy as np
import cv2

import matplotlib.pyplot as plt

import random, itertools, sys, os
from multiprocessing import Pool
import json

from utilities import *
from joblib import Parallel, delayed

import glob, re, os, sys, subprocess, argparse
import pprint
import cPickle as pickle

from skimage.color import hsv2rgb, label2rgb, gray2rgb

# <codecell>

# parser = argparse.ArgumentParser(
# formatter_class=argparse.RawDescriptionHelpFormatter,
# description='Generate textons from a set of filtered images',
# epilog="""%s
# """%(os.path.basename(sys.argv[0]), ))

# parser.add_argument("stack_name", type=str, help="stack name")
# parser.add_argument("resolution", type=str, help="resolution string")
# parser.add_argument("begin_slice", type=str, help="slice number to begin, zero-padding to 4 digits")
# parser.add_argument("end_slice", type=str, help="slice number to end, zero-padding to 4 digits")
# parser.add_argument("param_id", type=str, help="parameter identification name")
# args = parser.parse_args()


class args:
    stack_name = 'RS141'
    resolution = 'x5'
    begin_slice = 1
    end_slice = 5
    param_id = 'redNissl'

    
instance = Instance(args.stack_name, args.resolution, paramset=args.param_id)

# <codecell>

features_fullstack = []

for slice in range(args.begin_slice, args.end_slice + 1):
    instance.set_slice(slice)
    
    features = instance.load_pipeline_result('features', 'npy')
    features_fullstack.append(features)
    
    n_feature = features.shape[-1]
    print n_feature
    
compute_textons(features_fullstack)

# <codecell>

def compute_textons(features):
    """
    Compute rotation-invariant texton map using K-Means
    """

    print '=== compute rotation-invariant texton map using K-Means ==='

    n_texton = int(param['n_texton'])

    def compute_dist_per_proc(X_partial, c_all_rot):
        D = cdist(X_partial, c_all_rot, 'sqeuclidean')
        ci, ri = np.unravel_index(D.argmin(axis=1), (n_texton, n_angle))
        return np.c_[ci, ri]

    X = features.reshape(-1, n_feature)
    n_data = X.shape[0]
    n_splits = 1000
    n_sample = int(param['n_sample'])

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
#     save_array(centroids, 'centroids')

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

# #     save_array(textonmap, 'texMap')
#     save_array(textonmap.astype(np.int16), 'texMap')

    textonmap_rgb = label2rgb(textonmap, image=None, colors=None, alpha=0.3, image_alpha=1)
    save_image(textonmap_rgb, 'texMap')

