# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# %load_ext autoreload
# %autoreload 2

import sigboost
import numpy as np
import cv2
import argparse, os, json, pprint
import random
import itertools
from skimage.filter import gabor_kernel
from utilities import *

# from multiprocessing import Pool
# from pathos.multiprocessing import ProcessingPool as Pool
from joblib import Parallel, delayed

from scipy.signal import fftconvolve
from scipy.spatial.distance import pdist, squareform, euclidean, cdist


def load_stuff(args):
    params_dir = os.path.realpath(args.params_dir)
    param_file = os.path.join(params_dir, 'param_%s.json'%args.param_id)
    param_default_file = os.path.join(params_dir, 'param_default.json')
    param = json.load(open(param_file, 'r'))
    param_default = json.load(open(param_default_file, 'r'))

    for k, v in param_default.iteritems():
        if not isinstance(param[k], basestring):
            if np.isnan(param[k]):
                param[k] = v

    img_file = os.path.realpath(args.img_file)
    img_path, ext = os.path.splitext(img_file)
    img_dir, img_name = os.path.split(img_path)

    print img_file
    img = cv2.imread(img_file, 0)
    im_height, im_width = img.shape[:2]
    
    output_dir = os.path.realpath(args.output_dir)

    result_name = img_name + '_param_' + str(param['param_id'])
    result_dir = os.path.join(output_dir, result_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    return img, param, result_name, output_dir


class FeatureExtractor(object):

    def __init__(self, img, param):
        self.img = img
        self.param = param
        
    def get_kernels(self):

        theta_interval = self.param['theta_interval']
        self.n_angle = int(180/theta_interval)
        freq_step = self.param['freq_step']
        freq_max = 1./self.param['min_wavelen']
        freq_min = 1./self.param['max_wavelen']
        bandwidth = self.param['bandwidth']
        self.n_freq = int(np.log(freq_max/freq_min)/np.log(freq_step)) + 1
        frequencies = freq_max/freq_step**np.arange(self.n_freq)

        kernels = [gabor_kernel(f, theta=t, bandwidth=bandwidth) for f in frequencies 
                  for t in np.arange(0, np.pi, np.deg2rad(theta_interval))]
        self.kernels = map(np.real, kernels)
        self.n_kernel = len(kernels)

        print '=== filter using Gabor filters ==='
        print 'num. of kernels: %d' % (self.n_kernel)
        print 'frequencies:', frequencies
        print 'wavelength (pixels):', 1/frequencies

        max_kern_size = np.max([kern.shape[0] for kern in kernels])
        print 'max kernel matrix size:', max_kern_size
        
    def compute_features(self):
        self.get_kernels()

        def convolve_per_proc(i):
            return fftconvolve(self.img, self.kernels[i], 'same').astype(np.half)
        
        filtered = Pool().map(convolve_per_proc, range(self.n_kernel))
        
        self.features = np.empty((self.img.shape[0], self.img.shape[1], self.n_kernel), dtype=np.half)
        for i in range(self.n_kernel):
            self.features[...,i] = filtered[i]

        del filtered

#         save_array(features, 'features')

        self.n_feature = self.features.shape[-1]

    
    def compute_texton(self):
        print '=== compute rotation-invariant texton map using K-Means ==='

        n_texton = int(self.param['n_texton'])

        X = self.features.reshape(-1, self.n_feature)
        n_data = X.shape[0]
        n_splits = 1000
        n_sample = int(self.param['n_sample'])
        centroids = np.array(random.sample(X, n_texton))

        n_iter = int(self.param['n_iter'])

        def compute_dist_per_proc(x):
            X_partial, c_all_rot = x
            D = cdist(X_partial, c_all_rot, 'sqeuclidean')
            ci, ri = np.unravel_index(D.argmin(axis=1), (n_texton, self.n_angle))
            return np.c_[ci, ri]

        for iteration in range(n_iter):

            data = random.sample(X, n_sample)

            print 'iteration', iteration
            centroid_all_rotations = np.vstack([np.concatenate(np.roll(np.split(c, self.n_freq), i)) 
                                    for c,i in itertools.product(centroids, range(self.n_angle))])

            r = Pool().map(compute_dist_per_proc, zip(np.array_split(data, n_splits, axis=0), 
                                                itertools.repeat(centroid_all_rotations, n_splits)))
            
            res = np.vstack(r)  

            labels = res[:,0]
            rotations = res[:,1]

            centroids_new = np.zeros((n_texton, self.n_feature))
            for d, l, r in itertools.izip(data, labels, rotations):
                rot = np.concatenate(np.roll(np.split(d, self.n_freq), i))
                centroids_new[l] += rot

            counts = np.bincount(labels, minlength=n_texton)
            centroids_new /= counts[:, np.newaxis]
            centroids_new[counts==0] = centroids[counts==0]
            print np.sqrt(np.sum((centroids - centroids_new)**2, axis=1)).mean()

            centroids = centroids_new

        print 'kmeans completes'
        centroid_all_rotations = np.vstack([np.concatenate(np.roll(np.split(c, self.n_freq), i)) 
                                    for c,i in itertools.product(centroids, range(self.n_angle))])

        r = Pool().map(compute_dist_per_proc, zip(np.array_split(X, n_splits, axis=0), 
                                                itertools.repeat(centroid_all_rotations, n_splits)))
        
        res = np.vstack(r)

        labels = res[:,0]
        rotations = res[:,1]

        textonmap = labels.reshape(self.features.shape[:2])
        textonmap[~mask] = -1
    
    
if __name__ == '__main__':
    class args:
        param_id = 'nissl324'
        img_file = '../DavidData/RS155_x5/RS155_x5_0004.tif'
        output_dir = '/oasis/scratch/csd181/yuncong/output'
        params_dir = '/oasis/projects/nsf/csd181/yuncong/Brain/params'
        
    img, param, result_name, output_dir = load_stuff(args)
    feature_extractor = FeatureExtractor(img, param)
    feature_extractor.compute_features()
    feature_extractor.compute_texton()

