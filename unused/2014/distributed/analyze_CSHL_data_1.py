#! /usr/bin/env python

from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.util import img_as_float, img_as_ubyte, pad
from skimage.transform import integral_image

import numpy as np

from joblib import Parallel, delayed

from scipy.signal import fftconvolve
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial.distance import cdist, pdist

import matplotlib.pyplot as plt

from utilities2015 import *

import os, sys
import cv2
import time

stack = sys.argv[1]
secind = int(sys.argv[2])

os.environ['GORDON_DATA_DIR'] = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_processed'
os.environ['GORDON_REPO_DIR'] = '/oasis/projects/nsf/csd395/yuncong/Brain'
os.environ['GORDON_RESULT_DIR'] = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_results'

dm = DataManager(data_dir=os.environ['GORDON_DATA_DIR'],
                 repo_dir=os.environ['GORDON_REPO_DIR'],
                 result_dir=os.environ['GORDON_RESULT_DIR'],
                 stack=stack, section=secind)

dm._load_image()

dm.mask = np.zeros_like(dm.image, np.bool)
dm.mask[1848:1848+4807, 924:924+10186] = True

rs, cs = np.where(dm.mask)
ymax = rs.max()
ymin = rs.min()
xmax = cs.max()
xmin = cs.min()
h = ymax-ymin+1
w = xmax-xmin+1


t = time.time()
print 'gabor filtering...',

def convolve_per_proc(i):
    pf = fftconvolve(dm.image[ymin-dm.max_kern_size:ymax+1+dm.max_kern_size, 
                              xmin-dm.max_kern_size:xmax+1+dm.max_kern_size], 
                       dm.kernels[i], 'same').astype(np.half)
    sys.stderr.write('filtered kernel %d\n'%i)
    
    return pf[dm.max_kern_size:-dm.max_kern_size, dm.max_kern_size:-dm.max_kern_size]

filtered = Parallel(n_jobs=4)(delayed(convolve_per_proc)(i) for i in range(dm.n_kernel))
features = np.asarray(filtered)

del filtered

print 'done in', time.time() - t, 'seconds'


def rotate_features(fs):
    features_tabular = fs.reshape((fs.shape[0], dm.n_freq, dm.n_angle))
    max_angle_indices = features_tabular.max(axis=1).argmax(axis=-1)
    features_rotated = np.reshape([np.roll(features_tabular[i], -ai, axis=-1) 
                               for i, ai in enumerate(max_angle_indices)], (fs.shape[0], dm.n_freq * dm.n_angle))
    
    return features_rotated
    
    
t = time.time()
print 'rotate features ...',

n_splits = 1000
features_rotated = Parallel(n_jobs=16)(delayed(rotate_features)(fs) 
                           for fs in np.array_split(features.reshape((dm.n_kernel,-1)).T, n_splits))
features_rotated = np.vstack(features_rotated)

print 'done in', time.time() - t, 'seconds'

dm.save_pipeline_result(features_rotated, 'featuresRotated', 'npy')