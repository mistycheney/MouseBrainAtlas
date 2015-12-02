#! /usr/bin/env python

import os
import argparse
import sys
import time

sys.stderr.write('initialize ...')
t = time.time()

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generate features using Gabor filters')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("slice_ind", type=int, help="slice index")
# parser.add_argument("part", type=int, help="which part of the image to filter (0,1,2,3)")
parser.add_argument("xmin", type=int, help="xmin")
parser.add_argument("xmax", type=int, help="xmax")
parser.add_argument("ymin", type=int, help="ymin")
parser.add_argument("ymax", type=int, help="ymax")
parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')

# parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
# parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
args = parser.parse_args()

from joblib import Parallel, delayed

sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], 'utilities'))
from utilities2015 import *

sys.stderr.write('done in %f seconds\n' % (time.time() - t))

sys.stderr.write('initialize data manager ...')
t = time.time()

dm = DataManager(gabor_params_id=args.gabor_params_id, 
                 # segm_params_id=args.segm_params_id, 
                 # vq_params_id=args.vq_params_id,
                 stack=args.stack_name, 
                 section=args.slice_ind,
                 result_dir='/scratch/yuncong/CSHL_data_results')

sys.stderr.write('done in %f seconds\n' % (time.time() - t))

sys.stderr.write('reading image ...')
t = time.time()

dm._load_image(versions=['gray'])
dm._generate_kernels()

sys.stderr.write('done in %f seconds\n' % (time.time() - t))

#============================================================

import numpy as np
from joblib import Parallel, delayed
from scipy.signal import fftconvolve
import bloscpack as bp

# try:
#     raise
#     features_rotated = dm.load_pipeline_result('featuresRotated')
# # if dm.check_pipeline_result('featuresRotated'):
#     print "features_rotated.npy already exists, skip"
# # else:
# except:
    
    # if dm.check_pipeline_result('features'):
    # # if False:
    #     print "features.npy already exists, load"
    #     features = dm.load_pipeline_result('features')
    # else:

# if args.part == 0:
#     xmin = dm.xmin
#     xmax = dm.image_width/2-1
#     ymin = dm.ymin
#     ymax = dm.image_height/2-1
# elif args.part == 1:
#     xmin = dm.image_width/2
#     xmax = dm.xmax
#     ymin = dm.ymin
#     ymax = dm.image_height/2-1
# elif args.part == 2:
#     xmin = dm.xmin
#     xmax = dm.image_width/2-1
#     ymin = dm.image_height/2
#     ymax = dm.ymax
# elif args.part == 3:
#     xmin = dm.image_width/2
#     xmax = dm.xmax
#     ymin = dm.image_height/2
#     ymax = dm.ymax

# if args.part == 0:
#     xmin = dm.xmin
#     xmax = dm.xmin+dm.w/2-1
#     ymin = dm.ymin
#     ymax = dm.ymin+dm.h/2-1
# elif args.part == 1:
#     xmin = dm.xmin+dm.w/2
#     xmax = dm.xmax
#     ymin = dm.ymin
#     ymax = dm.ymin+dm.h/2-1
# elif args.part == 2:
#     xmin = dm.xmin
#     xmax = dm.xmin+dm.w/2-1
#     ymin = dm.ymin+dm.h/2
#     ymax = dm.ymax
# elif args.part == 3:
#     xmin = dm.xmin+dm.w/2
#     xmax = dm.xmax
#     ymin = dm.ymin+dm.h/2
#     ymax = dm.ymax

# xmin = dm.xmin
# xmax = dm.xmax
# ymin = dm.ymin
# ymax = dm.ymax

xmin = args.xmin
xmax = args.xmax
ymin = args.ymin
ymax = args.ymax

def convolve_per_proc(i):

    pf = fftconvolve(dm.image[ymin-dm.max_kern_size : ymax+1+dm.max_kern_size, 
                              xmin-dm.max_kern_size : xmax+1+dm.max_kern_size], 
                     dm.kernels[i], 'same').astype(np.half)

    sys.stderr.write('filtered kernel %d\n'%i)

    return pf[dm.max_kern_size:-dm.max_kern_size, dm.max_kern_size:-dm.max_kern_size].flatten()

t = time.time()

mask = dm.mask[ymin:ymax+1, xmin:xmax+1]

perc = np.count_nonzero(mask) / float(mask.size)
sys.stderr.write('masked pixels = %.2f, ' % perc)

if perc == 0:
    sys.exit(0)

# sys.stderr.write('gabor filtering test ...')

# f = convolve_per_proc(88)

# sys.stderr.write('done in %f seconds\n' % (time.time() - t))

# t = time.time()

sys.stderr.write('gabor filtering ...')

# features = np.empty((dm.n_kernel, (ymax+1-ymin)*(xmax+1-xmin)), np.half)
# for i in range(dm.n_kernel):
#     features[i] = convolve_per_proc(i)
# features = features.T

features = Parallel(n_jobs=16, backend='threading')(delayed(convolve_per_proc)(i) for i in range(dm.n_kernel))
# features = Parallel(n_jobs=16)(delayed(convolve_per_proc)(i) for i in range(dm.n_kernel))
# features = Parallel(n_jobs=4)(delayed(convolve_per_proc)(i) for i in range(dm.n_kernel))
# features = Parallel(n_jobs=10)(delayed(convolve_per_proc)(i) for i in range(88,98))
# features = Parallel(n_jobs=16, backend='threading')(delayed(convolve_per_proc)(i) for i in range(88,96)+range(88,96))

sys.stderr.write('done in %f seconds\n' % (time.time() - t))

# sys.exit(0)

t = time.time()
sys.stderr.write('transpose ...')

features = np.asarray(features).T  # n x 99

sys.stderr.write('done in %f seconds\n' % (time.time() - t))

t = time.time()
sys.stderr.write('mask ...')

features_masked = features[mask.flat]

sys.stderr.write('done in %f seconds\n' % (time.time() - t))

sys.exit(0)

# t = time.time()
# sys.stderr.write('dump features ...')

# bp.pack_ndarray_file(features, os.environ['GORDON_RESULT_DIR']+'/features_%d_%d_%d_%d.bp'%(xmin, xmax, ymin, ymax))

# sys.stderr.write('done in %f seconds\n' % (time.time() - t))

# sys.exit(0)

def rotate_features(fs):
    features_tabular = fs.reshape((fs.shape[0], dm.n_freq, dm.n_angle))
    max_angle_indices = features_tabular.max(axis=1).argmax(axis=-1)
    features_rotated = np.reshape([np.roll(features_tabular[i], -ai, axis=-1) 
                                    for i, ai in enumerate(max_angle_indices)], 
                                (fs.shape[0], dm.n_freq * dm.n_angle))
    
    return features_rotated

t = time.time()
sys.stderr.write('rotate features ...')

items_per_job = 100

# n = features.shape[0]
n = features_masked.shape[0]

# features_rotated = Parallel(n_jobs=16)(delayed(rotate_features)(fs, i) 
                   # for i, fs in enumerate(np.array_split(features, features.shape[0]/items_per_job)))

features_masked_rotated = Parallel(n_jobs=16)(delayed(rotate_features)(features_masked[si:ei]) 
                                        for si, ei in zip(np.arange(0, n, items_per_job), 
                                            np.arange(0, n, items_per_job) + items_per_job))

# features_rotated = Parallel(n_jobs=16)(delayed(rotate_features)(features[si:ei]) 
#                                         for si, ei in zip(np.arange(0, n, items_per_job), 
#                                             np.arange(0, n, items_per_job) + items_per_job))
# features_rotated = np.vstack(features_rotated)

features_masked_rotated = np.vstack(features_masked_rotated)

sys.stderr.write('done in %f seconds\n' % (time.time() - t))


t = time.time()
sys.stderr.write('dump featuresRotated ...')

# bp.pack_ndarray_file(features_rotated, os.environ['GORDON_RESULT_DIR']+'/2featuresRotated_%d.bp'%args.part)
# dm.save_pipeline_result(features_rotated, 'featuresRotated%d'%args.part)

# dm.save_pipeline_result(features_masked_rotated, 'featuresMaskedRotated%d'%args.part)

sys.stderr.write('done in %f seconds\n' % (time.time() - t))