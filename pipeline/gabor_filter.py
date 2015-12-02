#! /usr/bin/env python

import sys
import time

sys.stderr.write('import ...')
t = time.time()

import os
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generate features using Gabor filters')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("slice_ind", type=int, help="slice index")
parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
# parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
# parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
args = parser.parse_args()

from joblib import Parallel, delayed

import numpy as np
from scipy.signal import fftconvolve
import bloscpack as bp

sys.stderr.write('done in %f seconds\n' % (time.time() - t))

sys.stderr.write('import utilities ...')
t = time.time()

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

from itertools import product

block_size = 7000

def convolve_per_proc(i, xmin, xmax, ymin, ymax):

    pf = fftconvolve(dm.image[ymin-dm.max_kern_size : ymax+1+dm.max_kern_size, 
                              xmin-dm.max_kern_size : xmax+1+dm.max_kern_size], 
                     dm.kernels[i], 'same').astype(np.half)

    sys.stderr.write('filtered kernel %d\n'%i)

    # return pf[dm.max_kern_size:-dm.max_kern_size, dm.max_kern_size:-dm.max_kern_size].flatten()
    return pf[dm.max_kern_size:-dm.max_kern_size, dm.max_kern_size:-dm.max_kern_size]


def rotate_features(fs):
    features_tabular = fs.reshape((fs.shape[0], dm.n_freq, dm.n_angle))
    max_angle_indices = features_tabular.max(axis=1).argmax(axis=-1)
    features_rotated = np.reshape([np.roll(features_tabular[i], -ai, axis=-1) 
                                    for i, ai in enumerate(max_angle_indices)], 
                                (fs.shape[0], dm.n_freq * dm.n_angle))

    # del features_tabular, max_angle_indices
    
    return features_rotated

# def rotate_features(fs):

#     features_tabular = fs.reshape((fs.shape[0], dm.n_freq, dm.n_angle))
#     max_angle_indices = features_tabular.max(axis=1).argmax(axis=-1)

#     # t = time.time()
#     features_rotated = np.empty_like(features_tabular)

#     for a in range(dm.n_angle):
#         q = np.where(max_angle_indices == a)[0]
#         if len(q) == 0:
#             continue
#         v = np.roll(np.vstack(features_tabular[q]), -a, axis=-1)
#         features_rotated[q] = np.split(v, len(q), axis=0)

#     # sys.stderr.write('3, done in %f seconds\n' % (time.time() - t))
#     features_rotated = np.reshape(features_rotated, (fs.shape[0], dm.n_freq * dm.n_angle))

#     # del features_tabular, max_angle_indices
    
#     return features_rotated


for col, xmin in enumerate(range(dm.xmin, dm.xmax, block_size)):
    for row, ymin in enumerate(range(dm.ymin, dm.ymax, block_size)):

        xmax = xmin + block_size - 1
        ymax = ymin + block_size - 1

        print xmin, xmax, ymin, ymax

        # os.system(os.environ['GORDON_REPO_DIR']+'/pipeline/gabor_filter_part.py %s %s %d %d %d %d' % (args.stack_name, args.slice_ind, xmin, xmax, ymin, ymax))

        mask = dm.mask[ymin:ymax+1, xmin:xmax+1]

        perc = np.count_nonzero(mask) / float(mask.size)
        sys.stderr.write('masked pixels = %.2f, ' % perc)

        if perc == 0:
            continue

        t = time.time()
        sys.stderr.write('gabor filtering ...')

        features = Parallel(n_jobs=16, backend='threading')(delayed(convolve_per_proc)(i, xmin, xmax, ymin, ymax) for i in range(dm.n_kernel))
        h, w = features[0].shape
        # 99 x h x w

        sys.stderr.write('done in %f seconds\n' % (time.time() - t))

        # t = time.time()
        # sys.stderr.write('transpose ...')

        # # features = np.asarray(features).T  # n x 99
        # # features = np.roll(features, 0, -1)  # h x w x 99
        # features = np.reshape(features, (dm.n_kernel, h * w)).T  # n x 99
        # print features.shape

        # sys.stderr.write('done in %f seconds\n' % (time.time() - t))

        t = time.time()
        sys.stderr.write('mask ...')

        # features_masked = features[mask.flat]

        features_masked = np.asarray([features[i][mask] for i in range(dm.n_kernel)]).T # n x 99

        # bp.pack_ndarray_file(features_masked, '/scratch/yuncong/tmp.bp')

        print features_masked.shape

        sys.stderr.write('done in %f seconds\n' % (time.time() - t))

        del features
        # sys.exit(0)

        # features_masked = bp.unpack_ndarray_file('/scratch/yuncong/tmp.bp')

        t = time.time()
        sys.stderr.write('rotate features ...')

        items_per_job = 1000   # 100 takes 144s, 500 takes 95s, 1000 takes 95s, 10000 takes 139s, 5000 takes 98s

        n = features_masked.shape[0]

        features_masked_rotated = Parallel(n_jobs=16)(delayed(rotate_features)(features_masked[si:ei]) 
                                for si, ei in zip(np.arange(0, n, items_per_job), np.arange(0, n, items_per_job) + items_per_job))

        features_masked_rotated = np.vstack(features_masked_rotated)

        sys.stderr.write('done in %f seconds\n' % (time.time() - t))

        t = time.time()
        sys.stderr.write('dumping ...')

        # (xmin, min(xmax, xmin+w-1), ymin, min(ymax, ymin+h-1))
        # bp.pack_ndarray_file(features_masked_rotated, '/scratch/yuncong/features_masked_rotated_r%d_c%d.bp' % (row, col))

        dm.save_pipeline_result(features_masked_rotated, 'featuresMaskedRotatedRow%dCol%d'%(row,col))

        del features_masked, features_masked_rotated

        sys.stderr.write('done in %f seconds\n' % (time.time() - t))    