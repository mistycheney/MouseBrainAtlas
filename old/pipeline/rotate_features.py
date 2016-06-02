#! /usr/bin/env python

import os
import argparse
import sys
import time

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Rotate features using Gabor filters')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("slice_ind", type=int, help="slice index")
parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
# parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
# parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
parser.add_argument("-w", "--which_part", type=int, help="which part 0 or 1 (default: %(default)s)", default=0)
args = parser.parse_args()


from joblib import Parallel, delayed

sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], 'utilities'))
from utilities2015 import *

dm = DataManager(gabor_params_id=args.gabor_params_id, 
                 # segm_params_id=args.segm_params_id, 
                 # vq_params_id=args.vq_params_id,
                 stack=args.stack_name, 
                 section=args.slice_ind)

dm._generate_kernels()

#============================================================

which_part = args.which_part

import numpy as np
from joblib import Parallel, delayed
import bloscpack as bp

def rotate_features(fs, j):
    features_tabular = fs.reshape((fs.shape[0], dm.n_freq, dm.n_angle))
    max_angle_indices = features_tabular.max(axis=1).argmax(axis=-1)
    features_rotated = np.reshape([np.roll(features_tabular[i], -ai, axis=-1) 
                               for i, ai in enumerate(max_angle_indices)], (fs.shape[0], dm.n_freq * dm.n_angle))
    
    return features_rotated

t = time.time()
sys.stderr.write('load filtered values ...')

features = []
for i in range(dm.n_kernel):

    sys.stderr.write('%d\n'%i)
    a = bp.unpack_ndarray_file(os.environ['GORDON_RESULT_DIR']+'/feature_%03d.bp'%i).reshape((-1,))

    if which_part == 0:
        features.append(a[:len(a)/2])
    else: 
        features.append(a[len(a)/2:])

features = np.asarray(features).T

sys.stderr.write('done in %f seconds\n' % (time.time() - t))

t = time.time()
sys.stderr.write('rotate features ...')

items_per_job = 100

features_rotated = Parallel(n_jobs=8)(delayed(rotate_features)(fs, i) 
                   for i, fs in enumerate(np.array_split(features, features.shape[0]/items_per_job)))
features_rotated = np.vstack(features_rotated)

print features_rotated.shape
bp.pack_ndarray_file(features_rotated, os.environ['GORDON_RESULT_DIR']+'/featuresRotated_%d.bp'%which_part)


sys.stderr.write('done in %f seconds\n' % (time.time() - t))