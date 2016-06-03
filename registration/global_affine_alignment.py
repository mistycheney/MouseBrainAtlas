#! /usr/bin/env python

import numpy as np

import sys
import os

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from registration_utilities import *

from joblib import Parallel, delayed
import time

import logging

volume_dir = '/oasis/projects/nsf/csd395/yuncong/CSHL_volumes/'

atlasAlignOptLogs_dir = create_if_not_exists('/oasis/projects/nsf/csd395/yuncong/CSHL_atlasAlignOptLogs')
atlasAlignParams_dir = create_if_not_exists('/oasis/projects/nsf/csd395/yuncong/CSHL_atlasAlignParams')
annotationsViz_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_annotaionsPojectedViz'

labels_twoSides = []
labels_twoSides_indices = {}
with open(volume_dir + '/MD589/volume_MD589_annotation_withOuterContour_labelIndices.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        name, index = line.split()
        labels_twoSides.append(name)
        labels_twoSides_indices[name] = int(index)

labelMap_sidedToUnsided = {name: name if '_' not in name else name[:-2] for name in labels_twoSides_indices.keys()}
labels_unsided = ['BackG'] + sorted(set(labelMap_sidedToUnsided.values()) - {'BackG', 'outerContour'}) + ['outerContour']
labels_unsided_indices = dict((j, i) for i, j in enumerate(labels_unsided))

from collections import defaultdict

labelMap_unsidedToSided = defaultdict(list)
for name_sided, name_unsided in labelMap_sidedToUnsided.iteritems():
    labelMap_unsidedToSided[name_unsided].append(name_sided)
labelMap_unsidedToSided.default_factory = None

atlas_volume = bp.unpack_ndarray_file(os.path.join(volume_dir, 'MD589/volume_MD589_annotation_withOuterContour.bp'))

atlas_ydim, atlas_xdim, atlas_zdim = atlas_volume.shape
atlas_centroid = np.array([.5*atlas_xdim, .5*atlas_ydim, .5*atlas_zdim])
print atlas_centroid


def parallel_where(name, num_samples=None):
    """Return voxel locations of a certain structure."""
    w = np.where(atlas_volume == labels_twoSides_indices[name])

    if num_samples is not None:
        n = len(w[0])
        sample_indices = np.random.choice(range(n), min(num_samples, n), replace=False)
        return np.c_[w[1][sample_indices].astype(np.int16),
                     w[0][sample_indices].astype(np.int16),
                     w[2][sample_indices].astype(np.int16)]
    else:
        return np.c_[w[1].astype(np.int16), w[0].astype(np.int16), w[2].astype(np.int16)]

t = time.time()

atlas_nzs = Parallel(n_jobs=16)(delayed(parallel_where)(name, num_samples=int(1e5)) for name in labels_twoSides[1:])
atlas_nzs = {name: nzs for name, nzs in zip(labels_twoSides[1:], atlas_nzs)}

sys.stderr.write('load atlas: %f seconds\n' % (time.time() - t))
#~ 7s

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

pts_centered = {name: (np.concatenate([atlas_nzs[n] for n in labelMap_unsidedToSided[name]]) - atlas_centroid).astype(np.int16)
                         for name in labels_unsided[1:]}

# label_weights = {name: 0 if name == 'outerContour' else 1. for name in labels_unsided[1:]}
label_weights = {name: .1 if name == 'outerContour' else 1. for name in labels_unsided[1:]}

def compute_score_and_gradient(T):
    """Compute score and gradient."""
    global pts_centered

    score = 0
    dMdA = np.zeros((12,))

    for name in labels_unsided[1:]:
#         t1 = time.time()

        pts_prime = transform_points(T, pts_centered=pts_centered[name], c_prime=test_centroid)

        xs_prime, ys_prime, zs_prime = pts_prime.T.astype(np.int16)

        valid = (xs_prime >= 0) & (ys_prime >= 0) & (zs_prime >= 0) & \
                (xs_prime < test_xdim) & (ys_prime < test_ydim) & (zs_prime < test_zdim)

        if np.count_nonzero(valid) > 0:

            xs_prime_valid = xs_prime[valid]
            ys_prime_valid = ys_prime[valid]
            zs_prime_valid = zs_prime[valid]

            voxel_probs_valid = volume2_allLabels[name][ys_prime_valid, xs_prime_valid, zs_prime_valid] / 1e4

            score += label_weights[name] * voxel_probs_valid.sum()

            Sx = dSdxyz[name][0, ys_prime_valid, xs_prime_valid, zs_prime_valid]
            Sy = dSdxyz[name][1, ys_prime_valid, xs_prime_valid, zs_prime_valid]
            Sz = dSdxyz[name][2, ys_prime_valid, xs_prime_valid, zs_prime_valid]

            dxs, dys, dzs = pts_centered[name][valid].T

            q = np.c_[Sx*dxs, Sx*dys, Sx*dzs, Sx,
                          Sy*dxs, Sy*dys, Sy*dzs, Sy,
                          Sz*dxs, Sz*dys, Sz*dzs, Sz]

            dMdA += label_weights[name] * q.sum(axis=0)

            del voxel_probs_valid, q, Sx, Sy, Sz, dxs, dys, dzs, xs_prime_valid, ys_prime_valid, zs_prime_valid

#         sys.stderr.write('########### %s: %f seconds\n' % (labels[l], time.time() - t1))

        del valid, xs_prime, ys_prime, zs_prime, pts_prime

    return score, dMdA

def compute_score(T):
    """Compute score."""
    score = 0
    for name in labels_unsided[1:]:

        pts_prime = transform_points(T, pts_centered=pts_centered[name], c_prime=test_centroid)

        xs_prime, ys_prime, zs_prime = pts_prime.T.astype(np.int16)

        valid = (xs_prime >= 0) & (ys_prime >= 0) & (zs_prime >= 0) & \
            (xs_prime < test_xdim) & (ys_prime < test_ydim) & (zs_prime < test_zdim)

        voxel_probs_valid = volume2_allLabels[name][ys_prime[valid], xs_prime[valid], zs_prime[valid]] / 1e4

        score += label_weights[name] * voxel_probs_valid.sum()

        del voxel_probs_valid, valid, xs_prime, ys_prime, zs_prime, pts_prime

    return score

def compute_score_gradient(T):
    """Compute gradient of score."""
    dMdA = np.zeros((12,))

    for name in labels_unsided[1:]:
#
        pts_prime = transform_points(T, pts_centered=pts_centered[name], c_prime=test_centroid)

        xs_prime, ys_prime, zs_prime = pts_prime.T.astype(np.int16)

        valid = (xs_prime >= 0) & (ys_prime >= 0) & (zs_prime >= 0) & \
            (xs_prime < test_xdim) & (ys_prime < test_ydim) & (zs_prime < test_zdim)

        if np.count_nonzero(valid) > 0:

            xs_prime_valid = xs_prime[valid]
            ys_prime_valid = ys_prime[valid]
            zs_prime_valid = zs_prime[valid]

            Sx = dSdxyz[name][0, ys_prime_valid, xs_prime_valid, zs_prime_valid]
            Sy = dSdxyz[name][1, ys_prime_valid, xs_prime_valid, zs_prime_valid]
            Sz = dSdxyz[name][2, ys_prime_valid, xs_prime_valid, zs_prime_valid]

            dxs, dys, dzs = pts_centered[name][valid].T

            dMdA += label_weights[name] * np.c_[Sx*dxs, Sx*dys, Sx*dzs, Sx,
                          Sy*dxs, Sy*dys, Sy*dzs, Sy,
                          Sz*dxs, Sz*dys, Sz*dzs, Sz].sum(axis=0)

    return dMdA

stack = sys.argv[1]


################# Load Test Volume ######################

t = time.time()

volume2_allLabels = {}

for name in labels_unsided:

    if name == 'BackG':
        continue

    volume2_roi = bp.unpack_ndarray_file(os.path.join(volume_dir, '%(stack)s/%(stack)s_scoreVolume_%(label)s.bp' % \
                                                      {'stack': stack, 'label': name})).astype(np.float16)
    volume2_allLabels[name] = volume2_roi
    del volume2_roi

test_ydim, test_xdim, test_zdim = volume2_allLabels.values()[0].shape
test_centroid = np.r_[.5*test_xdim, .5*test_ydim, .5*test_zdim]

print 'test_xdim, test_ydim, test_zdim:', test_xdim, test_ydim, test_zdim
print 'test_centroid:', test_centroid

# test_xdim = volume_xmax - volume_xmin + 1
# test_ydim = volume_ymax - volume_ymin + 1
# test_zdim = volume_zmax - volume_zmin + 1

sys.stderr.write('load score volumes: %f seconds\n' % (time.time() - t))

###################### Load Gradient #####################

dSdxyz = {name: np.empty((3, test_ydim, test_xdim, test_zdim), dtype=np.float16) for name in labels_unsided[1:]}

t1 = time.time()

for name in labels_unsided:

    if name == 'BackG':
        continue

    t = time.time()

    dSdxyz[name][0] = bp.unpack_ndarray_file(volume_dir + '/%(stack)s/%(stack)s_scoreVolume_%(label)s_gx.bp' % {'stack':stack, 'label':name})
    dSdxyz[name][1] = bp.unpack_ndarray_file(volume_dir + '/%(stack)s/%(stack)s_scoreVolume_%(label)s_gy.bp' % {'stack':stack, 'label':name})
    dSdxyz[name][2] = bp.unpack_ndarray_file(volume_dir + '/%(stack)s/%(stack)s_scoreVolume_%(label)s_gz.bp' % {'stack':stack, 'label':name})

    sys.stderr.write('load gradient %s: %f seconds\n' % (name, time.time() - t)) # ~6s

sys.stderr.write('overall: %f seconds\n' % \
(time.time() - t1)) #~100s

handler = logging.FileHandler(atlasAlignOptLogs_dir + '/%(stack)s_atlasAlignOpt.log' % {'stack': stack})
handler.setLevel(logging.INFO)
logger.addHandler(handler)

################# Random Grid Search ######################

grid_search_iteration_number = 5
# grid_search_iteration_number = 1

params_best_upToNow = (0, 0, 0)
score_best_upToNow = 0

init_n = 1000

for iteration in range(grid_search_iteration_number):

    logger.info('grid search iteration %d', iteration)

    init_tx, init_ty, init_tz  = params_best_upToNow

    n = int(init_n*np.exp(-iteration/3.))

    sigma_tx = 300*np.exp(-iteration/3.)
    sigma_ty = 300*np.exp(-iteration/3.)
    sigma_tz = 100*np.exp(-iteration/3.)

    tx_grid = init_tx + sigma_tx * (2 * np.random.random(n) - 1)
    ty_grid = init_ty + sigma_ty * (2 * np.random.random(n) - 1)
    tz_grid = init_tz + sigma_tz * (2 * np.random.random(n) - 1)

    samples = np.c_[tx_grid, ty_grid, tz_grid]

    import time

    t = time.time()
    # num jobs * memory each job

    scores = Parallel(n_jobs=8)(delayed(compute_score)(np.r_[1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz])
                                for tx, ty, tz in samples)

#     scores = []
#     for tx, ty, tz in samples:
#         scores.append(compute_score([1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz]))

    sys.stderr.write('grid search: %f seconds\n' % (time.time() - t)) # ~23s

    score_best = np.max(scores)

    tx_best, ty_best, tz_best = samples[np.argmax(scores)]

    if score_best > score_best_upToNow:
        logger.info('%f %f', score_best_upToNow, score_best)

        score_best_upToNow = score_best
        params_best_upToNow = tx_best, ty_best, tz_best

        logger.info('%f %f %f', tx_best, ty_best, tz_best)

    logger.info('\n')

################# Gradient Descent ######################

lr1, lr2 = (10., 1e-1)
# lr1, lr2 = (1., 1e-3)

# auto_corr = .95

max_iter_num = 1000
fudge_factor = 1e-6 #for numerical stability
dMdA_historical = np.zeros((12,))

tx_best, ty_best, tz_best = params_best_upToNow
T_best = np.r_[1,0,0, tx_best, 0,1,0, ty_best, 0,0,1, tz_best]

lr = np.r_[lr2, lr2, lr2, lr1, lr2, lr2, lr2, lr1, lr2, lr2, lr2, lr1]

score_best = 0

scores = []

for iteration in range(max_iter_num):

    logger.info('iteration %d', iteration)

#     t = time.time()
    s, dMdA = compute_score_and_gradient(T_best)
#     sys.stderr.write('compute_score_and_gradient: %f seconds\n' % (time.time() - t)) #~ 2s/iteration or ~.5s: 1e5 samples per landmark

    dMdA_historical += dMdA**2
#     dMdA_historical = auto_corr * dMdA_historical + (1-auto_corr) * dMdA**2

    dMdA_adjusted = dMdA / (fudge_factor + np.sqrt(dMdA_historical))

    T_best += lr*dMdA_adjusted

#         logger.info('A: ' + ' '.join(['%f']*12) % tuple(A_best))
#         logger.info('dMdA adjusted: ' + ' '.join(['%f']*12) % tuple(dMdA_adjusted))

    logger.info('score: %f', s)
    scores.append(s)

    logger.info('\n')

    history_len = 50
    if iteration > 100:
        if np.abs(np.mean(scores[iteration-history_len:iteration]) - \
                  np.mean(scores[iteration-2*history_len:iteration-history_len])) < 1e-1:
            break

    if s > score_best:
#             logger.info('Current best')
        best_gradient_descent_params = T_best
        score_best = s

    np.save(atlasAlignOptLogs_dir + '/%(stack)s_scoreEvolutions.npy' % {'stack':stack}, scores)

with open(os.path.join(atlasAlignParams_dir, '%(stack)s/%(stack)s_3dAlignParams.txt' % {'stack':stack}), 'w') as f:

    f.writelines(' '.join(['%f']*len(best_gradient_descent_params)) % tuple(best_gradient_descent_params) + '\n')
    f.write((' '.join(['%d']*3)+'\n') % tuple([atlas_xdim, atlas_ydim, atlas_zdim]))
    f.write((' '.join(['%.1f']*3)+'\n') % tuple(atlas_centroid))
    f.write((' '.join(['%d']*3)+'\n') % tuple([test_xdim, test_ydim, test_zdim]))
    f.write((' '.join(['%.1f']*3)+'\n') % tuple(test_centroid))
