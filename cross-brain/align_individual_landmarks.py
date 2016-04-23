#! /usr/bin/env python

"""
Align the individual landmarks on atlas to the score volume of test brains using rigid transform.
"""

import numpy as np

import sys
import os

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from registration_utilities import *


from joblib import Parallel, delayed
import time

import logging

from collections import defaultdict

volume_dir = '/home/yuncong/csd395/CSHL_volumes/'

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


stack = sys.argv[1]

atlasProjected_volume = bp.unpack_ndarray_file(os.path.join(volume_dir, '%(stack)s/%(stack)s_volume_atlasProjected.bp' % {'stack': stack}))


def parallel_where(name, num_samples=None):
    
    w = np.where(atlasProjected_volume == labels_twoSides_indices[name])
    
    if num_samples is not None:
        n = len(w[0])
        sample_indices = np.random.choice(range(n), min(num_samples, n), replace=False)
        return np.c_[w[1][sample_indices].astype(np.int16), 
                     w[0][sample_indices].astype(np.int16), 
                     w[2][sample_indices].astype(np.int16)]
    else:
        return np.c_[w[1].astype(np.int16), w[0].astype(np.int16), w[2].astype(np.int16)]

t = time.time()

atlasProjected_nzs = Parallel(n_jobs=16)(delayed(parallel_where)(name, num_samples=int(1e5)) 
                                for name in labels_twoSides[1:])
atlasProjected_nzs = {name: nzs for name, nzs in zip(labels_twoSides[1:], atlasProjected_nzs)}

sys.stderr.write('load atlas: %f seconds\n' % (time.time() - t)) #~ 4s, sometime 13s


downsample_factor = 16

section_thickness = 20 # in um
xy_pixel_distance_lossless = 0.46
xy_pixel_distance_tb = xy_pixel_distance_lossless * 32 # in um, thumbnail

xy_pixel_distance_downsampled = xy_pixel_distance_lossless * downsample_factor
z_xy_ratio_downsampled = section_thickness / xy_pixel_distance_downsampled


# Load test volume
atlasAlignParams_dir = '/oasis/projects/nsf/csd395/yuncong/CSHL_atlasAlignParams'
params_dir = create_if_not_exists(atlasAlignParams_dir + '/' + stack)

volume2_allLabels = {}

for name in labels_unsided:
    
    if name == 'BackG':
        continue

    volume2_roi = bp.unpack_ndarray_file(os.path.join(volume_dir, '%(stack)s/%(stack)s_scoreVolume_%(label)s.bp' % \
                                                      {'stack': stack, 'label': name})).astype(np.float16)
    volume2_allLabels[name] = volume2_roi
    del volume2_roi

test_ydim, test_xdim, test_zdim = volume2_allLabels.values()[0].shape

print test_xdim, test_ydim, test_zdim


########### Load Gradient ###########

dSdxyz = {name: np.empty((3, test_ydim, test_xdim, test_zdim), dtype=np.float16) for name in labels_unsided[1:]}

t1 = time.time()

for name in labels_unsided:
    
    if name == 'BackG':
        continue

    t = time.time()
    
    dSdxyz[name][0] = bp.unpack_ndarray_file(volume_dir + '/%(stack)s/%(stack)s_scoreVolume_%(label)s_gx.bp' % {'stack':stack, 'label':name})
    dSdxyz[name][1] = bp.unpack_ndarray_file(volume_dir + '/%(stack)s/%(stack)s_scoreVolume_%(label)s_gy.bp' % {'stack':stack, 'label':name})
    dSdxyz[name][2] = bp.unpack_ndarray_file(volume_dir + '/%(stack)s/%(stack)s_scoreVolume_%(label)s_gz.bp' % {'stack':stack, 'label':name})
    
    sys.stderr.write('load gradient %s: %f seconds\n' % (name, time.time() - t)) # ~7s

sys.stderr.write('overall: %f seconds\n' % (time.time() - t1)) # 140s

def matrix_exp(w):
    
    wx, wy, wz = w
    w_skew = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])
    
    theta = np.sqrt(np.sum(w**2))
    
    exp_w = np.eye(3) + np.sin(theta)/theta*w_skew + (1-np.cos(theta))/theta**2*np.dot(w_skew, w_skew)
    return exp_w

def matrix_exp_v(v):
    t = v[:3]
    w = v[3:]
    
    theta = np.sqrt(np.sum(w**2))
    
    wx, wy, wz = w
    w_skew = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])
    exp_w = np.eye(3) + np.sin(theta)/theta*w_skew + (1-np.cos(theta))/(theta**2)*np.dot(w_skew, w_skew)
    
    V = np.eye(3) + (1-np.cos(theta))/(theta**2)*w_skew + (theta-np.sin(theta))/(theta**3)*np.dot(w_skew, w_skew)
    
    return exp_w, np.dot(V, t)

def step(T, name, lr, verbose=False, num_samples=1000):
    '''
    T: 1x12 vector
    l: landmark class label
    '''
    
    name_unsided = labelMap_sidedToUnsided[name]
        
    pts_prime = transform_points(T, pts_centered=atlasProjected_pts_centered[name], c_prime=test_centroid2).astype(np.int16)
    
    xs_prime, ys_prime, zs_prime = pts_prime.T
        
    valid = (xs_prime >= 0) & (ys_prime >= 0) & (zs_prime >= 0) & \
            (xs_prime < test_xdim) & (ys_prime < test_ydim) & (zs_prime < test_zdim)
    
    if verbose:
        print 'nz', np.count_nonzero(valid) 
        
    assert np.count_nonzero(valid) > 0, 'No valid pixel after transform: %s' % name
    
    xs_prime_valid, ys_prime_valid, zs_prime_valid = pts_prime[valid].T
        
    voxel_probs_valid = volume2_allLabels[name_unsided][ys_prime_valid, xs_prime_valid, zs_prime_valid] / 1e6
    score = voxel_probs_valid.sum()
    
    if num_samples is not None:
        # sample some voxels # this seems to make optimization more stable than using all voxels
    
        ii = np.random.choice(range(np.count_nonzero(valid)), 
                              min(num_samples, np.count_nonzero(valid)), 
                              replace=False)
        
        dSdx = dSdxyz[name_unsided][0, ys_prime_valid, xs_prime_valid, zs_prime_valid][ii]        
        dSdy = dSdxyz[name_unsided][1, ys_prime_valid, xs_prime_valid, zs_prime_valid][ii]
        dSdz = dSdxyz[name_unsided][2, ys_prime_valid, xs_prime_valid, zs_prime_valid][ii]
        
        xss = xs_prime_valid.astype(np.float)[ii]
        yss = ys_prime_valid.astype(np.float)[ii]
        zss = zs_prime_valid.astype(np.float)[ii]
        
    else:
        # use all voxels    
        dSdx = dSdxyz[name_unsided][0, ys_prime_valid, xs_prime_valid, zs_prime_valid]
        dSdy = dSdxyz[name_unsided][1, ys_prime_valid, xs_prime_valid, zs_prime_valid]
        dSdz = dSdxyz[name_unsided][2, ys_prime_valid, xs_prime_valid, zs_prime_valid]

        xss = xs_prime_valid.astype(np.float)
        yss = ys_prime_valid.astype(np.float)
        zss = zs_prime_valid.astype(np.float)

    #############################################
    
    q = np.c_[dSdx, dSdy, dSdz, -dSdy*zss + dSdz*yss, dSdx*zss - dSdz*xss, -dSdx*yss + dSdy*xss]
    
    dMdv = q.sum(axis=0)

    if verbose:
        print 'q:', q
        print 'dMdv:', dMdv
        print 'score:', score

#     lr = np.array([0, 0, 0, 0, 0, 1e-2])
    global dMdv_historical
    dMdv_historical += dMdv**2
    dMdv_adjusted = dMdv / (1e-10 + np.sqrt(dMdv_historical))
    v_opt = lr * dMdv_adjusted # no minus sign because maximizing

#     global iteration
#     lr = np.array([0, 0, 0, 0, 0, 1e-7])
#     v_opt = lr * np.exp(-iteration/1000.) * dMdv # no minus sign because maximizing
#     v_opt = lr * dMdv # no minus sign because maximizing

    if verbose:
        print 'v_opt:', v_opt

    theta = np.sqrt(np.sum(v_opt[3:]**2))
    if verbose:
        print 'theta:', theta
    assert theta < np.pi
        
    exp_w, Vt = matrix_exp_v(v_opt)
    
    if verbose:
        print 'Vt:' , Vt
    
    Tm = np.reshape(T, (3,4))
    t = Tm[:, 3]
    R = Tm[:, :3]
                        
    R_new = np.dot(exp_w, R)
    t_new = np.dot(exp_w, t) + Vt
    
    if verbose:
        print '\n'

    return np.column_stack([R_new, t_new]).flatten(), score

history_len = 200
T0 = np.array([1,0,0,0,0,1,0,0,0,0,1,0])
max_iter = 5000

for name_of_interest in labels_twoSides:

    if name_of_interest == 'BackG' or name_of_interest == 'outerContour':
        continue
    
    print name_of_interest
    
    # set the rotation center of both atlas and test volume to the landmark centroid after affine projection
    
    global atlasProjected_centroid, test_centroid2, atlasProjected_pts_centered
    
    atlasProjected_centroid = atlasProjected_nzs[name_of_interest].mean(axis=0)
    test_centroid2 = atlasProjected_centroid.copy()
    atlasProjected_pts_centered = {name: nzs - atlasProjected_centroid for name, nzs in atlasProjected_nzs.iteritems()}
    
    ############ gradient descent ############

    dMdv_historical = np.zeros((6,))

    score_best = 0
    scores = []
    
    T = T0.copy()

    for iteration in range(max_iter):
        
        success = False
        c = 0
        while not success and c < 10:
            try:
                c += 1
                T, s = step(T, name=name_of_interest, lr=np.array([1,1,1,1e-2,1e-2,1e-2]), verbose=False,
                            num_samples=10000)
                success = True
            except:
                pass
        
        scores.append(s)

        if iteration > 2*history_len:
            if np.abs(np.mean(scores[iteration-history_len:iteration]) - \
                      np.mean(scores[iteration-2*history_len:iteration-history_len])) < 1e-5:
                break

        if s > score_best:
            best_gradient_descent_params = T
            score_best = s
    
    with open(params_dir + '/%(stack)s_%(name)s_transformUponAffineProjection.txt' % {'stack': stack, 'name': name_of_interest}, 
              'w') as f:
        f.write((' '.join(['%f']*12)+'\n') % tuple(best_gradient_descent_params))
        f.write((' '.join(['%d']*3)+'\n') % tuple(np.r_[test_xdim, test_ydim, test_zdim]))
        f.write((' '.join(['%.1f']*3)+'\n') % tuple(atlasProjected_centroid))
        f.write((' '.join(['%.1f']*3)+'\n') % tuple(test_centroid2))
