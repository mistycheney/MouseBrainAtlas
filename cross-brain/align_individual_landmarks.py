#! /usr/bin/env python

"""
Align the individual landmarks on atlas to the score volume of test brains using rigid transform.
"""

import numpy as np

import sys
import os

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *

from joblib import Parallel, delayed
import time

import logging

from registration_utilities import *

from collections import defaultdict

labels = ['BackG', '5N', '7n', '7N', '12N', 'Pn', 'VLL', 
          '6N', 'Amb', 'R', 'Tz', 'RtTg', 'LRt', 'LC', 'AP', 'sp5']

n_labels = len(labels)

labels_index = dict((j, i) for i, j in enumerate(labels))

labels_from_surround = dict( (l+'_surround', l) for l in labels[1:])

labels_surroundIncluded_list = labels[1:] + [l+'_surround' for l in labels[1:]]
labels_surroundIncluded = set(labels_surroundIncluded_list)

labels_surroundIncluded_index = dict((j, i) for i, j in enumerate(labels_surroundIncluded_list))

# colors = np.random.randint(0, 255, (len(labels_index), 3))
colors = np.loadtxt(os.environ['REPO_DIR'] + '/visualization/100colors.txt')
colors[labels_index['BackG']] = 1.

volume_dir = '/oasis/projects/nsf/csd395/yuncong/CSHL_volumes/'

volume1 = bp.unpack_ndarray_file(os.path.join(volume_dir, 'volume_MD589_annotation.bp'))

def parallel_where(l):
    w = np.where(volume1 == l)
    return np.array([w[1].astype(np.int16), w[0].astype(np.int16), w[2].astype(np.int16)]).T

t = time.time()

atlas_nzs = Parallel(n_jobs=16)(delayed(parallel_where)(l) for l in range(1, n_labels))

sys.stderr.write('load atlas: %f seconds\n' % (time.time() - t))

atlas_ydim, atlas_xdim, atlas_zdim = volume1.shape
atlas_centroid = (.5*atlas_xdim, .5*atlas_ydim, .5*atlas_zdim)
print atlas_centroid

atlas_vol_xmin, atlas_vol_xmax, atlas_vol_ymin, atlas_vol_ymax, atlas_vol_zmin, atlas_vol_zmax = \
np.loadtxt(os.path.join(volume_dir, 'volume_MD589_annotation_limits.txt'))


downsample_factor = 16

section_thickness = 20 # in um
xy_pixel_distance_lossless = 0.46
xy_pixel_distance_tb = xy_pixel_distance_lossless * 32 # in um, thumbnail
# factor = section_thickness/xy_pixel_distance_lossless

xy_pixel_distance_downsampled = xy_pixel_distance_lossless * downsample_factor
z_xy_ratio_downsampled = section_thickness / xy_pixel_distance_downsampled

from annotation_utilities import *
label_polygons = load_label_polygons_if_exists(stack='MD589', username='yuncong', force=False)

annotation_on_sections = get_annotation_on_sections(label_polygons=label_polygons, 
                                                    filtered_labels=labels_surroundIncluded)

landmark_range_limits = get_landmark_range_limits(stack='MD589', label_polygons=label_polygons, 
                                                  filtered_labels=labels_surroundIncluded)

landmark_zlimits = {l: [(int(z_xy_ratio_downsampled*e1) - atlas_vol_zmin, 
                         int(z_xy_ratio_downsampled*e2) -1 - atlas_vol_zmin) for e1, e2 in ranges] 
                    for l, ranges in landmark_range_limits.iteritems()}

landmark_zlimits_twoSides = {}
for l in range(1, n_labels):
    zlimits = landmark_zlimits[labels[l]]
    if len(zlimits) == 2:
        landmark_zlimits_twoSides[labels[l] + '_L'] = zlimits[0]
        landmark_zlimits_twoSides[labels[l] + '_R'] = zlimits[1]
    elif len(zlimits) == 1:
        landmark_zlimits_twoSides[labels[l]] = zlimits[0]
        
atlas_nzs_twoSides = {}
for name, (z_begin, z_end) in landmark_zlimits_twoSides.iteritems():
    
    if '_' in name:
        l = labels_index[name[:-2]]
    else:
        l = labels_index[name]
    
    nzs = atlas_nzs[l-1]
    atlas_nzs_twoSides[name] = nzs[(nzs[:,2] >= z_begin) & (nzs[:,2] <= z_end)]
    
############### Load test volume ###############

stack = sys.argv[1]

atlasAlignParams_dir = '/oasis/projects/nsf/csd395/yuncong/CSHL_atlasAlignParams'

with open(atlasAlignParams_dir + '/%(stack)s/%(stack)s_3dAlignParams.txt' % {'stack': stack}, 'r') as f:
    lines = f.readlines()
    
T_final = np.array(map(float, lines[1].strip().split()))

(volume_xmin, volume_xmax, volume_ymin, volume_ymax, volume_zmin, volume_zmax) = \
np.loadtxt(os.path.join(volume_dir, 'volume_%(stack)s_scoreMap_limits.txt' % {'stack': stack}), dtype=np.int)

global volume2_allLabels
# volume2_allLabels = []
volume2_allLabels = np.empty((n_labels-1, volume_ymax-volume_ymin+1, volume_xmax-volume_xmin+1, volume_zmax-volume_zmin+1), 
         dtype=np.float16) # use float32 is faster than float16 (2.5s/landmark), maybe because bp files are stored using float32

for l in range(1, n_labels):

    t = time.time()

    volume2 = bp.unpack_ndarray_file(os.path.join(volume_dir, 'volume_%(stack)s_scoreMap_%(label)s.bp' % \
                                                  {'stack': stack, 'label': labels[l]}))

    volume2_cropped = volume2[volume_ymin:volume_ymax+1, volume_xmin:volume_xmax+1]
    # copy is important, because then you can delete the large array

    volume2_allLabels[l-1] = volume2_cropped.copy()
    
#     volume2_allLabels.append(volume2_cropped.copy())

    del volume2, volume2_cropped
    
    sys.stderr.write('load scoremap %s: %f seconds\n' % (labels[l], time.time() - t)) # ~2.5s

test_ydim, test_xdim, test_zdim = volume2_allLabels[0].shape
test_centroid = (.5*test_xdim, .5*test_ydim, .5*test_ydim)
test_cx, test_cy, test_cz = test_centroid

print test_xdim, test_ydim, test_zdim
print test_centroid

dSdxyz = np.empty((n_labels-1, 3) + volume2_allLabels[0].shape, dtype=np.float16) 

# if memory is not limited, using float32 is faster, because the output of np.gradient is of type float32
# time for storing output: float16 4s (due to dtype conversion overhead), float32 1s

# using float16 avoids memory issues that make gradient computation utterly slow, 30s vs. 4s

################# COMPUTE GRADIENTS ######################

# dSdxyz = {}
# DO NOT use python list because python will use contiguous memory for it
# http://stackoverflow.com/questions/12274060/does-python-use-linked-lists-for-lists-why-is-inserting-slow  

t1 = time.time()

for l in range(1, n_labels):

    t = time.time()
    
    gy, gx, gz = np.gradient(volume2_allLabels[l-1], 3, 3, 3) # 3.3 second, much faster than loading
    # if memory is limited, this will be very slow
    
    sys.stderr.write('gradient %s: %f seconds\n' % (labels[l], time.time() - t))
    
    t = time.time()
    
    dSdxyz[l-1, 0] = gx
    dSdxyz[l-1, 1] = gy
    dSdxyz[l-1, 2] = gz
    
#     dSdxyz[labels[l]] = np.array([gx, gy, gz]) # use np.array is better; using python list also causes contiguous memory overhead
    
#     del gx, gy, gz # does not make a difference
    
    sys.stderr.write('store %s: %f seconds\n' % (labels[l], time.time() - t))
    
sys.stderr.write('overall: %f seconds\n' % (time.time() - t1))

def transform_points(T, pts=None, c=None, pts_centered=None, c_prime=0):
    '''
    T: 1x12 vector
    c: center of volume 1
    c_prime: center of volume 2
    pts: nx3
    '''
    
    if pts_centered is None:
        pts_centered = pts - c
    
    Tm = np.reshape(T, (3,4))
    t = Tm[:, 3]
    A = Tm[:, :3]
        
    pts_prime = np.dot(A, pts_centered.T) + (t + c_prime)[:,None]
        
    return pts_prime.T

atlas_nzs2_twoSides = {name: transform_points(T_final, pts=nzs, c=atlas_centroid, 
                                              c_prime=test_centroid).astype(np.int16) 
                       for name, nzs in atlas_nzs_twoSides.iteritems()}

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
    
    pts_prime = transform_points(T, pts_centered=pts2_centered[name], c_prime=test_centroid2)
    
    if '_' in name:
        l = labels_index[name[:-2]]
    else:
        l = labels_index[name]
    
    xs_prime = pts_prime[:,0]
    ys_prime = pts_prime[:,1]
    zs_prime = pts_prime[:,2]
    
    valid = (xs_prime >= 0) & (ys_prime >= 0) & (zs_prime >= 0) & \
            (xs_prime < test_xdim) & (ys_prime < test_ydim) & (zs_prime < test_zdim)
    
    if verbose:
        print 'nz', np.count_nonzero(valid) 
        
    assert np.count_nonzero(valid) > 0, 'No valid pixel after transform: %s' % name
    
    xs_prime_valid = xs_prime[valid].astype(np.int16)
    ys_prime_valid = ys_prime[valid].astype(np.int16)
    zs_prime_valid = zs_prime[valid].astype(np.int16)
    
    voxel_probs_valid = volume2_allLabels[l-1, ys_prime_valid, xs_prime_valid, zs_prime_valid] / 1e6
    score = voxel_probs_valid.sum()
    
    if num_samples is not None:
        # sample some voxels # this seems to make optimization more stable than using all voxels
    
        ii = np.random.choice(range(np.count_nonzero(valid)), num_samples, replace=False)

        dSdx = dSdxyz[l-1, 0, ys_prime_valid, xs_prime_valid, zs_prime_valid][ii]
        dSdy = dSdxyz[l-1, 1, ys_prime_valid, xs_prime_valid, zs_prime_valid][ii]
        dSdz = dSdxyz[l-1, 2, ys_prime_valid, xs_prime_valid, zs_prime_valid][ii]

        xss = xs_prime[valid].astype(np.float)[ii]
        yss = ys_prime[valid].astype(np.float)[ii]
        zss = zs_prime[valid].astype(np.float)[ii]
    else:
        # use all voxels    
        dSdx = dSdxyz[l-1, 0, ys_prime_valid, xs_prime_valid, zs_prime_valid]
        dSdy = dSdxyz[l-1, 1, ys_prime_valid, xs_prime_valid, zs_prime_valid]
        dSdz = dSdxyz[l-1, 2, ys_prime_valid, xs_prime_valid, zs_prime_valid]

        xss = xs_prime[valid].astype(np.float)
        yss = ys_prime[valid].astype(np.float)
        zss = zs_prime[valid].astype(np.float)

    #############################################
    
    dMdv = np.c_[dSdx, dSdy, dSdz, 
                 -dSdy*zss + dSdz*yss, 
                 dSdx*zss - dSdz*xss,
                 -dSdx*yss + dSdy*xss].sum(axis=0)

    if verbose:
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

############ align all landmarks ###########

params_dir = create_if_not_exists(atlasAlignParams_dir + '/' + stack)

history_len = 100
T0 = np.array([1,0,0,0,0,1,0,0,0,0,1,0])
# max_iter = 100
max_iter = 5000

# for name_of_interest in ['Pn_R']:
for name_of_interest in atlas_nzs_twoSides.keys():
    
    # set the rotation center of both atlas and test volume to the landmark centroid after affine projection
    
    atlas_centroid2 = atlas_nzs2_twoSides[name_of_interest].mean(axis=0)
    test_centroid2 = atlas_centroid2.copy()
    pts2_centered = {name: nzs - atlas_centroid2 for name, nzs in atlas_nzs2_twoSides.iteritems()}
    
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
                      np.mean(scores[iteration-2*history_len:iteration-history_len])) < 1e-4:
                break

        if s > score_best:
            best_gradient_descent_params = T
            score_best = s
            
    with open(params_dir + '/%(stack)s_%(name)s_transformUponAffineProjection.txt' % {'stack': stack, 'name': name_of_interest}, 
              'w') as f:
        f.write((' '.join(['%f']*12)+'\n') % tuple(best_gradient_descent_params))
        f.write((' '.join(['%f']*3)+'\n') % tuple(atlas_centroid2))
        f.write((' '.join(['%f']*3)+'\n') % tuple(test_centroid2))
        

############ Visualize ##############
        
parameters_allLandmarks = {}
atlas_centroid_allLandmarks = {}
test_centroid_allLandmarks = {}

for name in atlas_nzs_twoSides.keys():
    
    with open(atlasAlignParams_dir + '/%(stack)s/%(stack)s_%(name)s_transformUponAffineProjection.txt' % \
                        {'stack': stack, 'name': name}, 'r') as f:
        lines = f.readlines()
        params = np.array(map(float, lines[0].strip().split()))
        atlas_c = np.array(map(float, lines[1].strip().split()))
        test_c = np.array(map(float, lines[2].strip().split()))
    
    parameters_allLandmarks[name] = params
    atlas_centroid_allLandmarks[name] = atlas_c
    test_centroid_allLandmarks[name] = test_c
    

    
atlas_nzs_projected_to_test = {name: transform_points(parameters_allLandmarks[name], pts=nzs, 
                                                      c=atlas_centroid_allLandmarks[name], 
                                                      c_prime=test_centroid_allLandmarks[name]).astype(np.int16)
                               for name, nzs in atlas_nzs2_twoSides.iteritems()}

test_volume_atlas_projected = np.zeros(volume2_allLabels.shape[1:], np.int16)

for name in atlas_nzs_twoSides.keys():

    test_xs = atlas_nzs_projected_to_test[name][:,0]
    test_ys = atlas_nzs_projected_to_test[name][:,1]
    test_zs = atlas_nzs_projected_to_test[name][:,2]

    valid = (test_xs >= 0) & (test_ys >= 0) & (test_zs >= 0) & \
            (test_xs < test_xdim) & (test_ys < test_ydim) & (test_zs < test_zdim)

    atlas_xs = atlas_nzs_twoSides[name][:,0]
    atlas_ys = atlas_nzs_twoSides[name][:,1]
    atlas_zs = atlas_nzs_twoSides[name][:,2]
        
    test_volume_atlas_projected[test_ys[valid], test_xs[valid], test_zs[valid]] = \
    volume1[atlas_ys[valid], atlas_xs[valid], atlas_zs[valid]]
    
    
dm = DataManager(stack=stack)

section_bs_begin, section_bs_end = section_range_lookup[stack]
print section_bs_begin, section_bs_end

map_z_to_section = {}
for s in range(section_bs_begin, section_bs_end+1):
    for z in range(int(z_xy_ratio_downsampled*s) - volume_zmin, int(z_xy_ratio_downsampled*(s+1)) - volume_zmin + 1):
        map_z_to_section[z] = s
        
annotationsViz_rootdir = '/home/yuncong/csd395/CSHL_annotaionsIndividual3DShiftedViz/'
annotationsViz_dir = create_if_not_exists(annotationsViz_rootdir + '/' + stack)


for z in range(0, test_zdim, 10):

    dm.set_slice(map_z_to_section[z])
    dm._load_image(versions=['rgb-jpg'])
    viz = dm.image_rgb_jpg[::downsample_factor, ::downsample_factor][volume_ymin:volume_ymax+1, 
                                                                     volume_xmin:volume_xmax+1].copy()

    projected_cnts = find_contour_points(test_volume_atlas_projected[...,z])

    for label_ind, cnts in projected_cnts.iteritems():
        for cnt in cnts:
            cv2.polylines(viz, [cnt.astype(np.int)], True, tuple((colors[label_ind]*255).astype(np.int)), 2)

#     plt.figure(figsize=(10, 10));
#     plt.title('z = %d' % z)
#     plt.imshow(viz)
#     plt.show()
    
    cv2.imwrite(annotationsViz_dir + '/%(stack)s_%(sec)04d_annotaionsIndividual3DShiftedViz_z%(z)04d.jpg' % \
                {'stack': stack, 'sec': map_z_to_section[z], 'z': z}, 
                img_as_ubyte(viz[..., [2,1,0]]))