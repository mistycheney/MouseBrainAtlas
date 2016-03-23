#! /usr/bin/env python

import sys
import os

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *

import pandas as pd
import numpy as np

from joblib import Parallel, delayed
import time

from scipy.interpolate import RectBivariateSpline
from skimage.transform import resize

stack = sys.argv[1]
first_sec = int(sys.argv[2])
last_sec = int(sys.argv[3])

##################################################

labels = ['BackG', '5N', '7n', '7N', '12N', 'Pn', 'VLL', 
          '6N', 'Amb', 'R', 'Tz', 'RtTg', 'LRt', 'LC', 'AP', 'sp5']

label_dict = dict([(l,i) for i, l in enumerate(labels)])

patches_rootdir = '/home/yuncong/CSHL_data_patches'

# scoremaps_rootdir = '/home/yuncong/CSHL_scoremaps_lossless_inceptionModel'
scoremaps_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_scoremaps_lossless_svm'
if not os.path.exists(scoremaps_rootdir):
    os.makedirs(scoremaps_rootdir)
    
# predictions_rootdir = '/home/yuncong/CSHL_patch_predictions_inceptionModel'
predictions_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_patch_predictions_svm'
if not os.path.exists(predictions_rootdir):
    os.makedirs(predictions_rootdir)

    
first_bs_sec, last_bs_sec = section_range_lookup[stack]

table_filepath = os.path.join(patches_rootdir, '%(stack)s_indices_allROIs_allSections.h5'%{'stack':stack})
indices_allROIs_allSections = pd.read_hdf(table_filepath, 'indices_allROIs_allSections')
grid_parameters = pd.read_hdf(table_filepath, 'grid_parameters')

patch_size, stride, w, h = grid_parameters.tolist()
half_size = patch_size/2

ys, xs = np.meshgrid(np.arange(half_size, h-half_size, stride), np.arange(half_size, w-half_size, stride),
                 indexing='xy')

sample_locations = np.c_[xs.flat, ys.flat]

for sec in range(first_sec, last_sec+1):

    if sec not in indices_allROIs_allSections.columns:
        continue

    print sec

    indices_roi = indices_allROIs_allSections[sec]['roi1']

    predictions_dir = os.path.join(predictions_rootdir, stack, '%04d'%sec)
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    scoremaps_dir = os.path.join(scoremaps_rootdir, stack, '%04d'%sec)
    if not os.path.exists(scoremaps_dir):
        os.makedirs(scoremaps_dir)

    ## define grid, generate patches

    t = time.time()

    sample_locations_roi = sample_locations[indices_roi]

    ## interpolate

    interpolation_xmin, interpolation_ymin = sample_locations_roi.min(axis=0)
    interpolation_xmax, interpolation_ymax = sample_locations_roi.max(axis=0)
    interpolation_w = interpolation_xmax - interpolation_xmin + 1
    interpolation_h = interpolation_ymax - interpolation_ymin + 1

    ##### sample_locations_roi + scores to dense_score_map #####

    shrink_factor = 4 # do interpolation on a smaller grid, then resize to original dimension

    sample_locations_unique_xs = np.unique(sample_locations_roi[:,0])
    sample_locations_unique_ys = np.unique(sample_locations_roi[:,1])

    n_sample_x = sample_locations_unique_xs.size
    n_sample_y = sample_locations_unique_ys.size

    index_x = dict([(j,i) for i,j in enumerate(sample_locations_unique_xs)])
    index_y = dict([(j,i) for i,j in enumerate(sample_locations_unique_ys)])
    sample_location_indices = np.asarray([(index_x[x], index_y[y]) for x, y in sample_locations_roi])

    sample_locations_interpolatedArea_ys_matrix, \
    sample_locations_interpolatedArea_xs_matrix = np.meshgrid(range(interpolation_ymin/shrink_factor, 
                                                                    interpolation_ymax/shrink_factor), 
                                                              range(interpolation_xmin/shrink_factor, 
                                                                    interpolation_xmax/shrink_factor), 
                                                              indexing='ij')

    dataset = '%(stack)s_%(sec)04d_roi1' % {'stack': stack, 'sec': sec}

    probs_allClasses = dict([(label, np.load(predictions_dir + '/%(dataset)s_%(label)s_scores.npy' % \
                                             {'dataset': dataset, 'label': label}))
                             for label in labels[1:]])

    sys.stderr.write('preprocess: %.2f seconds\n' % (time.time() - t))

    def generate_score_map(label):

        if label == 'BackG':
            return None

#             probs = np.load(predictions_dir + '/%(dataset)s_%(label)s_scores.npy'% {'dataset': dataset, 'label': label})
#             probs = probs_allClasses[label]

        score_matrix = np.zeros((n_sample_x, n_sample_y))
        score_matrix[sample_location_indices[:,0], sample_location_indices[:,1]] = probs_allClasses[label]

        spline = RectBivariateSpline(sample_locations_unique_xs/shrink_factor, 
                                     sample_locations_unique_ys/shrink_factor, 
                                     score_matrix, 
                                     bbox=[interpolation_xmin/shrink_factor, 
                                           interpolation_xmax/shrink_factor, 
                                           interpolation_ymin/shrink_factor, 
                                           interpolation_ymax/shrink_factor])

#             t = time.time()
        dense_score_map = spline.ev(sample_locations_interpolatedArea_xs_matrix, 
                                    sample_locations_interpolatedArea_ys_matrix)
#             sys.stderr.write('evaluate spline: %.2f seconds\n' % (time.time() - t))

#             t = time.time()
        dense_score_map = resize(dense_score_map, (interpolation_h, interpolation_w)) # similar speed as rescale
#             dense_score_map = rescale(dense_score_map, shrink_factor)
#             sys.stderr.write('scale up: %.2f seconds\n' % (time.time() - t))

#             t = time.time()
        dense_score_map[dense_score_map < 1e-1] = 0
#             sys.stderr.write('threshold: %.2f seconds\n' % (time.time() - t))

        if np.count_nonzero(dense_score_map) < 1e5:
            sys.stderr.write('No %s is detected on section %d\n' % (label, sec))
            return None

#             t = time.time()
#             bp.pack_ndarray_file(dense_score_map.astype(np.float32), 
#                                    os.path.join(scoremaps_dir, '%(dataset)s_denseScoreMapLossless_%(label)s.bp' % \
#                                                 {'dataset': dataset, 'label': label}))
        save_hdf(dense_score_map.astype(np.float16), 
                 os.path.join(scoremaps_dir, '%(dataset)s_denseScoreMapLossless_%(label)s.hdf' % \
                                            {'dataset': dataset, 'label': label}),
                complevel=5)
#             sys.stderr.write('save: %.2f seconds\n' % (time.time() - t))

        np.savetxt(os.path.join(scoremaps_dir, '%(dataset)s_denseScoreMapLossless_%(label)s_interpBox.txt' % \
                                    {'dataset': dataset, 'label': label}),
               np.array((interpolation_xmin, interpolation_xmax, interpolation_ymin, interpolation_ymax))[None], 
               fmt='%d')

    t = time.time()

    # if too many disk saves are simultaneous, they will be conflicting, so split into two sessions
    _ = Parallel(n_jobs=16)(delayed(generate_score_map)(l) for l in labels[1:len(labels)/2])
    _ = Parallel(n_jobs=16)(delayed(generate_score_map)(l) for l in labels[len(labels)/2:])

    sys.stderr.write('interpolate: %.2f seconds\n' % (time.time() - t)) # ~20 seconds
