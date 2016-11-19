#! /usr/bin/env python

import numpy as np

import sys
import os

#sys.path.append(os.environ['REPO_DIR'] + '/utilities')
sys.path.append('/home/ubuntu/MouseBrainAtlas/utilities')
from utilities2015 import *
from metadata import *
from data_manager import *

from joblib import Parallel, delayed
import time

from scipy.interpolate import RectBivariateSpline
from skimage.transform import resize

stack = sys.argv[1]
first_sec = int(sys.argv[2])
last_sec = int(sys.argv[3])

##################################################

paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', 'SNC', 'SNR', '3N', '4N',
                    'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
singular_structures = ['AP', '12N', 'RtTg', 'SC', 'IC']

structures = paired_structures + singular_structures

# filenames_to_sections, sections_to_filenames = DataManager.load_sorted_filenames(stack)
# first_sec, last_sec = DataManager.load_cropbox(stack)[4:]
# anchor_fn = DataManager.load_anchor_filename(stack)

sections_to_filenames = metadata_cache['sections_to_filenames'][stack]
# first_sec, last_sec = metadata_cache['section_limits'][stack]
anchor_fn = metadata_cache['anchor_fn'][stack]

train_sample_scheme = 1

for sec in range(first_sec, last_sec+1):

    sys.stderr.write('Section %d\n' % sec)

    fn = sections_to_filenames[sec]
    if fn in ['Nonexisting', 'Rescan', 'Placeholder']:
        continue

    # output
    scoremaps_dir = os.path.join(SCOREMAPS_ROOTDIR, stack,
                                 '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped' % \
                                 dict(stack=stack, fn=fn, anchor_fn=anchor_fn))
    create_if_not_exists(scoremaps_dir)

    ## define grid, generate patches

    t = time.time()

    locations_fn = PATCH_FEATURES_ROOTDIR + '/%(stack)s/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_patch_locations.txt' % dict(stack=stack, fn=fn, anchor_fn=anchor_fn)

    with open(locations_fn, 'r') as f:
        sample_locations_roi = np.array([map(int, line.split()[1:]) for line in f.readlines()])

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

    probs_allClasses = {label: DataManager.load_sparse_scores(stack, fn=fn, anchor_fn=anchor_fn,
                                          label=label, train_sample_scheme=train_sample_scheme)
                            for label in structures}

    sys.stderr.write('preprocess: %.2f seconds\n' % (time.time() - t))

    def generate_score_map(label):

        if label == 'BackG':
            return None

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

        t1 = time.time()
        dense_score_map = resize(dense_score_map, (interpolation_h, interpolation_w)) # similar speed as rescale
#             dense_score_map = rescale(dense_score_map, shrink_factor)
        sys.stderr.write('scale up: %.2f seconds\n' % (time.time() - t1))

#             t = time.time()
        dense_score_map[dense_score_map < 1e-1] = 0
        dense_score_map[dense_score_map > 1.] = 1.
#             sys.stderr.write('threshold: %.2f seconds\n' % (time.time() - t))

        if np.count_nonzero(dense_score_map) < 1e5:
            sys.stderr.write('No %s is detected on section %d\n' % (label, sec))
            return None

        scoremap_bp_filepath, scoremap_interpBox_filepath = DataManager.get_scoremap_filepath(stack=stack, fn=fn, anchor_fn=anchor_fn, label=label,
                                                                    return_bbox_fp=True, train_sample_scheme=train_sample_scheme)

        save_hdf(dense_score_map.astype(np.float16), scoremap_bp_filepath, complevel=5)
        np.savetxt(scoremap_interpBox_filepath,
                   np.array((interpolation_xmin, interpolation_xmax, interpolation_ymin, interpolation_ymax))[None],
                   fmt='%d')

    t = time.time()

    _ = Parallel(n_jobs=8)(delayed(generate_score_map)(l) for l in structures)

    sys.stderr.write('interpolate: %.2f seconds\n' % (time.time() - t)) # ~ 30 seconds / section
