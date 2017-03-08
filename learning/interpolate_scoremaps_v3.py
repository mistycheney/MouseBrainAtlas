#! /usr/bin/env python

import numpy as np

import sys
import os

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *
from data_manager import *

from multiprocess import Pool
import time

from scipy.interpolate import RectBivariateSpline
from skimage.transform import resize

stack = sys.argv[1]
first_sec = int(sys.argv[2])
last_sec = int(sys.argv[3])
setting = int(sys.argv[4])

##################################################

sections_to_filenames = metadata_cache['sections_to_filenames'][stack]
anchor_fn = metadata_cache['anchor_fn'][stack]

for sec in range(first_sec, last_sec+1):

    actual_setting = resolve_actual_setting(setting=setting, stack=stack, sec=sec)

    sys.stderr.write('Section %d\n' % sec)

    fn = sections_to_filenames[sec]
    if is_invalid(fn): continue

    # output
    scoremaps_dir = create_if_not_exists(os.path.join(SCOREMAPS_ROOTDIR, stack,
                                 '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped' % \
                                 dict(stack=stack, fn=fn, anchor_fn=anchor_fn)))

    ## define grid, generate patches

    t = time.time()

    _, sample_locations_roi = DataManager.load_dnn_feature_locations(stack=stack, model_name='Sat16ClassFinetuned', fn=fn, anchor_fn=anchor_fn)

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

    probs_allClasses = {}
    for structure in all_known_structures:
        try:
            probs_allClasses[structure] = DataManager.load_sparse_scores(stack, fn=fn, anchor_fn=anchor_fn,
                                                                     structure=structure, setting=actual_setting)
        except Exception as e:
            sys.stderr.write('Patch predictions for %s do not exist.\n' % structure)

    structures = probs_allClasses.keys()

    sys.stderr.write('preprocess: %.2f seconds\n' % (time.time() - t)) # 3s

    def generate_score_map(structure):

        if structure == 'BackG':
            return None

        score_matrix = np.zeros((n_sample_x, n_sample_y))
        score_matrix[sample_location_indices[:,0], sample_location_indices[:,1]] = probs_allClasses[structure]

        spline = RectBivariateSpline(sample_locations_unique_xs/shrink_factor,
                                     sample_locations_unique_ys/shrink_factor,
                                     score_matrix,
                                     bbox=[interpolation_xmin/shrink_factor,
                                           interpolation_xmax/shrink_factor,
                                           interpolation_ymin/shrink_factor,
                                           interpolation_ymax/shrink_factor])

        t1 = time.time()
        dense_score_map = spline.ev(sample_locations_interpolatedArea_xs_matrix,
                                    sample_locations_interpolatedArea_ys_matrix)
        sys.stderr.write('evaluate spline: %.2f seconds\n' % (time.time() - t1)) # 5s for shrink_factor=4; doubling results in quadratic time reduction

        t1 = time.time()
        dense_score_map = resize(dense_score_map, (interpolation_h, interpolation_w)) # similar speed as rescale
#             dense_score_map = rescale(dense_score_map, shrink_factor)
        sys.stderr.write('scale up: %.2f seconds\n' % (time.time() - t1)) # 10s, very high penalty when multiprocessing

#             t = time.time()
        dense_score_map[dense_score_map < 1e-1] = 0
        dense_score_map[dense_score_map > 1.] = 1.
#             sys.stderr.write('threshold: %.2f seconds\n' % (time.time() - t))

        if np.count_nonzero(dense_score_map) < 1e5:
            sys.stderr.write('No %s is detected on section %d\n' % (structure, sec))
            return None

        t1 = time.time()

        scoremap_bp_filepath, scoremap_interpBox_filepath = \
        DataManager.get_scoremap_filepath(stack=stack, fn=fn, anchor_fn=anchor_fn, structure=structure,
                                          return_bbox_fp=True, setting=actual_setting)

        save_hdf(dense_score_map.astype(np.float16), scoremap_bp_filepath, complevel=5)
        np.savetxt(scoremap_interpBox_filepath,
               np.array((interpolation_xmin, interpolation_xmax, interpolation_ymin, interpolation_ymax))[None],
               fmt='%d')

        sys.stderr.write('save: %.2f seconds\n' % (time.time() - t1)) # 4s, very high penalty when multiprocessing


    t = time.time()

    pool = Pool(4) # 8 causes contention, resuls in high upscaling and dumping to disk time.
    _ = pool.map(generate_score_map, structures)
    pool.close()
    pool.join()

    sys.stderr.write('interpolate: %.2f seconds\n' % (time.time() - t)) # ~ 30 seconds / section
