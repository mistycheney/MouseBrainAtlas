#! /usr/bin/env python

import numpy as np

import sys
import os

from matplotlib.path import Path

from collections import deque
from itertools import izip, chain

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *

sys.path.append('/home/yuncong/Brain/preprocess/morphsnakes')
import morphsnakes

from shapely.geometry import Polygon

from enum import Enum

class PolygonType(Enum):
    CLOSED = 'closed'
    OPEN = 'open'
    TEXTURE = 'textured'
    TEXTURE_WITH_CONTOUR = 'texture with contour'
    DIRECTION = 'directionality'
    
from skimage.morphology import binary_closing, disk, binary_dilation, binary_erosion, remove_small_holes
from skimage.measure import find_contours, grid_points_in_poly, subdivide_polygon, approximate_polygon

import time
from collections import defaultdict

from registration_utilities import *

#######################

volume_dir = '/oasis/projects/nsf/csd395/yuncong/CSHL_volumes/'
scoremaps_rootdir = '/home/yuncong/csd395/CSHL_scoremaps_lossless_svm_Sat16ClassFinetuned_v3//'
# autoAnnotations_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_autoAnnotations_snake'
autoAnnotations_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_labelings_losslessAlignCropped/'
autoAnnotationViz_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_autoAnnotationsViz_snake'

########################

stack = sys.argv[1]

autoAnnotationViz_dir = create_if_not_exists(autoAnnotationViz_rootdir + '/' + stack)

first_bs_sec, last_bs_sec = section_range_lookup[stack]
first_detect_sec, last_detect_sec = detect_bbox_range_lookup[stack]

localAdjusted_volume = bp.unpack_ndarray_file(volume_dir + '/%(stack)s/%(stack)s_localAdjustedVolume.bp'%{'stack':stack})
print localAdjusted_volume.shape

(volume_xmin, volume_xmax, volume_ymin, volume_ymax, volume_zmin, volume_zmax) = \
    np.loadtxt(os.path.join(volume_dir, '%(stack)s/%(stack)s_scoreVolume_limits.txt' % {'stack': stack}), dtype=np.int)


volume_landmark_names_unsided = ['12N', '5N', '6N', '7N', '7n', 'AP', 'Amb', 'LC',
                                 'LRt', 'Pn', 'R', 'RtTg', 'Tz', 'VLL', 'sp5']
linear_landmark_names_unsided = ['outerContour']

labels_unsided = volume_landmark_names_unsided + linear_landmark_names_unsided
labels_unsided_indices = dict((j, i+1) for i, j in enumerate(labels_unsided))  # BackG always 0

labelMap_unsidedToSided = {'12N': ['12N'],
                            '5N': ['5N_L', '5N_R'],
                            '6N': ['6N_L', '6N_R'],
                            '7N': ['7N_L', '7N_R'],
                            '7n': ['7n_L', '7n_R'],
                            'AP': ['AP'],
                            'Amb': ['Amb_L', 'Amb_R'],
                            'LC': ['LC_L', 'LC_R'],
                            'LRt': ['LRt_L', 'LRt_R'],
                            'Pn': ['Pn_L', 'Pn_R'],
                            'R': ['R_L', 'R_R'],
                            'RtTg': ['RtTg'],
                            'Tz': ['Tz_L', 'Tz_R'],
                            'VLL': ['VLL_L', 'VLL_R'],
                            'sp5': ['sp5'],
                           'outerContour': ['outerContour']}

labelMap_sidedToUnsided = {n: nu for nu, ns in labelMap_unsidedToSided.iteritems() for n in ns}

from itertools import chain
labels_sided = list(chain(*(labelMap_unsidedToSided[name_u] for name_u in labels_unsided)))
labels_sided_indices = dict((j, i+1) for i, j in enumerate(labels_sided)) # BackG always 0


downsample_factor = 16

section_thickness = 20 # in um
xy_pixel_distance_lossless = 0.46
xy_pixel_distance_tb = xy_pixel_distance_lossless * 32 # in um, thumbnail

xy_pixel_distance_downsampled = xy_pixel_distance_lossless * downsample_factor
z_xy_ratio_downsampled = section_thickness / xy_pixel_distance_downsampled

# build annotation volume
section_bs_begin, section_bs_end = section_range_lookup[stack]
print section_bs_begin, section_bs_end

map_z_to_section = {}
for s in range(section_bs_begin, section_bs_end+1):
    for z in range(int(z_xy_ratio_downsampled*s) - volume_zmin, 
                   int(z_xy_ratio_downsampled*(s+1)) - volume_zmin + 1):
        map_z_to_section[z] = s
        
        
available_labels_sided = [labels_sided[i-1] for i in np.unique(localAdjusted_volume) if i > 0]
available_labels_unsided = set([labelMap_sidedToUnsided[name] for name in available_labels_sided ])
        
#####################################################
        
dm = DataManager(stack=stack)

init_cnts_allSecs = load_initial_contours(initCnts_dir=volume_dir,
                                          stack=stack,
                                          test_volume_atlas_projected=localAdjusted_volume,
                                          z_xy_ratio_downsampled=z_xy_ratio_downsampled,
                                          volume_limits=(volume_xmin, volume_xmax, volume_ymin, volume_ymax, volume_zmin, volume_zmax),
                                          labels=['BackG'] + labels_sided,
                                         force=False)

    
first_sec = int(sys.argv[2])
last_sec = int(sys.argv[3])

for sec in range(first_sec, last_sec+1):
    
    t = time.time()
    
    autoAnnotations_dir = create_if_not_exists(autoAnnotations_rootdir + '/' + stack + '/' + '%04d'%sec)
    
    dm.set_slice(sec)
    dm._load_image(versions=['rgb-jpg'], force_reload=True)
    cropped_img = dm.image_rgb_jpg[::8, ::8]
        
    sys.stderr.write('load image %f s\n' % (time.time() - t))
        
    print '\n'
    print sec
    
    #########################
        
    init_cnts = init_cnts_allSecs[sec]

    labels_exist = set(init_cnts.keys())

    valid_labels = set(available_labels_sided) & labels_exist
    print valid_labels
    
    if len(valid_labels) == 0:
        sys.stderr.write('No valid labels exist.\n')
        continue

    new_res = []
    
    for l in valid_labels:
        
        print l
        
        t = time.time()
        
        dense_scoremap_lossless = pad_scoremap(stack, sec, labelMap_sidedToUnsided[l], scoremaps_rootdir=scoremaps_rootdir, 
                                               bg_size=(dm.image_height, dm.image_width))
        
        if dense_scoremap_lossless is None:
            dense_scoremap_lossless = np.zeros((dm.image_height, dm.image_width), np.float32)
        
        scoremap0 = dense_scoremap_lossless[::8, ::8]
        scoremap = scoremap0.copy()
                    
        sys.stderr.write('load scoremap %f s\n' % (time.time() - t))
                    
        t = time.time()
        
        scoremap_height, scoremap_width = scoremap.shape[:2]

        init_cnt = init_cnts[l]

        if len(init_cnt) < 3:
            sys.stderr.write('initial contour has less than 3 vertices. \n')
            continue
            
#         init_cnt = contour_to_concave_hull(init_cnt, levelset=None, alpha=.01)
        
#         init_cnt = approximate_polygon(init_cnt, 5).astype(np.int)
#         init_cnt = subdivide_polygon(init_cnt, 5).astype(np.int)
        
        init_cnt_xmin, init_cnt_ymin = init_cnt.min(axis=0)
        init_cnt_xmax, init_cnt_ymax = init_cnt.max(axis=0)
        init_cnt_height, init_cnt_width = (init_cnt_ymax - init_cnt_ymin + 1, init_cnt_xmax - init_cnt_xmin + 1)
        init_cnt_cx, init_cnt_cy = np.mean(init_cnt, axis=0)
        
#         in_grid = grid_points_in_poly((init_cnt_height, init_cnt_width), init_cnt - (init_cnt_xmin, init_cnt_ymin))
        
        init_cnt_poly = Path(init_cnt)
        init_cnt_bbox_xs, init_cnt_bbox_ys = np.meshgrid(range(init_cnt_xmin, init_cnt_xmax+1), 
                                                         range(init_cnt_ymin, init_cnt_ymax+1))
        grid_points = np.c_[init_cnt_bbox_xs.flat, init_cnt_bbox_ys.flat]
        is_inside = init_cnt_poly.contains_points(grid_points)
        inside_points = grid_points[is_inside]
        
#         shift_best = np.array((0,0))
#         score_max = 0
#         for xshift in range(-200, 200, 10):
#             for yshift in range(-200, 200, 10):
#                 shifted_ys = inside_points[:,1] + int(yshift)
#                 shifted_xs = inside_points[:,0] + int(xshift)
#                 valid = (shifted_ys >= 0) & (shifted_ys < scoremap_height-1) & (shifted_xs >= 0) & (shifted_xs < scoremap_width-1)
#                 shifted_ys = shifted_ys[valid]
#                 shifted_xs = shifted_xs[valid]
#                 score = scoremap0[shifted_ys, shifted_xs].mean()
#                 if score_max < score:
#                     score_max = score
#                     shift_best = np.array([int(xshift), int(yshift)])
        
#         print 'initial shift', shift_best, score_max
                
#         if l == 'RtTg':
#             shift_best = (0, 0)

        shift_best = (0, 0)
        
#         init_cnt = init_cnt + shift_best
        init_cnt_xmin = max(init_cnt_xmin + shift_best[0], 0)
        init_cnt_ymin = max(init_cnt_ymin + shift_best[1], 0)
        init_cnt_xmax = min(init_cnt_xmax + shift_best[0], scoremap_width-1)
        init_cnt_ymax = min(init_cnt_ymax + shift_best[1], scoremap_height-1)
        
#         init_cnt_cx += shift_best[0]
#         init_cnt_cy += shift_best[1]
        inside_points += shift_best
        inside_points = inside_points[(inside_points[:,0] > init_cnt_xmin) & \
                                      (inside_points[:,0] < init_cnt_xmax) & \
                                      (inside_points[:,1] > init_cnt_ymin) & \
                                      (inside_points[:,1] < init_cnt_ymax)]
        
        sys.stderr.write('initial shift %f s\n' % (time.time() - t))
        
        
        roi_margin = max(200, (400-min(init_cnt_height, init_cnt_width))/2)
        # should be set to the largest landmark diameter (so that it is ok even if contour is placed at the end of it)
        
        roi_xmin, roi_ymin = (max(0, init_cnt_xmin - roi_margin), max(0, init_cnt_ymin - roi_margin))
        roi_xmax, roi_ymax = (min(scoremap_width-1, init_cnt_xmax + roi_margin), min(scoremap_height-1, init_cnt_ymax + roi_margin))
        roi_height, roi_width = (roi_ymax + 1 - roi_ymin, roi_xmax + 1 - roi_xmin)
                
        inside_points_inroi = inside_points - (roi_xmin, roi_ymin)
        
#         print 'init_cnt', init_cnt_xmin, init_cnt_ymin, init_cnt_xmax, init_cnt_ymax
#         print 'inside_points mins', inside_points.min(axis=0)
#         print 'inside_points maxs', inside_points.max(axis=0)
#         print 'inside_points_inroi mins', inside_points_inroi.min(axis=0)
#         print 'inside_points_inroi maxs', inside_points_inroi.max(axis=0)
#         print roi_xmin, roi_ymin, roi_xmax, roi_ymax, roi_width, roi_height
        
        scoremap_roi = scoremap[roi_ymin:roi_ymax+1, roi_xmin:roi_xmax+1] 
                
        ######## landmark specific settings ########
        
        score_thresh = .3
                    
        if l == 'sp5':
            smoothing = 1
        else:
            smoothing = 3

        alphas = np.arange(.20, 0.009, -.01)
            
        #############################################
    
        scoremap_thresholded = scoremap_roi > score_thresh
        
        scoremap_thresholded_padded = np.zeros((roi_height + 100, roi_width + 100), np.bool)
        scoremap_thresholded_padded[50:-50, 50:-50] = scoremap_thresholded[:]
#         scoremap_thresholded_padded = binary_closing(scoremap_thresholded_padded, disk(10))

#         for _ in range(3):
#             scoremap_thresholded_padded = binary_dilation(scoremap_thresholded_padded, disk(3))
#         for _ in range(3):
#             scoremap_thresholded_padded = binary_erosion(scoremap_thresholded_padded, disk(3))

        scoremap_thresholded_padded = remove_small_holes(scoremap_thresholded_padded, 1000)
#         scoremap_thresholded_padded = remove_small_objects(scoremap_thresholded_padded, 100)
        scoremap_thresholded = scoremap_thresholded_padded[50:-50, 50:-50][:]

        init_levelset = np.zeros((roi_height, roi_width))
        init_levelset[inside_points_inroi[:,1], inside_points_inroi[:,0]] = 1.
        
        t = time.time()
        
        msnake = morphsnakes.MorphACWE(scoremap_thresholded.astype(np.float), smoothing=smoothing, lambda1=1., lambda2=1.)
        
        msnake.levelset = init_levelset.copy()
        # levelset values are either 1.0 or 0.0
        
        dq = deque([None, None])
        for i in range(1000): 
            
            # at stable stage, the levelset (thus contour) will oscilate, 
            # so instead of comparing to previous levelset, must compare to the one before the previous
            oneBefore_levelset = dq.popleft()
            
            if i > 10:
#                 print np.count_nonzero(msnake.levelset - oneBefore_levelset)
                if np.count_nonzero(msnake.levelset - oneBefore_levelset) < 3:
                    break

            dq.append(msnake.levelset)
        
            msnake.step()
        
        # in the final levelset, inside could be 0. or 1., hard to say        
        edge_arr = np.r_[msnake.levelset[:,0], msnake.levelset[:,-1], msnake.levelset[0], msnake.levelset[-1]]        
        pos_edge_num = np.count_nonzero(edge_arr)
        bool_arr = msnake.levelset.astype(np.bool)
        
        if pos_edge_num < len(edge_arr) - pos_edge_num:
            # inside is 1.
            mean_inside_score = scoremap_roi[bool_arr].mean()
        else:
            # inside is 0.
            mean_inside_score = scoremap_roi[~bool_arr].mean()
            msnake.levelset = 1. - msnake.levelset
        
        # after this, all inside pixels have value 1.
            
        print 'mean inside score:', mean_inside_score
        print 'area:', np.count_nonzero(bool_arr)
        print 'snake iteration:', i
        
        if mean_inside_score < .3:
            continue
        
        sys.stderr.write('snake completes %f s\n' % (time.time() - t))
        
        t = time.time()
    
        new_cnts = find_contours(msnake.levelset, 0.5)
        new_cnts = [c[:, ::-1] for c in new_cnts]
#         cnt_is_dense = True
        
#         if len(new_cnts) == 1:
#             new_cnt = new_cnts[0]
        if len(new_cnts) == 0:
            sys.stderr.write('No contour detected from snake levelset.\n')
            continue

        all_cnt_points = np.concatenate(new_cnts)
        
        new_cnt_subsampled, auto_alpha = contour_to_concave_hull(all_cnt_points.astype(np.int), levelset=msnake.levelset, alphas=alphas)

#         alpha_stats[l].append(auto_alpha)
        
        if new_cnt_subsampled is None:
            continue
    
        area_lowerlim = 1000
        area = Polygon(new_cnt_subsampled).area
        if area < area_lowerlim:
            sys.stderr.write('Concave hull area %d is too small.\n' % area)
            continue

        new_cnt_subsampled = new_cnt_subsampled + (roi_xmin, roi_ymin)
        
        
        sys.stderr.write('concave hull finishes %f s\n' % (time.time() - t))

        
        new_lm = {}
        new_lm['label'] = l
        new_lm['vertices'] = new_cnt_subsampled.astype(np.int) * 8
        new_lm['labelPos'] = new_lm['vertices'].mean(axis=0)
        new_lm['refVertices'] = np.array(init_cnt).copy() * 8
        new_lm['subtype'] = PolygonType.CLOSED
            
        new_res.append(new_lm)
        
    ######################################
        
    timestamp = datetime.datetime.now().strftime("%m%d%Y%H%M%S")

    autoAnnotation_filepath = autoAnnotations_dir + '/%(stack)s_%(sec)04d_autoAnnotate_%(timestamp)s_consolidated.pkl' % \
                            {'stack': stack, 'sec': sec, 'timestamp': timestamp}

    pickle.dump(new_res, open(autoAnnotation_filepath, 'w'))
        

# pickle.dump(alpha_stats, open('/home/yuncong/alpha_stats_%s_%04d_%04d.pkl' % (stack, first_sec, last_sec), 'w'))
