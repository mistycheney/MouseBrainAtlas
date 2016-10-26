#! /usr/bin/env python

# This must be before any other matplotlib imports
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

import sys
import os

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from registration_utilities import parallel_where_binary, Aligner4
from metadata import *
from data_manager import *


from joblib import Parallel, delayed
import time

stack_fixed = sys.argv[1]
train_sample_scheme = int(sys.argv[2])
global_transform_scheme = int(sys.argv[3])

stack_moving = 'atlas_on_MD589'

paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', 'SNC', 'SNR', '3N', '4N',
                    'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
singular_structures = ['AP', '12N', 'RtTg', 'SC', 'IC']
structures = paired_structures + singular_structures

label_to_name_fixed = {i+1: name for i, name in enumerate(sorted(structures))}
name_to_label_fixed = {n:l for l, n in label_to_name_fixed.iteritems()}

label_to_name_moving = {i+1: name for i, name in enumerate(sorted(structures))}
name_to_label_moving = {n:l for l, n in label_to_name_moving.iteritems()}

labelIndexMap_m2f = {}
for label_m, name_m in label_to_name_moving.iteritems():
    labelIndexMap_m2f[label_m] = name_to_label_fixed[name_m]

volume_fixed = {name_to_label_fixed[name]: bp.unpack_ndarray_file(os.path.join(VOLUME_ROOTDIR, '%(stack)s/score_volumes/%(stack)s_down32_scoreVolume_%(name)s_trainSampleScheme_%(scheme)d.bp' % \
                                                    {'stack': stack_fixed, 'name': name, 'scheme':train_sample_scheme}))
               for name in structures}

print volume_fixed.values()[0].shape

vol_fixed_xmin, vol_fixed_ymin, vol_fixed_zmin = (0,0,0)
vol_fixed_ymax, vol_fixed_xmax, vol_fixed_zmax = volume_fixed.values()[0].shape


volume_moving = {name_to_label_moving[name]: bp.unpack_ndarray_file(os.path.join(VOLUME_ROOTDIR, '%(stack)s/score_volumes/%(stack)s_down32_scoreVolume_%(name)s.bp' % \
                                                    {'stack': stack_moving, 'name': name}))
               for name in structures}

print volume_moving.values()[0].shape

vol_moving_xmin, vol_moving_ymin, vol_moving_zmin = (0,0,0)
vol_moving_ymax, vol_moving_xmax, vol_moving_zmax = volume_moving.values()[0].shape

#############################

aligner = Aligner4(volume_fixed, volume_moving, labelIndexMap_m2f=labelIndexMap_m2f)

aligner.set_centroid(centroid_m='volume_centroid', centroid_f='volume_centroid')
# aligner.set_centroid(centroid_m='structure_centroid', centroid_f='centroid_m', indices_m=[name_to_label_moving['SNR_R']])

gradient_filepath_map_f = {ind_f: VOLUME_ROOTDIR + '/%(stack)s/score_volume_gradients/%(stack)s_down32_scoreVolume_%(label)s_trainSampleScheme_%(scheme)d_%%(suffix)s.bp' % \
                           {'stack': stack_fixed, 'label': label_to_name_fixed[ind_f], 'scheme':train_sample_scheme}
                           for ind_m, ind_f in labelIndexMap_m2f.iteritems()}

aligner.load_gradient(gradient_filepath_map_f=gradient_filepath_map_f, indices_f=None)

# largely the same optimization path regardless of the starting condition

# For rigid,
# grad_computation_sample_number = 1e5 is desired
# grid_search_iteration_number and grid_search_sample_number seem to be unimportant as well, set to 100
# lr1=10, lr2=.1 is best

# For affine,
# lr2 = .001 is too slow; 0.1 rises faster than 0.01
# lr1 does not matter
# plateus around iteration 100, but keep rising afterwards.
# grad_computation_sample_number does not make a difference

if global_transform_scheme == 1:

    T, scores = aligner.optimize(type='affine', max_iter_num=1000, history_len=50, terminate_thresh=1e-4,
    #                              indices_m=[name_to_label_fixed['SNR_L']],
                                 indices_m=None,
                                grid_search_iteration_number=30,
                                 grid_search_sample_number=100,
                                 grad_computation_sample_number=1e5,
                                 lr1=10, lr2=0.1)

elif global_transform_scheme == 2:

    T, scores = aligner.optimize(type='rigid', max_iter_num=1000, history_len=50, terminate_thresh=1e-4,
    #                              indices_m=[name_to_label_fixed['SNR_L']],
                                 indices_m=None,
                                grid_search_iteration_number=50,
                                 grid_search_sample_number=100,
                                 grad_computation_sample_number=1e5,
                                 lr1=10, lr2=0.1)

# print T.reshape((3,4))
# plt.plot(scores);
# print max(scores), scores[-1]

params_fp = DataManager.get_global_alignment_parameters_filepath(stack_moving=stack_moving,
                                                                stack_fixed=stack_fixed,
                                                                train_sample_scheme=train_sample_scheme,
                                                                global_transform_scheme=global_transform_scheme)


# with open(atlasAlignParams_dir + '/%(stack_moving)s_down32_scoreVolume_to_%(stack_fixed)s_down32_scoreVolume_trainSampleScheme_%(scheme)d_globalTxScheme_%(gtf_sheme)d_parameters.txt' % \
#           {'stack_moving': stack_moving, 'stack_fixed': stack_fixed, 'scheme':train_sample_scheme, 'gtf_sheme':global_transform_scheme}, 'w') as f:
with open(params_fp, 'w') as f:

    f.write(array_to_one_liner(T))
    f.write(array_to_one_liner(aligner.centroid_m))
    f.write(array_to_one_liner([aligner.xdim_m, aligner.ydim_m, aligner.zdim_m]))
    f.write(array_to_one_liner(aligner.centroid_f))
    f.write(array_to_one_liner([aligner.xdim_f, aligner.ydim_f, aligner.zdim_f]))


score_plot_fp = DataManager.get_global_alignment_score_plot_filepath(stack_moving=stack_moving,
                                                                    stack_fixed=stack_fixed,
                                                                    train_sample_scheme=train_sample_scheme,
                                                                    global_transform_scheme=global_transform_scheme)

fig = plt.figure();
plt.plot(scores);
plt.savefig(score_plot_fp, bbox_inches='tight')
plt.close(fig)

#####################################

stacks_annotation = ['MD589', 'MD594']

params_fp = DataManager.get_global_alignment_parameters_filepath(stack_moving=stack_moving,
                                                                stack_fixed=stack_fixed,
                                                                train_sample_scheme=train_sample_scheme,
                                                                global_transform_scheme=global_transform_scheme)

with open(params_fp, 'r') as f:

    lines = f.readlines()

    global_params = one_liner_to_arr(lines[0], float)
    centroid_m = one_liner_to_arr(lines[1], float)
    xdim_m, ydim_m, zdim_m  = one_liner_to_arr(lines[2], int)
    centroid_f = one_liner_to_arr(lines[3], float)
    xdim_f, ydim_f, zdim_f  = one_liner_to_arr(lines[4], int)


volumes_annotation = {'MD594':bp.unpack_ndarray_file('/home/yuncong/csd395/CSHL_atlasAlignParams_atlas_v2/MD594_to_MD589/MD594_down32_annotationVolume_alignedTo_MD589_down32_annotationVolume.bp'),
                      'MD589': bp.unpack_ndarray_file(VOLUME_ROOTDIR + '/MD589/MD589_down32_annotationVolume.bp')}

from registration_utilities import transform_points

annotation_volumes_volume_m_aligned_to_f = {}

for stack, volume_annotation in volumes_annotation.iteritems():
    all_indices_m = set(np.unique(volume_annotation)) - {0}
    nzvoxels_m_temp = {i: parallel_where_binary(volume_annotation==i) for i in all_indices_m}
    # "_temp" is appended to avoid name conflict with module level variable defined in registration.py

    nzs_m_aligned_to_f = {ind_m: transform_points(global_params, pts=nzs_m,
                                              c=centroid_m, c_prime=centroid_f).astype(np.int16)
                      for ind_m, nzs_m in nzvoxels_m_temp.iteritems()}

    volume_m_aligned_to_f = np.zeros((ydim_f, xdim_f, zdim_f), np.int)

    for ind_m in nzs_m_aligned_to_f.iterkeys():

        xs_f, ys_f, zs_f = nzs_m_aligned_to_f[ind_m].T

        valid = (xs_f >= 0) & (ys_f >= 0) & (zs_f >= 0) & \
        (xs_f < xdim_f) & (ys_f < ydim_f) & (zs_f < zdim_f)

        xs_m, ys_m, zs_m = nzvoxels_m_temp[ind_m].T

        volume_m_aligned_to_f[ys_f[valid], xs_f[valid], zs_f[valid]] = \
        volume_annotation[ys_m[valid], xs_m[valid], zs_m[valid]]

    del nzs_m_aligned_to_f

    annotation_volumes_volume_m_aligned_to_f[stack] = volume_m_aligned_to_f

xmin_vol_f, xmax_vol_f, ymin_vol_f, ymax_vol_f, zmin_vol_f, zmax_vol_f = np.loadtxt('/home/yuncong/csd395/CSHL_volumes2/%(stack_fixed)s/score_volumes/%(stack_fixed)s_down32_scoreVolume_7N_bbox.txt' %\
          dict(stack_fixed=stack_fixed)).astype(np.int)
print xmin_vol_f, xmax_vol_f, ymin_vol_f, ymax_vol_f, zmin_vol_f, zmax_vol_f


from registration_utilities import find_contour_points

downsample_factor = 32
xy_pixel_distance_downsampled = xy_pixel_distance_lossless * downsample_factor
voxel_z_size = section_thickness / xy_pixel_distance_downsampled

viz_dir = create_if_not_exists(DataManager.get_global_alignment_viz_filepath(stack_moving=stack_moving,
                                                        stack_fixed=stack_fixed,
                                                        train_sample_scheme=train_sample_scheme,
                                                        global_transform_scheme=global_transform_scheme))

first_sec, last_sec = metadata_cache['section_limits'][stack_fixed]

stack_colors = {'MD589': (255,0,0), 'MD594': (0,255,0)}

for sec in range(first_sec, last_sec+1):

    if metadata_cache['sections_to_filenames'][stack_fixed][sec] in ['Placeholder', 'Rescan', 'Nonexisting']:
        continue

    img_fn = DataManager.get_image_filepath(stack=stack_fixed, section=sec, resol='thumbnail', version='cropped_tif')
    img = imread(img_fn)

    viz = img.copy()

    z = voxel_z_size * (sec - 1) - zmin_vol_f

    # Find fixed volume annotation contours
#     contours_f_on_volume = find_contour_points(volume_fixed[..., int(z)])
#     contours_f_on_cropped = {i: [cnt + (xmin_vol_f, ymin_vol_f) for cnt in cnts] for i, cnts in contours_f_on_volume.iteritems()}

    # Find moving volume annotation contours

    for stack, volume_m_aligned_to_f in annotation_volumes_volume_m_aligned_to_f.iteritems():
        contours_m_alignedTo_f_on_volume = find_contour_points(volume_m_aligned_to_f[..., int(z)])
        contours_m_alignedTo_f_on_cropped = {i: [cnt + (xmin_vol_f, ymin_vol_f) for cnt in cnts]
                                             for i, cnts in contours_m_alignedTo_f_on_volume.iteritems()}

    #     # Draw fixed volume annotation contours
    #     for ind_f, cnts_f in contours_f_on_cropped.iteritems():
    #         for cnt_f in cnts_f:
    #             cv2.polylines(viz, [cnt_f.astype(np.int)], True, (0,255,0), 2)

        # Draw moving volume annotation contours
        for ind_m, cnts_m in contours_m_alignedTo_f_on_cropped.iteritems():
            for cnt_m in cnts_m:
                cv2.polylines(viz, [cnt_m.astype(np.int)], True, stack_colors[stack], 2)

    viz_fn = os.path.join(viz_dir, '%(stack_moving)s_to_%(stack_fixed)s_%(sec)04d.jpg' % \
          {'stack_moving': stack_moving, 'stack_fixed': stack_fixed, 'sec': sec})
    imsave(viz_fn, viz)
