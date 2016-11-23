#! /usr/bin/env python

import numpy as np

import sys
import os

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from metadata import *
from data_manager import *

import time

stack_fixed = sys.argv[1]
train_sample_scheme = int(sys.argv[2])
global_transform_scheme = int(sys.argv[3])
trial_idx =  int(sys.argv[4])

stack_moving = 'atlas_on_MD589'

############## Generate fixed-stack section images with aligned atlas overlay ################

global_params, centroid_m, centroid_f, xdim_m, ydim_m, zdim_m, xdim_f, ydim_f, zdim_f = \
DataManager.load_global_alignment_parameters(stack_moving=stack_moving,
                                                    stack_fixed=stack_fixed,
                                                    train_sample_scheme=train_sample_scheme,
                                                    global_transform_scheme=global_transform_scheme,
                                                    trial_idx=trial_idx)

from registration_utilities import transform_volume

volumes_annotation = {'MD594': bp.unpack_ndarray_file(DataManager.get_transformed_volume_filepath(stack_m='MD594', type_m='annotation',
                                                stack_f='MD589', type_f='annotation',
                                                downscale=32, global_transform_scheme=global_transform_scheme)),
                      'MD589': bp.unpack_ndarray_file(DataManager.get_annotation_volume_filepath(stack='MD589', downscale=32))}

annotation_volumes_volume_m_aligned_to_f = {}

for stack, volume_annotation in volumes_annotation.iteritems():

    annotation_volumes_volume_m_aligned_to_f[stack] = transform_volume(vol=volume_annotation,
                                                                       global_params=global_params,
                                                                       centroid_m=centroid_m,
                                                                       centroid_f=centroid_f,
                                                                      xdim_f=xdim_f,
                                                                      ydim_f=ydim_f,
                                                                      zdim_f=zdim_f)

    output_fn = DataManager.get_transformed_volume_filepath(stack_m=stack, type_m='annotation',
                                                stack_f=stack_fixed, type_f='score',
                                                downscale=32, train_sample_scheme_f=train_sample_scheme)

    create_if_not_exists(os.path.dirname(output_fn))

    bp.pack_ndarray_file(annotation_volumes_volume_m_aligned_to_f[stack], output_fn)


xmin_vol_f, xmax_vol_f, ymin_vol_f, ymax_vol_f, zmin_vol_f, zmax_vol_f = \
DataManager.load_score_volume_bbox(stack=stack_fixed, label='7N', downscale=32)
print xmin_vol_f, xmax_vol_f, ymin_vol_f, ymax_vol_f, zmin_vol_f, zmax_vol_f


from registration_utilities import find_contour_points

downsample_factor = 32
xy_pixel_distance_downsampled = xy_pixel_distance_lossless * downsample_factor
voxel_z_size = section_thickness / xy_pixel_distance_downsampled

viz_dir = create_if_not_exists(DataManager.get_global_alignment_viz_dir(stack_moving=stack_moving,
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
