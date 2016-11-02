#! /usr/bin/env python

# This must be before any other matplotlib imports
import matplotlib
matplotlib.use('Agg')

import numpy as np

import sys
import os

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from registration_utilities import parallel_where_binary, Aligner4
from metadata import *
from data_manager import *

import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import time

stack_fixed = sys.argv[1]
train_sample_scheme = int(sys.argv[2])
global_transform_scheme = int(sys.argv[3])
local_transform_scheme = int(sys.argv[4])

# 1: no regularization
# 2: with regularization
# 3: no regularization, with surround
# 4: with regularization, with surround

if local_transform_scheme == 1:
    reg_weights = np.array([0.,0.,0.])
elif local_transform_scheme == 2:
    reg_weights = np.array([1e-4, 1e-4, 1e-4])
elif local_transform_scheme == 3:
    reg_weights = np.array([0.,0.,0.])
elif local_transform_scheme == 4:
    reg_weights = np.array([1e-4, 1e-4, 1e-4])

stack_moving = 'atlas_on_MD589'

paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', 'SNC', 'SNR', '3N', '4N',
                    'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
singular_structures = ['AP', '12N', 'RtTg', 'SC', 'IC']
structures = paired_structures + singular_structures

label_to_name_fixed = {i+1: name for i, name in enumerate(sorted(structures))}
name_to_label_fixed = {n:l for l, n in label_to_name_fixed.iteritems()}

structures_sided = sum([[n] if n in singular_structures else [convert_to_left_name(n), convert_to_right_name(n)] for n in structures], [])

if local_transform_scheme == 1 or local_transform_scheme == 2:

    label_to_name_moving = {i+1: name for i, name in enumerate(structures_sided)}
    name_to_label_moving = {n:l for l, n in label_to_name_moving.iteritems()}

elif local_transform_scheme == 3 or local_transform_scheme == 4:

    structures_sided_plus_surround = sum([[s, s+'_surround'] for s in structures_sided], [])

    label_to_name_moving = {i+1: name for i, name in enumerate(structures_sided_plus_surround)}
    name_to_label_moving = {n:l for l, n in label_to_name_moving.iteritems()}

def convert_to_original_name(name):
    return name.split('_')[0]

labelIndexMap_m2f = {}
for label_m, name_m in label_to_name_moving.iteritems():
    labelIndexMap_m2f[label_m] = name_to_label_fixed[convert_to_original_name(name_m)]

label_weights_m = {}
for label_m, name_m in label_to_name_moving.iteritems():
    if 'surround' in name_m:
        if local_transform_scheme == 1 or local_transform_scheme == 2:
            label_weights_m[label_m] = 0
        elif local_transform_scheme == 3 or local_transform_scheme == 4:
            label_weights_m[label_m] = -1
    else:
        label_weights_m[label_m] = 1

# Load fixed volumes

volume_fixed = {name_to_label_fixed[name]: DataManager.load_score_volume(stack=stack_fixed, label=name, downscale=32, train_sample_scheme=train_sample_scheme)
               for name in structures}

print volume_fixed.values()[0].shape
print volume_fixed.values()[0].dtype

vol_fixed_xmin, vol_fixed_ymin, vol_fixed_zmin = (0,0,0)
vol_fixed_ymax, vol_fixed_xmax, vol_fixed_zmax = np.array(volume_fixed.values()[0].shape) - 1
vol_fixed_xdim = vol_fixed_xmax + 1 - vol_fixed_xmin
vol_fixed_ydim = vol_fixed_ymax + 1 - vol_fixed_ymin
vol_fixed_zdim = vol_fixed_zmax + 1 - vol_fixed_zmin


# Load moving volumes

if local_transform_scheme == 1 or local_transform_scheme == 2:

    volume_moving = {name_to_label_moving[name_s]: DataManager.load_transformed_volume(stack_m='atlas_on_MD589',
                                                                                       type_m='score',
                                                                                       stack_f=stack_fixed,
                                                                                       type_f='score',
                                                                                       downscale=32,
                                                                                       train_sample_scheme_f=train_sample_scheme,
                                                                                       label=name_s)
                     for name_s in structures_sided}

elif local_transform_scheme == 3 or local_transform_scheme == 4:

    volume_moving = {name_to_label_moving[name_s]: DataManager.load_transformed_volume(stack_m='atlas_on_MD589',
                                                                                       type_m='score',
                                                                                       stack_f=stack_fixed,
                                                                                       type_f='score',
                                                                                       downscale=32,
                                                                                       train_sample_scheme_f=train_sample_scheme,
                                                                                       label=name_s)
                     for name_s in structures_sided_plus_surround}


print volume_moving.values()[0].shape
print volume_moving.values()[0].dtype

vol_moving_xmin, vol_moving_ymin, vol_moving_zmin = (0,0,0)
vol_moving_ymax, vol_moving_xmax, vol_moving_zmax = np.array(volume_moving.values()[0].shape) - 1


#####################################################

for name_s in structures_sided:
# for name_s in ['4N_L']:

    print name_s

    trial_num = 5

    for trial_idx in range(trial_num):

        if local_transform_scheme == 1 or local_transform_scheme == 2:

            aligner = Aligner4(volume_fixed, {name_to_label_moving[name_s]:
                                              volume_moving[name_to_label_moving[name_s]]}, \
                               labelIndexMap_m2f={name_to_label_moving[name_s]:
                                                  name_to_label_fixed[convert_name_to_unsided(name_s)]})

        elif local_transform_scheme == 3 or local_transform_scheme == 4:

            aligner = Aligner4(volume_fixed, {name_to_label_moving[name_s]: volume_moving[name_to_label_moving[name_s]],
                                             name_to_label_moving[name_s+'_surround']: volume_moving[name_to_label_moving[name_s+'_surround']]}, \
                            labelIndexMap_m2f={name_to_label_moving[name_s]: name_to_label_fixed[convert_name_to_unsided(name_s)],
                                              name_to_label_moving[name_s+'_surround']: name_to_label_fixed[convert_to_original_name(name_s+'_surround')]})


        # aligner.set_centroid(centroid_m='volume_centroid', centroid_f='volume_centroid')
        aligner.set_centroid(centroid_m='structure_centroid', centroid_f='centroid_m',
                             indices_m=[name_to_label_moving[name_s]])

        gradient_filepath_map_f = {ind_f: VOLUME_ROOTDIR + '/%(stack)s/score_volume_gradients/%(stack)s_down32_scoreVolume_%(label)s_trainSampleScheme_%(scheme)d_%%(suffix)s.bp' % \
                               {'stack': stack_fixed, 'label': label_to_name_fixed[ind_f], 'scheme':train_sample_scheme}
                               for ind_m, ind_f in labelIndexMap_m2f.iteritems()}

        aligner.load_gradient(gradient_filepath_map_f=gradient_filepath_map_f, indices_f=None)

        T, scores = aligner.optimize(type='rigid', max_iter_num=10000, history_len=50, terminate_thresh=1e-5,
                                     indices_m=None,
                                    grid_search_iteration_number=30,
                                     grid_search_sample_number=10000,
                                     grad_computation_sample_number=1e5,
                                     lr1=10, lr2=0.1,
                                    label_weights=label_weights_m,
                                    std_tx=50, std_ty=50, std_tz=100, std_theta_xy=np.deg2rad(10),
                                    reg_weights=reg_weights,
                                    epsilon=1e-8)

        ########################################################

        params_fp = DataManager.get_local_alignment_parameters_filepath(stack_moving=stack_moving,
                                                                    moving_volume_type='score',
                                                                    stack_fixed=stack_fixed,
                                                                    fixed_volume_type='score',
                                                                    train_sample_scheme=train_sample_scheme,
                                                                    global_transform_scheme=global_transform_scheme,
                                                                    local_transform_scheme=local_transform_scheme,
                                                                   label=name_s,
                                                                   trial_idx=trial_idx)

        DataManager.save_alignment_parameters(params_fp,
                                              T, aligner.centroid_m, aligner.centroid_f,
                                              aligner.xdim_m, aligner.ydim_m, aligner.zdim_m,
                                              aligner.xdim_f, aligner.ydim_f, aligner.zdim_f)

        score_plot_fp = DataManager.get_local_alignment_score_plot_filepath(stack_moving=stack_moving,
                                                                        moving_volume_type='score',
                                                                        stack_fixed=stack_fixed,
                                                                        fixed_volume_type='score',
                                                                        train_sample_scheme=train_sample_scheme,
                                                                        global_transform_scheme=global_transform_scheme,
                                                                        local_transform_scheme=local_transform_scheme,
                                                                       label=name_s,
                                                                       trial_idx=trial_idx)

        fig = plt.figure();
        plt.plot(scores);
        plt.savefig(score_plot_fp, bbox_inches='tight')
        plt.close(fig)

#####################################################


from registration_utilities import transform_volume, transform_points, find_contour_points

volumes_annotation = {'MD594': bp.unpack_ndarray_file(DataManager.get_transformed_volume_filepath(stack_m='MD594', type_m='annotation',
                                                stack_f=stack_fixed, type_f='score',
                                                downscale=32, train_sample_scheme_f=1)),

                      'MD589': bp.unpack_ndarray_file(DataManager.get_transformed_volume_filepath(stack_m='MD589', type_m='annotation',
                                                stack_f=stack_fixed, type_f='score',
                                                downscale=32, train_sample_scheme_f=1))}

name_to_label_annotation = DataManager.load_annotation_volume_nameToLabel('MD589', downscale=32)
label_to_name_annotation = {l: n for n, l in name_to_label_annotation.iteritems()}

stack_colors = {'MD589': (255,0,0), 'MD594': (0,255,0)}

first_sec, last_sec = metadata_cache['section_limits'][stack_fixed]

xmin_vol_f, xmax_vol_f, ymin_vol_f, ymax_vol_f, zmin_vol_f, zmax_vol_f = np.loadtxt('/home/yuncong/csd395/CSHL_volumes2/%(stack_fixed)s/score_volumes/%(stack_fixed)s_down32_scoreVolume_7N_bbox.txt' %\
          dict(stack_fixed=stack_fixed)).astype(np.int)
print xmin_vol_f, xmax_vol_f, ymin_vol_f, ymax_vol_f, zmin_vol_f, zmax_vol_f

downsample_factor = 32
xy_pixel_distance_downsampled = xy_pixel_distance_lossless * downsample_factor
voxel_z_size = section_thickness / xy_pixel_distance_downsampled


viz_dir = create_if_not_exists(DataManager.get_local_alignment_viz_dir(stack_moving=stack_moving,
                                                        stack_fixed=stack_fixed,
                                                        moving_volume_type='score',
                                                        fixed_volume_type='score',
                                                        train_sample_scheme=train_sample_scheme,
                                                        global_transform_scheme=global_transform_scheme,
                                                        local_transform_scheme=local_transform_scheme))

# Transforming each volume (only relevant structure is activated) according to computed local transforms

volume_m_aligned_to_f_allNames = {'MD589': {}, 'MD594': {}}

for name_s in structures_sided:

    print name_s

    try:
        tx_params, centroid_m, centroid_f, xdim_m, ydim_m, zdim_m, xdim_f, ydim_f, zdim_f = \
        DataManager.load_local_alignment_parameters(stack_moving=stack_moving,
                                                    moving_volume_type='score',
                                                    stack_fixed=stack_fixed,
                                                    fixed_volume_type='score',
                                                    train_sample_scheme=train_sample_scheme,
                                                    global_transform_scheme=global_transform_scheme,
                                                    local_transform_scheme=local_transform_scheme,
                                                   label=name_s,
                                                   trial_idx=1)

    except Exception as e:

        sys.stderr.write(e.message + '\n')

        tx_params = (1,0,0,0,0,1,0,0,0,0,1,0)
        centroid_m = (0,0,0)
        centroid_f = (0,0,0)
        xdim_f, ydim_f, zdim_f = (vol_fixed_xdim, vol_fixed_ydim, vol_fixed_zdim)


    for stack, volume_annotation in volumes_annotation.iteritems():
        volume_m_aligned_to_f_allNames[stack][name_s] = \
        transform_volume(vol=volume_annotation==name_to_label_annotation[name_s],
                           global_params=tx_params,
                           centroid_m=centroid_m,
                           centroid_f=centroid_f,
                          xdim_f=xdim_f,
                          ydim_f=ydim_f,
                          zdim_f=zdim_f)

for sec in range(first_sec, last_sec+1):
# for sec in range(281, 282):

    if metadata_cache['sections_to_filenames'][stack_fixed][sec] in ['Placeholder', 'Rescan', 'Nonexisting']:
            continue

    img_fn = DataManager.get_image_filepath(stack=stack_fixed, section=sec, resol='thumbnail', version='cropped_tif')
    img = imread(img_fn)

#         img_fn = DataManager.get_scoremap_viz_filepath(stack=stack_fixed, section=sec, label='7N', train_sample_scheme=train_sample_scheme)
#         img = imread(img_fn)[::4, ::4]

    viz = img.copy()

    z = voxel_z_size * (sec - 1) - zmin_vol_f

    ##############################################

    for stack, x in volume_m_aligned_to_f_allNames.iteritems():
        for name_s, volume_m_aligned_to_f in x.iteritems():

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

                    # put label texts
                    label_pos = cnt_m.mean(axis=0).astype(np.int)
                    cv2.putText(viz, convert_name_to_unsided(name_s), tuple(label_pos),
                                cv2.FONT_HERSHEY_DUPLEX, .5, ((0,0,0)), 1)

    viz_fn = os.path.join(viz_dir, '%(stack_moving)s_over_%(stack_fixed)s_%(sec)04d.jpg' % \
          {'stack_moving': stack_moving, 'stack_fixed': stack_fixed, 'sec': sec})
    imsave(viz_fn, viz)
