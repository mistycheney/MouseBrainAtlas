#! /usr/bin/env python

# This must be before any other matplotlib imports
import matplotlib
matplotlib.use('Agg')

import numpy as np

import sys
import os

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from registration_utilities import *
from metadata import *
from data_manager import *

import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import time

stack_fixed = sys.argv[1]
train_sample_scheme = int(sys.argv[2])
global_transform_scheme = int(sys.argv[3])
local_transform_scheme = int(sys.argv[4])
atlas_name = sys.argv[5]

# 1: no regularization, structures weight the same
# 2: with regularization, structures weight the same
# 3: no regularization, with surround
# 4: with regularization, with surround
# 5: no regularization, structure weight inversely prop to size
# 6: with regularization, structure weight inversely prop to size

###############################
# Set regularization weights

if local_transform_scheme == [1,3,5]:
    reg_weights = np.array([0.,0.,0.])
elif local_transform_scheme == 2:
    reg_weights = np.array([1e-6, 1e-6, 1e-6])
elif local_transform_scheme == 4:
    reg_weights = np.array([1e-4, 1e-4, 1e-4])

stack_moving = atlas_name

###########################
# All structures in atlas

paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', 'SNC', 'SNR', '3N', '4N',
                    'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
singular_structures = ['AP', '12N', 'RtTg', 'SC', 'IC']
structures = paired_structures + singular_structures

structures_sided = sum([[n] if n in singular_structures
                        else [convert_to_left_name(n), convert_to_right_name(n)]
                        for n in structures], [])

structures_sided_plus_surround = sum([[s, s+'_surround'] for s in structures_sided], [])

####################################
# Load moving
####################################

if local_transform_scheme in [1,2,5,6]:
    label_to_name_moving = {i+1: name for i, name in enumerate(sorted(structures_sided))}
elif local_transform_scheme in [3,4]:
    label_to_name_moving = {i+1: name for i, name in enumerate(sorted(structures_sided) + \
                                sorted([s+'_surround' for s in structures_sided]))}

name_to_label_moving = {n:l for l, n in label_to_name_moving.iteritems()}

#####################################

if local_transform_scheme in [3,4]:

    volume_moving = {name_to_label_moving[name_s]: \
    DataManager.load_transformed_volume(stack_m=stack_moving,
                                       type_m='score',
                                       stack_f=stack_fixed,
                                       type_f='score',
                                       downscale=32,
                                       train_sample_scheme_f=train_sample_scheme,
                                       global_transform_scheme=global_transform_scheme,
                                       label=name_s)
                     for name_s in structures_sided_plus_surround}

else:

    volume_moving = {name_to_label_moving[name_s]: \
    DataManager.load_transformed_volume(stack_m=stack_moving,
                                       type_m='score',
                                       stack_f=stack_fixed,
                                       type_f='score',
                                       downscale=32,
                                       train_sample_scheme_f=train_sample_scheme,
                                       global_transform_scheme=global_transform_scheme,
                                       label=name_s)
                     for name_s in structures_sided}

print volume_moving.values()[0].shape
print volume_moving.values()[0].dtype

####################################
# Load fixed
####################################

volume_fixed = {}
name_to_label_fixed = {}
c = 1 # label starts from 1
for name in sorted(structures):
    try:
        volume_fixed[c] = \
        DataManager.load_score_volume(stack=stack_fixed, label=name, downscale=32, train_sample_scheme=train_sample_scheme)
        # valid_names_f.append(name)
        name_to_label_fixed[name] = c
        c += 1
    except:
        sys.stderr.write('Score volume for %s does not exist.\n' % name)

label_to_name_fixed = {l: n for n, l in name_to_label_fixed.iteritems()}
print volume_fixed.values()[0].shape
print volume_fixed.values()[0].dtype

vol_fixed_ydim, vol_fixed_xdim, vol_fixed_zdim = volume_fixed.values()[0].shape

######################################

labelIndexMap_m2f = {}
for label_m, name_m in label_to_name_moving.iteritems():
    name_m_orig = convert_to_original_name(name_m)
    if name_m_orig in name_to_label_fixed:
        labelIndexMap_m2f[label_m] = name_to_label_fixed[name_m_orig]

#######################################

for name_s in structures_sided:

    print name_s

    label_m = name_to_label_moving[name_s]

    name_u = convert_to_unsided_name(name_s)
    if name_u not in name_to_label_fixed:
        sys.stderr.write('Ignore %s.\n' % name_s)
        continue
    else:
        label_f = name_to_label_fixed[name_u]

    trial_num = 3

    for trial_idx in range(trial_num):

        if local_transform_scheme in [3,4]:
            # consider two structures: x and x_surround

            name_s_surr = convert_to_surround_name(name_s)
            label_surr_m = name_to_label_moving[name_s_surr]

            aligner = Aligner4(volume_fixed, \
                            {label_m: volume_moving[label_m],
                            label_surr_m: volume_moving[label_surr_m]}, \
                            labelIndexMap_m2f={label_m: label_f, label_surr_m: label_f})
            label_weights_m = {label_m: 1, label_surr_m: -1}

        else:
            # consider one structure: x

            aligner = Aligner4(volume_fixed, {label_m: volume_moving[label_m]}, \
                               labelIndexMap_m2f={label_m: label_f})
            label_weights_m = {label_m: 1}

        # aligner.set_centroid(centroid_m='volume_centroid', centroid_f='volume_centroid')
        aligner.set_centroid(centroid_m='structure_centroid', centroid_f='centroid_m',
                             indices_m=[label_m])

        gradient_filepath_map_f = {ind_f: VOLUME_ROOTDIR + '/%(stack)s/score_volume_gradients/%(stack)s_down32_scoreVolume_%(name)s_trainSampleScheme_%(scheme)d_%%(suffix)s.bp' % \
                               {'stack': stack_fixed, 'name': label_to_name_fixed[ind_f], 'scheme':train_sample_scheme}
                               for ind_m, ind_f in labelIndexMap_m2f.iteritems()}

        aligner.load_gradient(gradient_filepath_map_f=gradient_filepath_map_f, indices_f=None)

        T, scores = aligner.optimize(type='rigid', max_iter_num=10000, history_len=50, terminate_thresh=1e-5,
                                     indices_m=None,
                                    grid_search_iteration_number=20,
                                     grid_search_sample_number=1000,
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
