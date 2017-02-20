#! /usr/bin/env python

# This must be before any other matplotlib imports
import matplotlib
matplotlib.use('Agg')

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from registration_utilities import *
from annotation_utilities import *
from metadata import *
from data_manager import *

import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='')
parser.add_argument("stack_fixed", type=str, help="Fixed stack name")
parser.add_argument("stack_moving", type=str, help="Moving stack name")
parser.add_argument("warp_setting", type=int, help="Warp setting")
parser.add_argument("classifier_setting", type=int, help="classifier_setting")
parser.add_argument("--trial_idx", type=int, help="trial index", default=0)
args = parser.parse_args()

stack_fixed = args.stack_fixed
stack_moving = args.stack_moving
warp_setting = args.warp_setting
classifier_setting = args.classifier_setting
trial_idx = args.trial_idx

if warp_setting == 1:
    upstream_warp_setting = None
    transform_type = 'affine'
elif warp_setting == 2:
    upstream_warp_setting = 1
    transform_type = 'rigid'
else:
    raise Exception('Warp setting not recognized.')

volume_fixed, structure_to_label_fixed, label_to_structure_fixed = \
DataManager.load_score_volume_all_known_structures(stack=stack_fixed, classifier_setting=classifier_setting)

volume_moving = DataManager.load_transformed_volume_all_known_structures(stack_m=stack_moving, stack_f=stack_fixed,
                                                                         classifier_setting_m=classifier_setting,
                                                                         classifier_setting_f=classifier_setting,
                                                                         warp_setting=upstream_warp_setting, sided=True)

structure_to_label_moving = {s: l+1 for l, s in enumerate(sorted(volume_moving.keys()))}
label_to_structure_moving = {l+1: s for l, s in enumerate(sorted(volume_moving.keys()))}
volume_moving = {structure_to_label_moving[s]: v for s, v in volume_moving.items()}

label_mapping_m2f = {label_m: structure_to_label_fixed[convert_to_original_name(name_m)]
                     for label_m, name_m in label_to_structure_moving.iteritems()}

# 1: no regularization, structures weight the same
# 2: with regularization, structures weight the same
# 3: no regularization, with surround
# 4: with regularization, with surround
# 5: no regularization, structure weight inversely prop to size
# 6: with regularization, structure weight inversely prop to size
#
# if local_transform_scheme == [1,3,5]:
#     reg_weights = np.array([0.,0.,0.])
# elif local_transform_scheme == 2:
#     reg_weights = np.array([1e-6, 1e-6, 1e-6])
# elif local_transform_scheme == 4:
#     reg_weights = np.array([1e-4, 1e-4, 1e-4])

gradient_filepath_map_f = {ind_f: DataManager.get_score_volume_gradient_filepath_template(\
                            stack=stack_fixed, structure=label_to_structure_fixed[ind_f],
                                setting=classifier_setting)
                           for ind_m, ind_f in label_mapping_m2f.iteritems()}

for structure in all_known_structures_sided:

    try:

        label_mapping_m2f_one_structure = {label_m: label_f for label_m, label_f in label_mapping_m2f.iteritems()
                                           if label_to_structure_moving[label_m] == structure}

        volume_moving_one_structure = {l: v for l, v in volume_moving.items()
                                       if label_to_structure_moving[l] == structure}

        aligner = Aligner4(volume_fixed, volume_moving_one_structure,
                           labelIndexMap_m2f=label_mapping_m2f_one_structure)

        aligner.set_centroid(centroid_m='structure_centroid', centroid_f='centroid_m',
                             indices_m=[structure_to_label_moving[structure]])

        aligner.load_gradient(gradient_filepath_map_f=gradient_filepath_map_f) # 120s = 2 mins

        T, scores = aligner.optimize(type=transform_type,
                                     max_iter_num=1000, history_len=50, terminate_thresh=1e-5,
                                     indices_m=None,
                                    grid_search_iteration_number=20,
                                     grid_search_sample_number=1000,
                                     grad_computation_sample_number=1e5,
                                     lr1=10, lr2=0.1,
                                    std_tx=50, std_ty=50, std_tz=100, std_theta_xy=np.deg2rad(10),
                                    epsilon=1e-8)

        params_fp = \
        DataManager.get_alignment_parameters_filepath(stack_m=stack_moving, stack_f=stack_fixed,
                                                      classifier_setting_m=classifier_setting,
                                                      classifier_setting_f=classifier_setting,
                                                      warp_setting=warp_setting,
                                                      param_suffix=structure,
                                                      trial_idx=0)

        DataManager.save_alignment_parameters(params_fp, T,
                                              aligner.centroid_m, aligner.centroid_f,
                                              aligner.xdim_m, aligner.ydim_m, aligner.zdim_m,
                                              aligner.xdim_f, aligner.ydim_f, aligner.zdim_f)

        score_plot_fp = \
        DataManager.get_alignment_score_plot_filepath(stack_m=stack_moving, stack_f=stack_fixed,
                                                             classifier_setting_m=classifier_setting,
                                                             classifier_setting_f=classifier_setting,
                                                             warp_setting=warp_setting,
                                                      param_suffix=structure,
                                                             trial_idx=0)
        fig = plt.figure();
        plt.plot(scores);
        plt.savefig(score_plot_fp, bbox_inches='tight')
        plt.close(fig)

    except Exception as e:
        sys.stderr.write('%s\n' % e)
        sys.stderr.write('Error transforming volume %s.\n' % structure)
