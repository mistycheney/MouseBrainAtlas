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


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='')
parser.add_argument("stack_fixed", type=str, help="Fixed stack name")
parser.add_argument("stack_moving", type=str, help="Moving stack name")
parser.add_argument("warp_setting", type=str, help="Warp setting")
parser.add_argument("classifier_setting", type=str, help="classifier_setting")
parser.add_argument("--trial_idx", type=int, help="trial index", default=0)
args = parser.parse_args()

stack_fixed = args.stack_fixed
stack_moving = args.stack_moving
warp_setting = args.warp_setting
classifier_setting = args.classifier_setting
trial_idx = args.trial_idx

if warp_setting == 1:
    global_transform_scheme = 1
    local_transform_scheme = 2
else:
    raise Exception('Warp setting not recognized.')

volume_moving, structure_to_label_moving, label_to_structure_moving = \
DataManager.load_score_volume_all_known_structures(stack=stack_moving, sided=True)

volume_fixed, structure_to_label_fixed, label_to_structure_fixed = \
DataManager.load_score_volume_all_known_structures(stack=stack_fixed, classifier_setting=classifier_setting)

label_mapping_m2f = {label_m: structure_to_label_fixed[convert_to_original_name(name_m)]
                     for label_m, name_m in label_to_structure_moving.iteritems()}

label_weights_m = {}
for label_m, name_m in label_to_structure_moving.iteritems():
    if 'surround' in name_m:
        label_weights_m[label_m] = 0
    else:
        label_weights_m[label_m] = 1
#         label_weights_m[label_m] = np.minimum(1e5 / volume_moving_structure_sizes[label_m], 1.)

################## Optimization ######################

aligner = Aligner4(volume_fixed, volume_moving, labelIndexMap_m2f=label_mapping_m2f)

aligner.set_centroid(centroid_m='volume_centroid', centroid_f='volume_centroid')
# aligner.set_centroid(centroid_m='structure_centroid', centroid_f='centroid_m', indices_m=[name_to_label_moving['SNR_R']])

gradient_filepath_map_f = {ind_f: DataManager.get_score_volume_gradient_filepath_template(\
                            stack=stack_fixed, structure=label_to_structure_fixed[ind_f],
                            downscale=32, setting=classifier_setting)
                           for ind_m, ind_f in label_mapping_m2f.iteritems()}

aligner.load_gradient(gradient_filepath_map_f=gradient_filepath_map_f) # 120s = 2 mins
aligner.set_label_weights(label_weights=label_weights_m)

trial_num = 1

for trial_idx in range(trial_num):

    if global_transform_scheme == 1:

        T, scores = aligner.optimize(type='affine', max_iter_num=1000, history_len=10, terminate_thresh=1e-4,
                                     indices_m=None,
                                    grid_search_iteration_number=30,
                                     grid_search_sample_number=100,
                                     grad_computation_sample_number=1e5,
                                     lr1=10, lr2=0.1,
                                    std_tx=50, std_ty=50, std_tz=100, std_theta_xy=np.deg2rad(10))

    elif global_transform_scheme == 2:

        T, scores = aligner.optimize(type='rigid', max_iter_num=1000, history_len=10, terminate_thresh=1e-4,
                                     indices_m=None,
                                    grid_search_iteration_number=30,
                                     grid_search_sample_number=100,
                                     grad_computation_sample_number=1e5,
                                     lr1=10, lr2=0.1,
                                    std_tx=50, std_ty=50, std_tz=100, std_theta_xy=np.deg2rad(10))

    params_fp = \
    DataManager.get_alignment_parameters_filepath(stack_moving=stack_moving, stack_fixed=stack_fixed,
                                                  classifier_setting_m=classifier_setting,
                                                  classifier_setting_f=classifier_setting,
                                                  warp_setting=warp_setting,
                                                  trial_idx=0)

    DataManager.save_alignment_parameters(params_fp, T,
                                          aligner.centroid_m, aligner.centroid_f,
                                          aligner.xdim_m, aligner.ydim_m, aligner.zdim_m,
                                          aligner.xdim_f, aligner.ydim_f, aligner.zdim_f)

    score_plot_fp = \
    DataManager.get_alignment_score_plot_filepath(stack_moving=stack_moving, stack_fixed=stack_fixed,
                                                     classifier_setting_m=classifier_setting,
                                                     classifier_setting_f=classifier_setting,
                                                     warp_setting=warp_setting,
                                                     trial_idx=0)
    fig = plt.figure();
    plt.plot(scores);
    plt.savefig(score_plot_fp, bbox_inches='tight')
    plt.close(fig)
