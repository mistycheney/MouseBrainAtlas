#! /usr/bin/env python

# This must be before any other matplotlib imports
import matplotlib
matplotlib.use('Agg')

import sys
import os
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from registration_utilities import *
from annotation_utilities import *
from metadata import *
from data_manager import *

###########################################################################

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='')
parser.add_argument("stack_fixed", type=str, help="Fixed stack name")
parser.add_argument("stack_moving", type=str, help="Moving stack name")
parser.add_argument("warp_setting", type=int, help="Warp setting")
parser.add_argument("classifier_setting", type=int, help="classifier_setting")
parser.add_argument("-n", "--trial_num", type=int, help="number of trials", default=1)
args = parser.parse_args()

stack_fixed = args.stack_fixed
stack_moving = args.stack_moving
warp_setting = args.warp_setting
classifier_setting = args.classifier_setting
trial_num = args.trial_num

###################################################################

warp_properties = registration_settings.loc[warp_setting]
print warp_properties

upstream_warp_setting = warp_properties['upstream_warp_id']
if upstream_warp_setting == 'None':
    upstream_warp_setting = None
transform_type = warp_properties['transform_type']
terminate_thresh = warp_properties['terminate_thresh']
grad_computation_sample_number = warp_properties['grad_computation_sample_number']
grid_search_sample_number = warp_properties['grid_search_sample_number']
std_tx = warp_properties['std_tx']
std_ty = warp_properties['std_ty']
std_tz = warp_properties['std_tz']
std_theta_xy = np.deg2rad(warp_properties['std_theta_xy'])

MAX_ITER_NUM = 1000
HISTORY_LEN = 10
MAX_GRID_SEARCH_ITER_NUM = 30

lr1 = 10
lr2 = 0.1

#####################################################################

volume_moving, structure_to_label_moving, label_to_structure_moving = \
DataManager.load_original_volume_all_known_structures(stack=stack_moving, sided=True, volume_type='score')

volume_fixed, structure_to_label_fixed, label_to_structure_fixed = \
DataManager.load_original_volume_all_known_structures(stack=stack_fixed, classifier_setting=classifier_setting, 
                                                   sided=False, volume_type='score')

# label_mapping_m2f = {label_m: structure_to_label_fixed[convert_to_original_name(name_m)]
#                      for label_m, name_m in label_to_structure_moving.iteritems()}
label_mapping_m2f = {label_m: structure_to_label_fixed[convert_to_original_name(name_m)] 
                     for label_m, name_m in label_to_structure_moving.iteritems()
                     if name_m in ['7N_L', '7N_R', '12N', '5N_L', 'Pn_R', 'SNR_L', 
                                   'VLL_R', '7n_L', 'Tz_R', 'VCA_L', 'VCP_R', 'Sp5C_L', 'Sp5C_R']}

label_weights_m = {}
for label_m, name_m in label_to_structure_moving.iteritems():
    if 'surround' in name_m:
        label_weights_m[label_m] = 0
    else:
        label_weights_m[label_m] = 1
#         label_weights_m[label_m] = np.minimum(1e5 / volume_moving_structure_sizes[label_m], 1.)

################## Optimization ######################

aligner = Aligner4(volume_fixed, volume_moving, labelIndexMap_m2f=label_mapping_m2f)
aligner.set_centroid(centroid_m='volume_centroid', centroid_f='centroid_m')
# aligner.set_centroid(centroid_m='volume_centroid', centroid_f='volume_centroid')
# aligner.set_centroid(centroid_m='structure_centroid', centroid_f='centroid_m', indices_m=[name_to_label_moving['SNR_R']])

gradient_filepath_map_f = {ind_f: DataManager.get_volume_gradient_filepath_template(\
                            stack=stack_fixed, structure=label_to_structure_fixed[ind_f],
                            downscale=32, classifier_setting=classifier_setting)
                           for ind_m, ind_f in label_mapping_m2f.iteritems()}

aligner.load_gradient(gradient_filepath_map_f=gradient_filepath_map_f) # 120s = 2 mins
aligner.set_label_weights(label_weights=label_weights_m)

for trial_idx in range(trial_num):

    while True:
        try:
            T, scores = aligner.optimize(type=transform_type, max_iter_num=MAX_ITER_NUM, history_len=HISTORY_LEN, 
                                     terminate_thresh=terminate_thresh,
                                     grid_search_iteration_number=MAX_GRID_SEARCH_ITER_NUM,
                                     grid_search_sample_number=grid_search_sample_number,
                                     grad_computation_sample_number=grad_computation_sample_number,
                                     lr1=lr1, lr2=lr2,
                                     std_tx=std_tx, std_ty=std_ty, std_tz=std_tz, std_theta_xy=std_theta_xy)
            break

        except Exception as e:
            sys.stderr.write(e.message + '\n')

    params_fp = \
    DataManager.get_alignment_parameters_filepath(stack_m=stack_moving, stack_f=stack_fixed,
                                                  classifier_setting_m=classifier_setting,
                                                  classifier_setting_f=classifier_setting,
                                                  warp_setting=warp_setting,
                                                  trial_idx=trial_idx)
    DataManager.save_alignment_parameters(params_fp, T, 
                                          aligner.centroid_m, aligner.centroid_f,
                                          aligner.xdim_m, aligner.ydim_m, aligner.zdim_m, 
                                          aligner.xdim_f, aligner.ydim_f, aligner.zdim_f)
    upload_from_ec2_to_s3(params_fp)
    
    history_fp = DataManager.get_score_history_filepath(stack_m=stack_moving, stack_f=stack_fixed,
                                                          classifier_setting_m=classifier_setting,
                                                          classifier_setting_f=classifier_setting,
                                                          warp_setting=warp_setting,
                                                          trial_idx=trial_idx)
    bp.pack_ndarray_file(np.array(scores), history_fp)
    upload_from_ec2_to_s3(history_fp)

    score_plot_fp = \
    DataManager.get_alignment_score_plot_filepath(stack_m=stack_moving, stack_f=stack_fixed,
                                                         classifier_setting_m=classifier_setting,
                                                         classifier_setting_f=classifier_setting,
                                                         warp_setting=warp_setting,
                                                         trial_idx=trial_idx)
    fig = plt.figure();
    plt.plot(scores);
    plt.savefig(score_plot_fp, bbox_inches='tight')
    plt.close(fig)

    upload_from_ec2_to_s3(score_plot_fp)
