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
parser.add_argument("detector_id", type=int, help="detector_id")
parser.add_argument("-n", "--trial_num", type=int, help="number of trials", default=1)
parser.add_argument("-s", "--structures", type=str, help="structures")
args = parser.parse_args()

stack_fixed = args.stack_fixed
stack_moving = args.stack_moving
warp_setting = args.warp_setting
detector_id = args.detector_id
trial_num = args.trial_num
if hasattr(args, "structures"):
    structures = json.loads(args.structures)
else:
    structures = all_known_structures_sided

###########################################################################

volume_fixed, structure_to_label_fixed, label_to_structure_fixed = \
DataManager.load_original_volume_all_known_structures(stack=stack_fixed, prep_id=2, detector_id=detector_id, 
                                                      sided=False, volume_type='score')

gradient_filepath_map_f = {ind_f: \
                           DataManager.get_volume_gradient_filepath_template(\
                            stack=stack_fixed, structure=struct_f, prep_id=2, detector_id=detector_id)
                           for ind_f, struct_f in label_to_structure_fixed.iteritems()}


warp_properties = registration_settings.loc[warp_setting]
print warp_properties

upstream_warp_setting = warp_properties['upstream_warp_id']
if upstream_warp_setting == 'None':
    upstream_warp_setting = None
else:
    upstream_warp_setting = int(upstream_warp_setting)
    
transform_type = warp_properties['transform_type']
terminate_thresh = warp_properties['terminate_thresh']
grad_computation_sample_number = warp_properties['grad_computation_sample_number']
grid_search_sample_number = warp_properties['grid_search_sample_number']
std_tx_um = warp_properties['std_tx_um']
std_ty_um = warp_properties['std_ty_um']
std_tz_um = warp_properties['std_tz_um']
std_tx = std_tx_um/(XY_PIXEL_DISTANCE_LOSSLESS*32)
std_ty = std_ty_um/(XY_PIXEL_DISTANCE_LOSSLESS*32)
std_tz = std_tz_um/(XY_PIXEL_DISTANCE_LOSSLESS*32)
std_theta_xy = np.deg2rad(warp_properties['std_theta_xy_degree'])

try:
    surround_weight = float(warp_properties['surround_weight'])
    include_surround = surround_weight != 0 and not np.isnan(surround_weight)
except:
    surround_weight = str(warp_properties['surround_weight'])
    include_surround = True

reg_weight = warp_properties['regularization_weight']
if np.isnan(reg_weight):
    reg_weights = np.zeros((3,))
else:
    reg_weights = np.ones((3,))*reg_weight

MAX_ITER_NUM = 10000
HISTORY_LEN = 200
MAX_GRID_SEARCH_ITER_NUM = 30

lr1 = 10
lr2 = 0.1

########################################################

for structure in structures:

    try:

        if include_surround:
            volume_moving = DataManager.load_transformed_volume_all_known_structures(stack_m=stack_moving, 
                                         stack_f=stack_fixed, detector_id_f=detector_id, prep_id_f=2, warp_setting=upstream_warp_setting, 
                                        structures=[structure, convert_to_surround_name(structure, margin='200')])
        else:
            volume_moving = DataManager.load_transformed_volume_all_known_structures(stack_m=stack_moving, stack_f=stack_fixed,
                                                                         detector_id_f=detector_id,
                                                                                     prep_id_f=2,
                                                                         warp_setting=upstream_warp_setting, 
                                                                         structures=[structure])

        structure_to_label_moving = {s: l+1 for l, s in enumerate(sorted(volume_moving.keys()))}
        label_to_structure_moving = {l+1: s for l, s in enumerate(sorted(volume_moving.keys()))}
        volume_moving = {structure_to_label_moving[s]: v for s, v in volume_moving.items()}

        label_mapping_m2f = {label_m: structure_to_label_fixed[convert_to_original_name(name_m)] 
                             for label_m, name_m in label_to_structure_moving.iteritems()}

        if surround_weight == 'inverse':
            volume_moving_structure_sizes = {l: np.count_nonzero(vol > 0) for l, vol in volume_moving.iteritems()}
            label_weights_m = {label_m: -volume_moving_structure_sizes[structure_to_label_moving[convert_to_nonsurround_name(name_m)]]
                               /float(volume_moving_structure_sizes[label_m])
                               if is_surround_label(name_m) else 1. \
                               for label_m, name_m in label_to_structure_moving.iteritems()}
        elif isinstace(surround_weight, int) or isinstace(surround_weight, float):
            label_weights_m = {label_m: surround_weight if is_surround_label(name_m) else 1. \
                               for label_m, name_m in label_to_structure_moving.iteritems()}
        else:
            sys.stderr.write("surround_weight %s is not recognized. Using the default.\n" % surround_weight)

        aligner = Aligner4(volume_fixed, volume_moving, 
                           labelIndexMap_m2f=label_mapping_m2f)

        aligner.set_centroid(centroid_m='structure_centroid', centroid_f='centroid_m', 
                             indices_m=[structure_to_label_moving[structure]])                            

        aligner.set_regularization_weights(reg_weights)
        aligner.set_label_weights(label_weights_m)

        grid_search_T, grid_search_score = aligner.do_grid_search(grid_search_iteration_number=MAX_GRID_SEARCH_ITER_NUM, 
                       grid_search_sample_number=5,
                      std_tx=std_tx, std_ty=std_ty, std_tz=std_tz, std_theta_xy=0,
                       grid_search_eta=3., stop_radius_voxel=3)
        
        aligner.load_gradient(gradient_filepath_map_f=gradient_filepath_map_f) # 120s = 2 mins

        scores_all_trials = []
        T_all_trials = []
        traj_all_trials = []
       
        for trial_idx in range(trial_num):

            T, scores = aligner.optimize(tf_type=transform_type, 
                                         max_iter_num=MAX_ITER_NUM, 
                                         history_len=HISTORY_LEN, 
                                         terminate_thresh=1e-5,
                                         grad_computation_sample_number=grad_computation_sample_number,
                                         lr1=lr1, lr2=lr2,
                                         init_T=grid_search_T)

            T_all_trials.append(T)
            scores_all_trials.append(scores)
            traj_all_trials.append(np.array(aligner.Ts))
            
            #################################
            
            params_fp = DataManager.get_alignment_result_filepath(stack_m=stack_moving, 
                                                                  stack_f=stack_fixed,
                                                                  detector_id_f=detector_id,
                                                                  prep_id_f=2,
                                                                  structure_f=structure,
                                                                  structure_m=structure,
                                                                  warp_setting=warp_setting,
                                                                 trial_idx=trial_idx, what='parameters')
            DataManager.save_alignment_parameters(params_fp, T, 
                                                  aligner.centroid_m, aligner.centroid_f,
                                                  aligner.xdim_m, aligner.ydim_m, aligner.zdim_m, 
                                                  aligner.xdim_f, aligner.ydim_f, aligner.zdim_f)
            upload_to_s3(params_fp)

            ##################################

            history_fp = DataManager.get_alignment_result_filepath(stack_m=stack_moving, 
                                                          stack_f=stack_fixed,
                                                          detector_id_f=detector_id,
                                                          prep_id_f=2,
                                                        structure_f=structure,
                                                        structure_m=structure,        
                                                          warp_setting=warp_setting,
                                                         trial_idx=trial_idx, what='scoreHistory')
            bp.pack_ndarray_file(np.array(scores), history_fp)
            upload_to_s3(history_fp)

            ##################################

            score_plot_fp = DataManager.get_alignment_result_filepath(stack_m=stack_moving, 
                                                          stack_f=stack_fixed,
                                                          detector_id_f=detector_id,
                                                          prep_id_f=2,
                                                        structure_f=structure,
                                                        structure_m=structure,
                                                          warp_setting=warp_setting,
                                                         trial_idx=trial_idx, what='scoreEvolution')
            fig = plt.figure();
            plt.plot(scores);
            plt.savefig(score_plot_fp, bbox_inches='tight')
            plt.close(fig)

            upload_to_s3(score_plot_fp)

            ##################################

            trajectory_fp = DataManager.get_alignment_result_filepath(stack_m=stack_moving, 
                                                          stack_f=stack_fixed,
                                                          detector_id_f=detector_id,
                                                          prep_id_f=2,
                                                        structure_f=structure,
                                                          structure_m=structure,
                                                          warp_setting=warp_setting,
                                                         trial_idx=trial_idx, what='trajectory')
            bp.pack_ndarray_file(np.array(aligner.Ts), trajectory_fp)
            upload_to_s3(trajectory_fp)

        #########################
        # Save the best trial
        #########################
        
        best_trial = np.argsort([np.max(scores) for scores in scores_all_trials])[-1]

        # Save parameters
        params_fp = \
            DataManager.get_alignment_result_filepath(stack_m=stack_moving, 
                                                          stack_f=stack_fixed,
                                                          detector_id_f=detector_id,
                                                          prep_id_f=2,
                                                                  structure_f=structure,
                                                                  structure_m=structure,
                                                          warp_setting=warp_setting,
                                                         trial_idx=None, what='parameters')
        DataManager.save_alignment_parameters(params_fp, T_all_trials[best_trial], 
                                              aligner.centroid_m, aligner.centroid_f,
                                              aligner.xdim_m, aligner.ydim_m, aligner.zdim_m, 
                                              aligner.xdim_f, aligner.ydim_f, aligner.zdim_f)
        upload_to_s3(params_fp)

        # Save score history
        history_fp = DataManager.get_alignment_result_filepath(stack_m=stack_moving, 
                                                      stack_f=stack_fixed,
                                                      detector_id_f=detector_id,
                                                      prep_id_f=2,
                                                        structure_f=structure,
                                                        structure_m=structure, 
                                                      warp_setting=warp_setting,
                                                     trial_idx=None, what='scoreHistory')
        bp.pack_ndarray_file(np.array(scores_all_trials[best_trial]), history_fp)
        upload_to_s3(history_fp)

        # Save score plot
        score_plot_fp = \
        history_fp = DataManager.get_alignment_result_filepath(stack_m=stack_moving, 
                                                      stack_f=stack_fixed,
                                                      detector_id_f=detector_id,
                                                      prep_id_f=2,
                                                        structure_f=structure,
                                                        structure_m=structure,
                                                      warp_setting=warp_setting,
                                                     trial_idx=None, what='scoreEvolution')
        fig = plt.figure();
        plt.plot(scores_all_trials[best_trial]);
        plt.savefig(score_plot_fp, bbox_inches='tight')
        plt.close(fig)
        upload_to_s3(score_plot_fp)

        # Save trajectory
        trajectory_fp = DataManager.get_alignment_result_filepath(stack_m=stack_moving, 
                                                          stack_f=stack_fixed,
                                                          detector_id_f=detector_id,
                                                          prep_id_f=2,
                                                        structure_f=structure,
                                                        structure_m=structure,
                                                          warp_setting=warp_setting,
                                                         trial_idx=None, what='trajectory')
        bp.pack_ndarray_file(np.array(traj_all_trials[best_trial]), trajectory_fp)
        upload_to_s3(trajectory_fp)

    except Exception as e:
        sys.stderr.write('Error transforming volume %s: %s.\n' % (structure, e))