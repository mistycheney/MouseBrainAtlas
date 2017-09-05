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
parser.add_argument("-d", "--detector_id", type=int, help="Detector id", default=None)
parser.add_argument("-n", "--trial_num", type=int, help="number of trials", default=1)
parser.add_argument("-s", "--structures", type=str, help="Json encoding of structure name list")
parser.add_argument("--stack_fixed_type", type=str, help="Fixed stack type", default='score')
parser.add_argument("--stack_moving_type", type=str, help="Moving stack type", default='score')
args = parser.parse_args()

stack_fixed = args.stack_fixed
stack_moving = args.stack_moving
warp_setting = args.warp_setting
detector_id = args.detector_id
trial_num = args.trial_num
if hasattr(args, "structures"):
    structure_subset = json.loads(args.structures)
else:
    structure_subset = all_known_structures_sided
stack_fixed_type = args.stack_fixed_type
stack_moving_type = args.stack_moving_type

if stack_fixed_type == 'score':
    prep_id = 2
    negate_surround = True
    f_is_sided = False
else:
    prep_id = None
    negate_surround = False
    f_is_sided = True
    
###################################################################

warp_properties = registration_settings.loc[warp_setting]
print warp_properties

upstream_warp_setting = warp_properties['upstream_warp_id']
if upstream_warp_setting == 'None':
    upstream_warp_setting = None
transform_type = warp_properties['transform_type']
terminate_thresh = warp_properties['terminate_thresh']
grad_computation_sample_number = int(warp_properties['grad_computation_sample_number'])
if not np.isnan(warp_properties['grid_search_sample_number']):
    grid_search_sample_number = int(warp_properties['grid_search_sample_number'])
std_tx_um = warp_properties['std_tx_um']
std_ty_um = warp_properties['std_ty_um']
std_tz_um = warp_properties['std_tz_um']
std_tx = std_tx_um/(XY_PIXEL_DISTANCE_LOSSLESS*32)
std_ty = std_ty_um/(XY_PIXEL_DISTANCE_LOSSLESS*32)
std_tz = std_tz_um/(XY_PIXEL_DISTANCE_LOSSLESS*32)
std_theta_xy = np.deg2rad(warp_properties['std_theta_xy_degree'])
if not np.isnan(warp_properties['max_iter_num']):
    max_iter_num = int(warp_properties['max_iter_num'])

positive_weight = 'size'
    
surround_weight = warp_properties['surround_weight']
if isinstance(surround_weight, float) or isinstance(surround_weight, int):
    surround_weight = float(surround_weight)
    include_surround = surround_weight != 0 and not np.isnan(surround_weight)
elif isinstance(surround_weight, str):
    surround_weight = str(surround_weight)
    # Setting surround_weight as inverse is very important. Using -1 often gives false peaks.
    include_surround = True
    
HISTORY_LEN = 20
# MAX_GRID_SEARCH_ITER_NUM = 30

lr1 = 10
lr2 = 0.1

#####################################################################

volume_moving, structure_to_label_moving, label_to_structure_moving = \
DataManager.load_original_volume_all_known_structures(stack=stack_moving, sided=True, volume_type=stack_moving_type, 
                                                      include_surround=include_surround)

volume_fixed, structure_to_label_fixed, label_to_structure_fixed = \
DataManager.load_original_volume_all_known_structures(stack=stack_fixed, detector_id=detector_id, prep_id=prep_id,
                                                      sided=f_is_sided, volume_type=stack_fixed_type)

if include_surround:
    structure_subset = structure_subset + [convert_to_surround_name(s, margin=200) for s in structure_subset]
    
label_mapping_m2f = {}    
for label_m, name_m in label_to_structure_moving.iteritems():
    if name_m in structure_subset:
        if f_is_sided:
            name_f = name_m
        else:
            name_f = convert_to_original_name(name_m)
        if name_f in structure_to_label_fixed:
            label_mapping_m2f[label_m] = structure_to_label_fixed[name_f]
    
# label_mapping_m2f = {label_m: structure_to_label_fixed[name_m if f_is_sided else convert_to_original_name(name_m)] 
#                      for label_m, name_m in label_to_structure_moving.iteritems()
#                      if name_m in structure_subset}

cutoff = .5 # Structure size is defined as the number of voxels whose value is above this cutoff probability.
pool = Pool(NUM_CORES)
volume_moving_structure_sizes = dict(zip(volume_moving.keys(), 
                                         pool.map(lambda l: np.count_nonzero(volume_moving[l] > cutoff), 
                                                  volume_moving.keys())))
pool.close()
pool.join()

label_weights_m = {}

for label_m in label_mapping_m2f.iterkeys():
    name_m = label_to_structure_moving[label_m]
    if not is_surround_label(name_m):
        if positive_weight == 'size':
            label_weights_m[label_m] = 1.
        elif positive_weight == 'inverse':
            p = np.percentile(volume_moving_structure_sizes.values(), 50)
            label_weights_m[label_m] =  np.minimum(p / volume_moving_structure_sizes[label_m], 1.)
        else:
            sys.stderr.write("positive_weight %s is not recognized. Using the default.\n" % positive_weight)
            
for label_m in label_mapping_m2f.iterkeys():
    name_m = label_to_structure_moving[label_m]
    if is_surround_label(name_m):
        label_ns = structure_to_label_moving[convert_to_nonsurround_name(name_m)]
        if surround_weight == 'inverse':
            if negate_surround:
                label_weights_m[label_m] = - label_weights_m[label_ns] * volume_moving_structure_sizes[label_ns]/float(volume_moving_structure_sizes[label_m])
            else:
                label_weights_m[label_m] = label_weights_m[label_ns] * volume_moving_structure_sizes[label_ns]/float(volume_moving_structure_sizes[label_m])
            
        elif isinstance(surround_weight, int) or isinstance(surround_weight, float):
            label_weights_m[label_m] = surround_weight
        else:
            sys.stderr.write("surround_weight %s is not recognized. Using the default.\n" % surround_weight)
            
################## Optimization ######################

aligner = Aligner4(volume_fixed, volume_moving, labelIndexMap_m2f=label_mapping_m2f)
# aligner.set_centroid(centroid_m='volume_centroid', centroid_f='centroid_m')
aligner.set_centroid(centroid_m='volume_centroid', centroid_f='volume_centroid')
# aligner.set_centroid(centroid_m='structure_centroid', centroid_f='centroid_m', indices_m=[name_to_label_moving['SNR_R']])

aligner.set_label_weights(label_weights=label_weights_m)

# grid_search_T, grid_search_score = aligner.do_grid_search(grid_search_iteration_number=MAX_GRID_SEARCH_ITER_NUM, 
#                        grid_search_sample_number=5,
#                       std_tx=std_tx, std_ty=std_ty, std_tz=std_tz, std_theta_xy=0,
#                        grid_search_eta=3.)

gradient_filepath_map_f = \
{ind_f: DataManager.get_volume_gradient_filepath_template(\
                                                          stack=stack_fixed, 
                                                          structure=label_to_structure_fixed[ind_f],
                                                          detector_id=detector_id, 
                                                         prep_id=prep_id,
                                                         volume_type=stack_fixed_type)
 for ind_m, ind_f in label_mapping_m2f.iteritems()}

gradients = {ind_f: np.zeros((3,)+volume_fixed.values()[0].shape, dtype=np.float16) 
             for ind_f in set(label_mapping_m2f.values())}

for ind_f in set(label_mapping_m2f.values()):

    t = time.time()

    download_from_s3(gradient_filepath_map_f[ind_f] % {'suffix': 'gx'}, is_dir=False)
    download_from_s3(gradient_filepath_map_f[ind_f] % {'suffix': 'gy'}, is_dir=False)
    download_from_s3(gradient_filepath_map_f[ind_f] % {'suffix': 'gz'}, is_dir=False)

    gradients[ind_f][0] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gx'})
    gradients[ind_f][1] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gy'})
    gradients[ind_f][2] = bp.unpack_ndarray_file(gradient_filepath_map_f[ind_f] % {'suffix': 'gz'})

    sys.stderr.write('load gradient %s: %f seconds\n' % (ind_f, time.time() - t)) # ~6s


# aligner.load_gradient(gradient_filepath_map_f=gradient_filepath_map_f) # 120s = 2 mins
aligner.load_gradient(gradients=gradients)

parameters_all_trials = []
scores_all_trials = []
traj_all_trials = []

for trial_idx in range(trial_num):

    while True:
        try:
            T, scores = aligner.optimize(tf_type=transform_type, 
                                         max_iter_num=max_iter_num, 
                                         history_len=HISTORY_LEN, 
                                     terminate_thresh_rot=.002,
                                     terminate_thresh_trans=.2,
                                         grad_computation_sample_number=grad_computation_sample_number,
                                         lr1=lr1, lr2=lr2,
                                         #init_T=grid_search_T
                                         # affine_scaling_limits=(.5, 1.2)
                                        )
            T = aligner.Ts[-1]
            break

        except Exception as e:
            sys.stderr.write(e.message + '\n')

    # Save parameters
    params_fp = \
    DataManager.get_alignment_result_filepath(stack_m=stack_moving, 
                                                  stack_f=stack_fixed,
                                                  detector_id_f=detector_id,
                                                  prep_id_f=prep_id,
                                                  warp_setting=warp_setting,
                                              vol_type_m=stack_moving_type,
                                              vol_type_f=stack_fixed_type,
                                                 trial_idx=trial_idx, what='parameters')
    DataManager.save_alignment_parameters(params_fp, T, 
                                          aligner.centroid_m, aligner.centroid_f,
                                          aligner.xdim_m, aligner.ydim_m, aligner.zdim_m, 
                                          aligner.xdim_f, aligner.ydim_f, aligner.zdim_f)
    upload_to_s3(params_fp)
    
    # Save score history
    history_fp = DataManager.get_alignment_result_filepath(stack_m=stack_moving, 
                                                  stack_f=stack_fixed,
                                                  detector_id_f=detector_id,
                                                  prep_id_f=prep_id,
                                                  warp_setting=warp_setting,
                                                           vol_type_m=stack_moving_type,
                                              vol_type_f=stack_fixed_type,
                                                 trial_idx=trial_idx, what='scoreHistory')
    bp.pack_ndarray_file(np.array(scores), history_fp)
    upload_to_s3(history_fp)

    # Save score plot
    score_plot_fp = DataManager.get_alignment_result_filepath(stack_m=stack_moving, 
                                                  stack_f=stack_fixed,
                                                  detector_id_f=detector_id,
                                                  prep_id_f=prep_id,
                                                  warp_setting=warp_setting,
                                                              vol_type_m=stack_moving_type,
                                              vol_type_f=stack_fixed_type,
                                                 trial_idx=trial_idx, what='scoreEvolution')
    fig = plt.figure();
    plt.plot(scores);
    plt.savefig(score_plot_fp, bbox_inches='tight')
    plt.close(fig)
    upload_to_s3(score_plot_fp)
    
    # Save trajectory
    trajectory_fp = DataManager.get_alignment_result_filepath(stack_m=stack_moving, 
                                                  stack_f=stack_fixed,
                                                  detector_id_f=detector_id,
                                                  prep_id_f=prep_id,
                                                  warp_setting=warp_setting,
                                                              vol_type_m=stack_moving_type,
                                              vol_type_f=stack_fixed_type,
                                                 trial_idx=trial_idx, what='trajectory')
    bp.pack_ndarray_file(np.array(aligner.Ts), trajectory_fp)
    upload_to_s3(trajectory_fp)
    
    ########
    
    parameters_all_trials.append(T)
    scores_all_trials.append(scores)
    traj_all_trials.append(aligner.Ts)

best_trial = np.argsort([np.max(scores) for scores in scores_all_trials])[-1]

# Save parameters
params_fp = \
    DataManager.get_alignment_result_filepath(stack_m=stack_moving, 
                                                  stack_f=stack_fixed,
                                                  detector_id_f=detector_id,
                                                  prep_id_f=prep_id,
                                                  warp_setting=warp_setting,
                                              vol_type_m=stack_moving_type,
                                              vol_type_f=stack_fixed_type,
                                                 trial_idx=None, what='parameters')
DataManager.save_alignment_parameters(params_fp, parameters_all_trials[best_trial], 
                                      aligner.centroid_m, aligner.centroid_f,
                                      aligner.xdim_m, aligner.ydim_m, aligner.zdim_m, 
                                      aligner.xdim_f, aligner.ydim_f, aligner.zdim_f)
upload_to_s3(params_fp)

# Save score history
history_fp = DataManager.get_alignment_result_filepath(stack_m=stack_moving, 
                                              stack_f=stack_fixed,
                                              detector_id_f=detector_id,
                                              prep_id_f=prep_id,
                                              warp_setting=warp_setting,
                                                       vol_type_m=stack_moving_type,
                                              vol_type_f=stack_fixed_type,
                                             trial_idx=None, what='scoreHistory')
bp.pack_ndarray_file(np.array(scores_all_trials[best_trial]), history_fp)
upload_to_s3(history_fp)

# Save score plot
score_plot_fp = \
history_fp = DataManager.get_alignment_result_filepath(stack_m=stack_moving, 
                                              stack_f=stack_fixed,
                                              detector_id_f=detector_id,
                                              prep_id_f=prep_id,
                                              warp_setting=warp_setting,
                                                       vol_type_m=stack_moving_type,
                                              vol_type_f=stack_fixed_type,
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
                                                  prep_id_f=prep_id,
                                                  warp_setting=warp_setting,
                                                          vol_type_m=stack_moving_type,
                                              vol_type_f=stack_fixed_type,
                                                 trial_idx=None, what='trajectory')
bp.pack_ndarray_file(np.array(traj_all_trials[best_trial]), trajectory_fp)
upload_to_s3(trajectory_fp)