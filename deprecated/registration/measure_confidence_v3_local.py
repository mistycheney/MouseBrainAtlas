#! /usr/bin/env python

# This must be before any other matplotlib imports
import matplotlib
matplotlib.use('Agg')

import numpy as np
import sys
import os
import time

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from registration_utilities import *
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
parser.add_argument("detector_id", type=int, help="detector_id")
#parser.add_argument("--trial_idx", type=int, help="trial index", default=0)
args = parser.parse_args()

stack_fixed = args.stack_fixed
stack_moving = args.stack_moving
warp_setting = args.warp_setting
detector_id = args.detector_id
#trial_idx = args.trial_idx

###################################################################################

import numdifftools as nd

warp_properties = registration_settings.loc[warp_setting]
print warp_properties

upstream_warp_setting = warp_properties['upstream_warp_id']
if upstream_warp_setting == 'None':
    upstream_warp_setting = None
else:
    upstream_warp_setting = int(upstream_warp_setting)
    
transform_type = warp_properties['transform_type']
terminate_thresh = warp_properties['terminate_thresh']
grad_computation_sample_number = int(warp_properties['grad_computation_sample_number'])
if not np.isnan(warp_properties['grid_search_sample_number']):
    grid_search_sample_number = int(warp_properties['grid_search_sample_number'])
if not np.isnan(warp_properties['std_tx_um']):
    std_tx_um = warp_properties['std_tx_um']
    std_tx = std_tx_um/(XY_PIXEL_DISTANCE_LOSSLESS*32)
if not np.isnan(warp_properties['std_ty_um']):
    std_ty_um = warp_properties['std_ty_um']
    std_ty = std_ty_um/(XY_PIXEL_DISTANCE_LOSSLESS*32)
if not np.isnan(warp_properties['std_tz_um']):
    std_tz_um = warp_properties['std_tz_um']
    std_tz = std_tz_um/(XY_PIXEL_DISTANCE_LOSSLESS*32)
if not np.isnan(warp_properties['std_theta_xy_degree']):
    std_theta_xy = np.deg2rad(warp_properties['std_theta_xy_degree'])
if not np.isnan(warp_properties['max_iter_num']):
    max_iter_num = int(warp_properties['max_iter_num'])
    
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

print
print 'surround', surround_weight
print 'regularization', reg_weights

positive_weight = 'size'

pool_radius_um_list = np.array([25, 50, 100, 150, 200, 300, 400])
stepsize_um_list = np.array([25, 50, 100, 150, 200, 300, 400])

#################################################################

volume_fixed, structure_to_label_fixed, label_to_structure_fixed = \
DataManager.load_original_volume_all_known_structures(stack=stack_fixed, 
                                                      prep_id=2,
                                                      detector_id=detector_id,
                                                     sided=False, volume_type='score')

for structure in all_known_structures_sided:
# for structure in ['7N_L']:

    try:

        ##############################
        # Initialize aligner object. #
        ##############################

        if include_surround:
            volume_moving = DataManager.load_transformed_volume_all_known_structures(stack_m=stack_moving, 
                                         stack_f=stack_fixed, prep_id_f=2, detector_id_f=detector_id, warp_setting=upstream_warp_setting, 
                                        structures=[structure, convert_to_surround_name(structure, margin='200')])
        else:
            volume_moving = DataManager.load_transformed_volume_all_known_structures(stack_m=stack_moving, 
                                                                                     stack_f=stack_fixed,
                                                                                     prep_id_f=2, 
                                                                         detector_id_f=detector_id,
                                                                         warp_setting=upstream_warp_setting, 
                                                                         structures=[structure])

        structure_to_label_moving = {s: l+1 for l, s in enumerate(sorted(volume_moving.keys()))}
        label_to_structure_moving = {l+1: s for l, s in enumerate(sorted(volume_moving.keys()))}
        volume_moving = {structure_to_label_moving[s]: v for s, v in volume_moving.items()}

        label_mapping_m2f = {label_m: structure_to_label_fixed[convert_to_original_name(name_m)] 
                             for label_m, name_m in label_to_structure_moving.iteritems()}

        ######################

        cutoff = .5 # Structure size is defined as the number of voxels whose value is above this cutoff probability.
        # volume_moving_structure_sizes = {m_ind: np.count_nonzero(volume_moving[m_ind] > cutoff) 
        #                                  for m_ind in label_mapping_m2f.iterkeys()}
        pool = Pool(NUM_CORES)
        volume_moving_structure_sizes = dict(zip(volume_moving.keys(), 
                                                 pool.map(lambda l: np.count_nonzero(volume_moving[l] > cutoff), 
                                                          label_mapping_m2f.iterkeys())))
        pool.close()
        pool.join()

        ########################

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
                    label_weights_m[label_m] = - label_weights_m[label_ns] * volume_moving_structure_sizes[label_ns]/float(volume_moving_structure_sizes[label_m])
                elif isinstance(surround_weight, int) or isinstance(surround_weight, float):
                    label_weights_m[label_m] = surround_weight
                else:
                    sys.stderr.write("surround_weight %s is not recognized. Using the default.\n" % surround_weight)


        aligner = Aligner4(volume_fixed, volume_moving, 
                           labelIndexMap_m2f=label_mapping_m2f)

        aligner.set_centroid(centroid_m='structure_centroid', centroid_f='centroid_m', 
                             indices_m=[structure_to_label_moving[structure]])

        aligner.set_regularization_weights(reg_weights)
        aligner.set_label_weights(label_weights_m)

        ########################################
        # Read previous computed best estimate #
        ########################################

        tx_params, centroid_m, centroid_f, xdim_m, ydim_m, zdim_m, xdim_f, ydim_f, zdim_f = \
        DataManager.load_alignment_parameters(stack_m=stack_moving, stack_f=stack_fixed,
                                              detector_id_f=detector_id,
                                                 prep_id_f=2,
                                              warp_setting=warp_setting,
                                              structure_f=structure,
                                             structure_m=structure)

#         structures_for_computing_confidence = {'7N_L', '7N_R', '5N_L', '5N_R', '12N'}
#         labels_for_computing_confidence = [structure_to_label_moving[s] for s in structures_for_computing_confidence]
#         labels_for_computing_confidence = label_to_structure_moving.keys()

        labels_for_computing_confidence = [structure_to_label_moving[structure]]

        fmax = aligner.compute_score(tx_params, indices_m=labels_for_computing_confidence)

        ###################
        # Compute z-score #
        ###################

        zscores = {}

        for pool_radius_um in pool_radius_um_list:

            pool_radius_pixel = pool_radius_um / XY_PIXEL_DISTANCE_TB

            t = time.time()

        #     grid_size = 3
        #     dxs = np.arange(-pooling_radius, pooling_radius, grid_size)
        #     dys = np.arange(-pooling_radius, pooling_radius, grid_size)
        #     dzs = np.arange(-pooling_radius, pooling_radius, grid_size)
        #     neighbor_scores = aligner.compute_scores_neighborhood_grid(tx_params, dxs=dxs, dys=dys, dzs=dzs,
        #                                                                indices_m=labels_for_computing_confidence)

            neighbor_scores = aligner.compute_scores_neighborhood_random(tx_params, n=3000, 
                            stds=np.array([0,0,0,pool_radius_pixel,0,0,0,pool_radius_pixel,0,0,0,pool_radius_pixel]), 
                            indices_m=labels_for_computing_confidence)

            sys.stderr.write('Compute scores: %.2f seconds.\n' % (time.time() - t))

            mean = np.mean(neighbor_scores)
            std = np.std(neighbor_scores)
            z = (fmax - mean) / std

            zscores[pool_radius_um] = (z, fmax, mean, std)

        #################
        # Save z-scores #
        #################

        fp = DataManager.get_confidence_filepath(stack_m=stack_moving, stack_f=stack_fixed, 
                                                 detector_id_f=detector_id,
                                                 prep_id_f=2,
                                                 warp_setting=warp_setting,
                                                 structure_f=structure,
                                                 structure_m=structure,
                                                 what='zscores')
        create_parent_dir_if_not_exists(fp)
        save_pickle(zscores, fp)
        upload_to_s3(fp)

        ####################
        # Compute Hessians #
        ####################

        def perturb(tx, ty, tz):
            return aligner.compute_score(tx_params + [0,0,0,tx,0,0,0,ty,0,0,0,tz],
                                         indices_m=labels_for_computing_confidence)

        hessians_all_stepsizes = {}
        for stepsize_um in stepsize_um_list:
            stepsize = stepsize_um / XY_PIXEL_DISTANCE_TB
            h = nd.Hessian(lambda (tx, ty, tz): perturb(tx, ty, tz), step=(stepsize, stepsize, stepsize))
            H = h((0,0,0))
            hessians_all_stepsizes[stepsize_um] = (H, fmax)
#                 U, S, UT = np.linalg.svd(H)
#                 steepest_dir = U[:,0]
#                 flattest_dir = U[:,-1]
#                 stepsize_um = stepsize * xy_pixel_distance
#                 hessians_all_stepsizes[stepsize_um] = (H, fmax, steepest_dir, flattest_dir)

        #################
        # Save hessians #
        #################

        fp = DataManager.get_confidence_filepath(stack_m=stack_moving, stack_f=stack_fixed,
                                               detector_id_f=detector_id,
                                                 prep_id_f=2,
                                               warp_setting=warp_setting,
                                                 structure_f=structure,
                                                 structure_m=structure,
                                                 what='hessians')

        create_parent_dir_if_not_exists(fp)
        save_pickle(hessians_all_stepsizes, fp)
        upload_to_s3(fp)           

    except Exception as e:
        sys.stderr.write('Error transforming volume %s: %s\n' % (structure, e))