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

volume_moving, structure_to_label_moving, label_to_structure_moving = \
DataManager.load_score_volume_all_known_structures(stack=stack_moving, sided=True)

volume_fixed, structure_to_label_fixed, label_to_structure_fixed = \
DataManager.load_score_volume_all_known_structures(stack=stack_fixed, classifier_setting=classifier_setting)

label_mapping_m2f = {label_m: structure_to_label_fixed[convert_to_original_name(name_m)]
                     for label_m, name_m in label_to_structure_moving.iteritems()}
                    #     if name_m in ['7N_L', '7N_R', '12N', '5N_L', '5N_R']}

aligner = Aligner4(volume_fixed, volume_moving, labelIndexMap_m2f=label_mapping_m2f)

aligner.set_centroid(centroid_m='volume_centroid', centroid_f='volume_centroid')
# aligner.set_centroid(centroid_m='structure_centroid', centroid_f='centroid_m', indices_m=[name_to_label_moving['SNR_R']])

tx_params, centroid_m, centroid_f, xdim_m, ydim_m, zdim_m, xdim_f, ydim_f, zdim_f = \
    DataManager.load_alignment_parameters(stack_f=stack_fixed, stack_m=stack_moving,
                                          classifier_setting_m=classifier_setting, classifier_setting_f=classifier_setting,
                                          warp_setting=warp_setting,
                                          downscale=32, trial_idx=trial_idx)

###############################
##### Compute Confidence ######
###############################

#     structures_for_computing_confidence = {'7N_L', '7N_R', '5N_L', '5N_R', '12N'}
#     labels_for_computing_confidence = [structure_to_label_moving[s] for s in structures_for_computing_confidence]
labels_for_computing_confidence = label_to_structure_moving.keys()

downscale = 32
xy_pixel_distance = XY_PIXEL_DISTANCE_LOSSLESS * downscale

fmax = aligner.compute_score(tx_params, indices_m=labels_for_computing_confidence)

######################
## Compute Hessians ##
######################

def perturb(tx, ty, tz):
    return aligner.compute_score(tx_params + [0,0,0,tx,0,0,0,ty,0,0,0,tz],
                             indices_m=labels_for_computing_confidence)

hessians_all_stepsizes = {}
stepsizes = np.linspace(1, 20, 5) # pixel size = 15um

for stepsize in stepsizes:
    h = nd.Hessian(lambda (tx, ty, tz): perturb(tx, ty, tz), step=(stepsize, stepsize, stepsize))
    H = h((0,0,0))
    stepsize_um = stepsize * xy_pixel_distance
    hessians_all_stepsizes[stepsize_um] = (H, fmax)

##################
## Save hessian ##
##################

fp = DataManager.get_confidence_filepath(stack_m=stack_moving, stack_f=stack_fixed,
                                   classifier_setting_m=classifier_setting, classifier_setting_f=classifier_setting,
                                   warp_setting=warp_setting, trial_idx=trial_idx,
                                   what='hessians')

create_if_not_exists(os.path.dirname(fp))
save_pickle(hessians_all_stepsizes, fp)

#####################
## Compute z-score ##
#####################

zscores = {}

# pool_radius_um_list = np.arange(25, 400, 20)
pool_radius_um_list = np.linspace(25, 400, 5)
for pool_radius_um in pool_radius_um_list:

    pool_radius_pixel = pool_radius_um / xy_pixel_distance

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


###################
## Save z-scores ##
###################

fp = DataManager.get_confidence_filepath(stack_m=stack_moving, stack_f=stack_fixed,
                                   classifier_setting_m=classifier_setting, classifier_setting_f=classifier_setting,
                                   warp_setting=warp_setting, trial_idx=trial_idx,
                                   what='zscores')

create_if_not_exists(os.path.dirname(fp))
save_pickle(zscores, fp)
