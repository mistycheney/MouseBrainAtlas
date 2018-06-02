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
parser.add_argument("classifier_setting", type=int, help="classifier_setting")
parser.add_argument("--trial_idx", type=int, help="trial index", default=0)
args = parser.parse_args()

stack_fixed = args.stack_fixed
stack_moving = args.stack_moving
warp_setting = args.warp_setting
classifier_setting = args.classifier_setting
trial_idx = args.trial_idx

###################################################################################

if warp_setting == 1:
    upstream_warp_setting = None
elif warp_setting == 2:
    upstream_warp_setting = 1
elif warp_setting == 3:
    global_warp_setting = 1
    upstream_warp_setting = 2
else:
    raise Exception('Warp setting not recognized.')

if trial_idx in [0, 1]:
    upstream_trial_idx = 0

# Load transform parameters

global_params, centroid_m, centroid_f, xdim_m, ydim_m, zdim_m, xdim_f, ydim_f, zdim_f = \
DataManager.load_alignment_parameters(stack_m=stack_moving, stack_f=stack_fixed,
                                      classifier_setting_m=classifier_setting,
                                      classifier_setting_f=classifier_setting,
                                      warp_setting=global_warp_setting,
                                      trial_idx=trial_idx)

for structure in all_known_structures_sided:

    try:
        local_transformed_moving_structure_vol = \
        DataManager.load_transformed_volume(stack_m=stack_moving, stack_f=stack_fixed,
                                            classifier_setting_m=classifier_setting,
                                            classifier_setting_f=classifier_setting,
                                            warp_setting=upstream_warp_setting,
                                            trial_idx=upstream_trial_idx,
                                            structure=structure)

        local_transformed_moving_structure_global_reversed_vol = \
        transform_volume_inverse(vol=local_transformed_moving_structure_vol,
                                 global_params=global_params,
                                 centroid_m=centroid_m, centroid_f=centroid_f,
                                 xdim_m=xdim_m, ydim_m=ydim_m, zdim_m=zdim_m)

        local_transformed_moving_structure_global_reversed_fn = \
        DataManager.get_transformed_volume_filepath(stack_m=stack_moving, type_m='score',
                                                        stack_f=stack_fixed, type_f='score',
                                                        downscale=32,
                                                        classifier_setting_m=classifier_setting,
                                                        classifier_setting_f=classifier_setting,
                                                        warp_setting=warp_setting,
                                                        structure=structure)

        create_if_not_exists(os.path.dirname(local_transformed_moving_structure_global_reversed_fn))
        bp.pack_ndarray_file(local_transformed_moving_structure_global_reversed_vol, local_transformed_moving_structure_global_reversed_fn)

    except Exception as e:
        sys.stderr.write('%s\n' % e)
        sys.stderr.write('Error transforming volume %s.\n' % structure)
