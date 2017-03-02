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

# Load transform parameters
global_params, centroid_m, centroid_f, xdim_m, ydim_m, zdim_m, xdim_f, ydim_f, zdim_f = \
DataManager.load_alignment_parameters(stack_m=stack_moving, stack_f=stack_fixed,
                                      classifier_setting_m=classifier_setting,
                                      classifier_setting_f=classifier_setting,
                                      warp_setting=warp_setting,
                                      trial_idx=trial_idx)

for structure in all_known_structures_sided_with_surround:
# for structure in all_known_structures_sided:

    try:
        vol_m = DataManager.load_score_volume(stack=stack_moving, structure=structure, downscale=32)

        volume_m_alignedTo_f = \
        transform_volume(vol=vol_m, global_params=global_params, centroid_m=centroid_m, centroid_f=centroid_f,
                          xdim_f=xdim_f, ydim_f=ydim_f, zdim_f=zdim_f)

        volume_m_alignedTo_f_fn = \
        DataManager.get_transformed_volume_filepath(stack_m=stack_moving, type_m='score',
                                                    stack_f=stack_fixed, type_f='score',
                                                    downscale=32,
                                                    classifier_setting_m=classifier_setting,
                                                    classifier_setting_f=classifier_setting,
                                                    warp_setting=warp_setting,
                                                    structure=structure)

        create_if_not_exists(os.path.dirname(volume_m_alignedTo_f_fn))
        bp.pack_ndarray_file(volume_m_alignedTo_f, volume_m_alignedTo_f_fn)

    except Exception as e:
        sys.stderr.write('%s\n' % e)
        sys.stderr.write('Error transforming volume %s.\n' % structure)
