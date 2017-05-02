#! /usr/bin/env python

import sys
import os
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from registration_utilities import *
from metadata import *
from data_manager import *

###############################################################

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='')
parser.add_argument("stack_fixed", type=str, help="Fixed stack name")
parser.add_argument("stack_moving", type=str, help="Moving stack name")
parser.add_argument("warp_setting", type=int, help="Warp setting")
parser.add_argument("classifier_setting", type=int, help="classifier_setting")
parser.add_argument("--trial_idx", type=int, help="which trial of warpping to use", default=0)
args = parser.parse_args()

stack_fixed = args.stack_fixed
stack_moving = args.stack_moving
warp_setting = args.warp_setting
classifier_setting = args.classifier_setting
trial_idx = args.trial_idx

#############################################################################

warp_properties = registration_settings.loc[warp_setting]

upstream_warp_setting = warp_properties['upstream_warp_id']
if upstream_warp_setting == 'None':
    upstream_warp_setting = None
else:
    upstream_warp_setting = int(upstream_warp_setting)
    upstream_trial_idx = 1 # Need to modify this for every warp
    
###################################################################################

if upstream_warp_setting is None:
    
    # Load transform parameters
    global_params, centroid_m, centroid_f, xdim_m, ydim_m, zdim_m, xdim_f, ydim_f, zdim_f = \
    DataManager.load_alignment_parameters(stack_m=stack_moving, stack_f=stack_fixed,
                                          classifier_setting_m=classifier_setting,
                                          classifier_setting_f=classifier_setting,
                                          warp_setting=warp_setting,
                                          trial_idx=trial_idx)
    
    def transform_volume_one_structure(structure):
        try:
            t = time.time()

            vol_m = DataManager.load_original_volume(stack=stack_moving, structure=structure, downscale=32, volume_type='score')

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
                                                        structure=structure,
                                                       trial_idx=trial_idx)

            create_parent_dir_if_not_exists(volume_m_alignedTo_f_fn)
            bp.pack_ndarray_file(volume_m_alignedTo_f, volume_m_alignedTo_f_fn)

            upload_from_ec2_to_s3(volume_m_alignedTo_f_fn)

            sys.stderr.write('Transform: %.2f seconds.\n' % (time.time() - t)) # 3s

        except Exception as e:
            sys.stderr.write('%s\n' % e)
            sys.stderr.write('Error transforming volume %s.\n' % structure)
            
    t = time.time()
    pool = Pool(NUM_CORES)
    pool.map(transform_volume_one_structure, all_known_structures_sided_with_surround)
    pool.close()
    pool.join()
    sys.stderr.write('Transform all structures: %.2f seconds.\n' % (time.time() - t))
            
else:
    
    def transform_volume_one_structure(structure):
        # Load local transform parameters
        try:

            t = time.time()
        
            local_params, centroid_m, centroid_f, xdim_m, ydim_m, zdim_m, xdim_f, ydim_f, zdim_f = \
            DataManager.load_alignment_parameters(stack_m=stack_moving, stack_f=stack_fixed,
                                                  classifier_setting_m=classifier_setting,
                                                  classifier_setting_f=classifier_setting,
                                                  warp_setting=warp_setting,
                                                  param_suffix=structure,
                                                  trial_idx=trial_idx)

            # Read global tx
            global_transformed_moving_structure_vol = \
            DataManager.load_transformed_volume(stack_m=stack_moving, stack_f=stack_fixed,
                                                classifier_setting_m=classifier_setting,
                                                classifier_setting_f=classifier_setting,
                                                warp_setting=upstream_warp_setting, 
                                                trial_idx=upstream_trial_idx,
                                                structure=structure)

            # Transform
            local_transformed_moving_structure_vol = transform_volume(vol=global_transformed_moving_structure_vol, 
                                                     global_params=local_params, 
                                                     centroid_m=centroid_m, centroid_f=centroid_f,
                                                     xdim_f=xdim_f, ydim_f=ydim_f, zdim_f=zdim_f)

            # Save
            local_transformed_moving_structure_fn = \
            DataManager.get_transformed_volume_filepath(stack_m=stack_moving, stack_f=stack_fixed,
                                                        classifier_setting_m=classifier_setting,
                                                        classifier_setting_f=classifier_setting,
                                                        warp_setting=warp_setting,
                                                        trial_idx=trial_idx,
                                                        structure=structure)

            create_parent_dir_if_not_exists(local_transformed_moving_structure_fn)
            bp.pack_ndarray_file(local_transformed_moving_structure_vol, local_transformed_moving_structure_fn)

            upload_from_ec2_to_s3(local_transformed_moving_structure_fn)

            sys.stderr.write('Transform: %.2f seconds.\n' % (time.time() - t))

        except Exception as e:
            sys.stderr.write('%s\n' % e)
            sys.stderr.write('Error transforming volume %s.\n' % structure)
    
    
    t = time.time()
    pool = Pool(NUM_CORES)
    pool.map(transform_volume_one_structure, all_known_structures_sided_with_surround)
    pool.close()
    pool.join()
    sys.stderr.write('Transform all structures: %.2f seconds.\n' % (time.time() - t))        