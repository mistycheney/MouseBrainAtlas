#! /usr/bin/env python

import numpy as np
import sys
import os
import time

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from registration_utilities import *
from annotation_utilities import *
from metadata import *
from data_manager import *

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

# Read transformed volumes

warped_volumes = DataManager.load_transformed_volume_all_known_structures(stack_m=stack_moving,
                                                                                  type_m='score',
                                                                                  stack_f=stack_fixed,
                                                                                  type_f='score',
                                                                                  downscale=32,
                                    classifier_setting_m=classifier_setting,
                                    classifier_setting_f=classifier_setting,
                                    warp_setting=warp_setting,
                                    trial_idx=trial_idx, sided=True)

# Set colors for different contour level
levels = [0.1, 0.25, 0.5, 0.75, .99]
level_colors = {level: (int(level*255),0,0) for level in levels}

# Generate overlay visualization

# estimate mapping between z and section
downsample_factor = 32
xy_pixel_distance_downsampled = XY_PIXEL_DISTANCE_LOSSLESS * downsample_factor
voxel_z_size = SECTION_THICKNESS / xy_pixel_distance_downsampled

# For getting correct contour location
xmin_vol_f, xmax_vol_f, ymin_vol_f, ymax_vol_f, zmin_vol_f, zmax_vol_f = \
DataManager.load_volume_bbox(stack=stack_fixed, type='score', structure='7N',
                            downscale=32, classifier_setting=classifier_setting)
print xmin_vol_f, xmax_vol_f, ymin_vol_f, ymax_vol_f, zmin_vol_f, zmax_vol_f

# Generate atlas overlay image for every section
first_sec, last_sec = metadata_cache['section_limits'][stack_fixed]

for sec in range(first_sec, last_sec+1):
# for sec in [200]:

    if is_invalid(metadata_cache['sections_to_filenames'][stack_fixed][sec]):
        continue

    img_fn = DataManager.get_image_filepath(stack=stack_fixed, section=sec, resol='thumbnail', version='cropped_tif')
    img = imread(img_fn)

    viz = img.copy()

    z = voxel_z_size * (sec - 1) - zmin_vol_f

    # Find moving volume annotation contours
    for name_s, vol in warped_volumes.iteritems():
        for level in levels:
            cnts = find_contours(vol[..., z], level=level) # rows, cols
            for cnt in cnts:
                # r,c to x,y
                cnt_on_cropped = cnt[:,::-1] + (xmin_vol_f, ymin_vol_f)
                cv2.polylines(viz, [cnt_on_cropped.astype(np.int)], True, level_colors[level], 2)


    viz_fp = DataManager.get_alignment_viz_filepath(stack_m=stack_moving,
                                            stack_f=stack_fixed,
                                            classifier_setting_m=classifier_setting,
                                            classifier_setting_f=classifier_setting,
                                            warp_setting=warp_setting,
                                          section=sec)

    create_if_not_exists(os.path.dirname(viz_fp))
    imsave(viz_fp, viz)
