#! /usr/bin/env python

import numpy as np

import sys
import os
#sys.path.append(os.environ['REPO_DIR'] + '/utilities')
sys.path.append('/shared/MouseBrainAtlas/utilities')
from utilities2015 import *
from metadata import *
from data_manager import *
from registration_utilities import *
from annotation_utilities import *

import time

stack_fixed = sys.argv[1]
train_sample_scheme = int(sys.argv[2])
global_transform_scheme = int(sys.argv[3])
trial_idx = int(sys.argv[4])
atlas_name = sys.argv[5]

# stack_moving = 'atlas_on_MD589'
stack_moving = atlas_name

############## Generate fixed-stack section images with aligned atlas overlay ################

paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', 'SNC', 'SNR', '3N', '4N',
                    'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
singular_structures = ['AP', '12N', 'RtTg', 'SC', 'IC']
structures = paired_structures + singular_structures

structures_sided = sum([[n] if n in singular_structures
                        else [convert_to_left_name(n), convert_to_right_name(n)]
                        for n in structures], [])

####################

# Read transformed volumes

vols_m_aligned_to_f = {}
for name_s in structures_sided:
    try:
        vols_m_aligned_to_f[name_s] = \
        DataManager.load_transformed_volume(stack_m=stack_moving, type_m='score',
                                                 stack_f=stack_fixed, type_f='score',
                                                 label=name_s, downscale=32,
                                                 train_sample_scheme_f=train_sample_scheme,
                                                 global_transform_scheme=global_transform_scheme)
    except:
        sys.stderr.write('No transformed volume for %s.\n' % name_s)

# vols_m_aligned_to_f = {name_s: DataManager.load_transformed_volume(stack_m=stack_moving, type_m='score',
#                                          stack_f=stack_fixed, type_f='score',
#                                          label=name_s, downscale=32,
#                                          train_sample_scheme_f=train_sample_scheme,
#                                          global_transform_scheme=global_transform_scheme)
#                         for name_s in structures_sided}

# Set colors for different contour level
levels = [0.1, 0.25, 0.5, 0.75, .99]
level_colors = {level: (int(level*255),0,0) for level in levels}

# Set output folder
viz_dir = create_if_not_exists(DataManager.get_global_alignment_viz_dir(stack_moving=stack_moving,
                                                        stack_fixed=stack_fixed,
                                                        train_sample_scheme=train_sample_scheme,
                                                        global_transform_scheme=global_transform_scheme))

# estimate mapping between z and section
downsample_factor = 32
xy_pixel_distance_downsampled = xy_pixel_distance_lossless * downsample_factor
voxel_z_size = section_thickness / xy_pixel_distance_downsampled

# For getting correct contour location
xmin_vol_f, xmax_vol_f, ymin_vol_f, ymax_vol_f, zmin_vol_f, zmax_vol_f = \
DataManager.load_volume_bbox(stack=stack_fixed, type='score', label='7N', downscale=32)
print xmin_vol_f, xmax_vol_f, ymin_vol_f, ymax_vol_f, zmin_vol_f, zmax_vol_f

# Generate atlas overlay image for every section
first_sec, last_sec = metadata_cache['section_limits'][stack_fixed]

for sec in range(first_sec, last_sec+1):

    if metadata_cache['sections_to_filenames'][stack_fixed][sec] in ['Placeholder', 'Rescan', 'Nonexisting']:
        continue

    img_fn = DataManager.get_image_filepath(stack=stack_fixed, section=sec, resol='thumbnail', version='cropped_tif')
    img = imread(img_fn)

    viz = img.copy()

    z = voxel_z_size * (sec - 1) - zmin_vol_f

    # Find moving volume annotation contours
    for name_s, vol in vols_m_aligned_to_f.iteritems():
        for level in levels:
            cnts = find_contours(vol[..., z], level=level) # rows, cols
            for cnt in cnts:
                # r,c to x,y
                cnt_on_cropped = cnt[:,::-1] + (xmin_vol_f, ymin_vol_f)
                cv2.polylines(viz, [cnt_on_cropped.astype(np.int)], True, level_colors[level], 2)


    viz_fn = os.path.join(viz_dir, '%(stack_moving)s_to_%(stack_fixed)s_%(sec)04d.jpg' % \
          {'stack_moving': stack_moving, 'stack_fixed': stack_fixed, 'sec': sec})
    imsave(viz_fn, viz)
