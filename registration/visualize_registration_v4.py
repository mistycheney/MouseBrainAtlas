#! /usr/bin/env python

import sys
import os
import time

import numpy as np

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from registration_utilities import *
from annotation_utilities import *
from metadata import *
from data_manager import *

##################################################################

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='')
parser.add_argument("stack_fixed", type=str, help="Fixed stack name")
parser.add_argument("stack_moving", type=str, help="Moving stack name")
parser.add_argument("warp_setting", type=int, help="Warp setting")
parser.add_argument("classifier_setting", type=int, help="classifier_setting")
parser.add_argument("-o", "--out_downsample", type=int, help="downsample of output visualization images", default=32)

args = parser.parse_args()

stack_fixed = args.stack_fixed
stack_moving = args.stack_moving
warp_setting = args.warp_setting
classifier_setting = args.classifier_setting
downsample_factor = args.out_downsample
    
##################################################################

# Read transformed volumes

volume_downsample = 32
warped_volumes = DataManager.load_transformed_volume_all_known_structures(stack_m=stack_moving, stack_f=stack_fixed,
                                    classifier_setting_m=classifier_setting,
                                    classifier_setting_f=classifier_setting,
                                    warp_setting=warp_setting,
                                                                          downscale=volume_downsample,
                                                                          sided=True)

if len(warped_volumes) == 0:
    raise Exception('No volumes are loaded.')

# Set colors for different contour level
levels = [0.1, 0.25, 0.5, 0.75, .99]
level_colors = {0.1: (0,255,255), 
                0.25: (0,255,0), 
                0.5: (255,0,0), 
                0.75: (255,255,0), 
                0.99: (255,0,255)}

# For getting correct contour location
bbox_down32 = \
DataManager.load_original_volume_bbox(stack=stack_fixed, vol_type='score', structure='7N', downscale=32, classifier_setting=classifier_setting)
xmin_vol_f, xmax_vol_f, ymin_vol_f, ymax_vol_f, zmin_vol_f, zmax_vol_f = bbox_down32 * 32 / volume_downsample
# print xmin_vol_f, xmax_vol_f, ymin_vol_f, ymax_vol_f, zmin_vol_f, zmax_vol_f

show_text = True
contour_width = 1

def visualize_registration_one_section(sec):
    
    t = time.time()
    
    if downsample_factor == 32:
        img = DataManager.load_image(stack=stack_fixed, section=sec, resol='thumbnail', version='cropped_tif')
    else:
        img = DataManager.load_image(stack=stack_fixed, section=sec, resol='lossless', version='cropped_gray_jpeg')
        img = img[::downsample_factor, ::downsample_factor]
    
    if img.ndim == 2:
        img = gray2rgb(img)
        
    viz = img.copy()
    
    z = int(np.mean(DataManager.convert_section_to_z(stack=stack_fixed, sec=sec, downsample=volume_downsample)))
    
    # Find moving volume annotation contours.
    for name_s, vol in warped_volumes.iteritems():
        
        label_pos = None
        
        for level in levels:
            cnts = find_contours(vol[..., z], level=level) # rows, cols
            for cnt in cnts:
                # r,c to x,y
                cnt_on_cropped_volRes = cnt[:,::-1] + (xmin_vol_f, ymin_vol_f)
                cnt_on_cropped_imgRes = cnt_on_cropped_volRes * volume_downsample / downsample_factor
                cv2.polylines(viz, [cnt_on_cropped_imgRes.astype(np.int)], 
                              True, level_colors[level], contour_width)
                
                if show_text:
                    if label_pos is None:
                        label_pos = np.mean(cnt_on_cropped_imgRes, axis=0)
    
        # Show text 
        if label_pos is not None:
            cv2.putText(viz, name_s, tuple(label_pos.astype(np.int)), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, ((0,0,0)), 3)
    
    viz_fp = DataManager.get_alignment_viz_filepath(stack_m=stack_moving,
                                                    stack_f=stack_fixed,
                                                    classifier_setting_m=classifier_setting,
                                                    classifier_setting_f=classifier_setting,
                                                    warp_setting=warp_setting,
                                                downscale=volume_downsample,
                                                    out_downscale=downsample_factor,
                                          section=sec)
    try:
        create_parent_dir_if_not_exists(viz_fp)
    except:
        pass
    imsave(viz_fp, viz)
    upload_to_s3(viz_fp)
    
#     sys.stderr.write('Time: %.2f seconds\n' % (time.time() - t)) # 5s
    
t = time.time()

pool = Pool(NUM_CORES)
pool.map(visualize_registration_one_section, metadata_cache['valid_sections'][stack_fixed])
pool.terminate()
pool.join()

sys.stderr.write('Visualize registration: %.2f seconds\n' % (time.time() - t)) # 110s

# for sec in range(first_sec, last_sec+1):
# # for sec in [200]:

#     if is_invalid(metadata_cache['sections_to_filenames'][stack_fixed][sec]):
#         continue

#     img_fn = DataManager.get_image_filepath(stack=stack_fixed, section=sec, resol='thumbnail', version='cropped_tif')
#     img = imread(img_fn)

#     viz = img.copy()

#     z = voxel_z_size * (sec - 1) - zmin_vol_f

#     # Find moving volume annotation contours
#     for name_s, vol in warped_volumes.iteritems():
#         for level in levels:
#             cnts = find_contours(vol[..., z], level=level) # rows, cols
#             for cnt in cnts:
#                 # r,c to x,y
#                 cnt_on_cropped = cnt[:,::-1] + (xmin_vol_f, ymin_vol_f)
#                 cv2.polylines(viz, [cnt_on_cropped.astype(np.int)], True, level_colors[level], 2)


#     viz_fp = DataManager.get_alignment_viz_filepath(stack_m=stack_moving,
#                                             stack_f=stack_fixed,
#                                             classifier_setting_m=classifier_setting,
#                                             classifier_setting_f=classifier_setting,
#                                             warp_setting=warp_setting,
#                                           section=sec)

#     create_if_not_exists(os.path.dirname(viz_fp))
#     imsave(viz_fp, viz)
