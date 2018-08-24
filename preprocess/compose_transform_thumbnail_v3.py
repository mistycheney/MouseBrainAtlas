#!/usr/bin/env python

import os
import numpy as np
import sys
import cPickle as pickle
import json

sys.path.append(os.environ['REPO_DIR'] + '/utilities/')
from metadata import *
from preprocess_utilities import *
from data_manager import DataManager

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generate a pkl file that stores a dict. Keys are image names and values are (3,3)-matrices.')

# parser.add_argument("stack", type=str, help="stack")
parser.add_argument("elastix_output_dir", type=str, help="")
parser.add_argument("custom_output_dir", type=str, help="")
parser.add_argument("image_name_list", type=str, help="")
parser.add_argument("anchor_img_name", type=str, help="")
parser.add_argument("toanchor_transforms_fp", type=str, help="")

args = parser.parse_args()

# stack = args.stack
elastix_output_dir = args.elastix_output_dir
custom_output_dir = args.custom_output_dir
image_name_list = load_txt(args.image_name_list)
toanchor_transforms_fp = args.toanchor_transforms_fp

#################################################

anchor_idx = image_name_list.index(args.anchor_img_name)

transformation_to_previous_sec = {}

for i in range(1, len(image_name_list)):
    
    transformation_to_previous_sec[i] = DataManager.load_consecutive_section_transform(moving_fn=image_name_list[i], fixed_fn=image_name_list[i-1], elastix_output_dir=elastix_output_dir, custom_output_dir=custom_output_dir)

transformation_to_anchor_sec = {}

for moving_idx in range(len(image_name_list)):

    if moving_idx == anchor_idx:
        # transformation_to_anchor_sec[moving_idx] = np.eye(3)
        transformation_to_anchor_sec[image_name_list[moving_idx]] = np.eye(3)

    elif moving_idx < anchor_idx:
        T_composed = np.eye(3)
        for i in range(anchor_idx, moving_idx, -1):
            T_composed = np.dot(np.linalg.inv(transformation_to_previous_sec[i]), T_composed)
        # transformation_to_anchor_sec[moving_idx] = T_composed
        transformation_to_anchor_sec[image_name_list[moving_idx]] = T_composed

    else:
        T_composed = np.eye(3)
        for i in range(anchor_idx+1, moving_idx+1):
            T_composed = np.dot(transformation_to_previous_sec[i], T_composed)
        # transformation_to_anchor_sec[moving_idx] = T_composed
        transformation_to_anchor_sec[image_name_list[moving_idx]] = T_composed
        
    print moving_idx, image_name_list[moving_idx], transformation_to_anchor_sec[image_name_list[moving_idx]]

#################################################

with open(output_fn, 'w') as f:
    pickle.dump(transformation_to_anchor_sec, f) 
    # Note that the index starts at 0, BUT the .._renamed folder index starts at 1.