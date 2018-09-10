#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Align consecutive images. Possible bad alignment pairs are written into a separate file.')

parser.add_argument("input_spec", type=str, help="txt. Each row is prev_img_name, prev_fp, curr_img_name, curr_fp")
parser.add_argument("elastix_output_dir", type=str, help="output dir. Files for each pairwise transform are stored in sub-folder <currImageName>_to_<prevImageName>.")
parser.add_argument("param_fp", type=str, help="elastix parameter file path")
#parser.add_argument("-r", help="re-generate even if already exists", action='store_true')

args = parser.parse_args()

import os
import sys
import json
import re

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from data_manager import *
from metadata import *
from distributed_utilities import *

input_spec = load_ini(args.input_spec)
image_name_list = input_spec['image_name_list']
stack = input_spec['stack']
prep_id = input_spec['prep_id']
if prep_id == 'None':
    prep_id = None
resol = input_spec['resol']
version = input_spec['version']
if version == 'None':
    version = None

run_distributed("%(script)s \"%(output_dir)s\" \'%%(kwargs_str)s\' -p %(param_fp)s -r" % \
                {'script': os.path.join(REPO_DIR, 'preprocess', 'align_sequential.py'),
                'output_dir': args.elastix_output_dir,
                 'param_fp': args.param_fp
                },
                kwargs_list=[{'prev_img_name': image_name_list[i-1],
                              'curr_img_name': image_name_list[i],
                              'prev_fp': DataManager.get_image_filepath_v2(stack=stack, fn=image_name_list[i-1], prep_id=prep_id, resol=resol, version=version),
                              'curr_fp': DataManager.get_image_filepath_v2(stack=stack, fn=image_name_list[i], prep_id=prep_id, resol=resol, version=version)
                             }
                            for i in range(1, len(image_name_list))],
                argument_type='list',
                jobs_per_node=8,
               local_only=True)
