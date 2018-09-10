#! /usr/bin/env python

import os
import sys
import time
import argparse

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *
from data_manager import *

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Rename thumbnail images according to our naming format with consecutive section numbers')

parser.add_argument("input_spec", type=str, help="Path to input files specification as ini file")
parser.add_argument("channel", type=int, help="Channel index")
parser.add_argument("out_version", type=str, help="Output image version")
args = parser.parse_args()

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

create_if_not_exists(DataManager.get_image_dir_v2(stack=stack, prep_id=prep_id, resol=resol, version=args.out_version))

run_distributed('convert \"%%(in_fp)s\" -channel %(channel)s -separate \"%%(out_fp)s\"' % {'channel': 'RGB'[args.channel]},
                kwargs_list=[{'in_fp': DataManager.get_image_filepath_v2(stack=stack, prep_id=prep_id, 
                                        resol=resol, version=version, fn=img_name),
                                       'out_fp': DataManager.get_image_filepath_v2(stack=stack, prep_id=prep_id, 
                                        resol=resol, version=args.out_version, fn=img_name)}
                                       for img_name in image_name_list],
                argument_type='single',
                jobs_per_node=1,
                local_only=True)
