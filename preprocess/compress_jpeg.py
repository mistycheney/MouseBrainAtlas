#! /usr/bin/env python

import sys
import os
import json

from multiprocess import Pool

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from data_manager import *
from metadata import *
from distributed_utilities import *

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Compress image as JPEG. The output version is the input version with "Jpeg" appended.')

parser.add_argument("input_spec", type=str, help="Input specifier")
parser.add_argument("--depth", type=int, help="Image depth", default=8) # imagemagick cannot generate 16-bit JPEG (?)
parser.add_argument("--quality", type=int, help="JPEG quality", default=80)
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

depth = args.depth
quality = args.quality

for img_name in image_name_list:

    in_fp = DataManager.get_image_filepath_v2(stack=stack, prep_id=prep_id, resol=resol, version=version, fn=img_name)
    out_fp = DataManager.get_image_filepath_v2(stack=stack, prep_id=prep_id, resol=resol, version=version+'Jpeg', fn=img_name)

    create_parent_dir_if_not_exists(out_fp)
    #download_from_s3(input_fp)
    execute_command("convert \"%(input_fp)s\" -depth %(depth)d -format jpg -quality %(quality)d \"%(output_fp)s\"" % dict(input_fp=in_fp, output_fp=out_fp, depth=depth, quality=quality))
    #upload_to_s3(output_fp)

