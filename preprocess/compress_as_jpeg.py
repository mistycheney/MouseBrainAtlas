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
    description='Compress image as JPEG')

parser.add_argument("input_fp", type=str, help="Input image name")
parser.add_argument("output_fp", type=str, help="Output image name")
parser.add_argument("--depth", type=int, help="Image depth", default=8) # imagemagick cannot generate 16-bit JPEG (?)
parser.add_argument("--quality", type=int, help="JPEG quality", default=80)
args = parser.parse_args()

input_fp = args.input_fp
output_fp = args.output_fp
depth = args.depth
quality = args.quality

create_parent_dir_if_not_exists(output_fp)
download_from_s3(input_fp)
execute_command("convert \"%(input_fp)s\" -depth %(depth)d -format jpg -quality %(quality)d \"%(output_fp)s\"" % dict(input_fp=input_fp, output_fp=output_fp, depth=depth, quality=quality))
upload_to_s3(output_fp)