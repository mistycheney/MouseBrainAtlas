#! /usr/bin/env python

import sys
import os
import json

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from data_manager import *
from metadata import *
from distributed_utilities import *

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Compress image as JPEG and rescale')

parser.add_argument("input_fp", type=str, help="Input image name")
parser.add_argument("output_fp", type=str, help="Output image name")
#parser.add_argument("-f", "--rescale_factor", type=float, help="Rescale factor")
parser.add_argument("-W", "--width", type=int, help="Width")
parser.add_argument("-H", "--height", type=int, help="Height")
args = parser.parse_args()

input_fp = args.input_fp
output_fp = args.output_fp
# if hasattr(args, "rescale_factor"):
#     rescale_factor = args.rescale_factor
# else:
#     w = args.width
#     h = args.height
w = args.width
h = args.height

create_parent_dir_if_not_exists(output_fp)
download_from_s3(input_fp, local_root=DATA_ROOTDIR)
execute_command("convert \"%(input_fp)s\" -resize %(w)dx%(h)d -format jpg \"%(output_fp)s\"" % dict(input_fp=input_fp, output_fp=output_fp, w=w, h=h))
upload_to_s3(output_fp, local_root=DATA_ROOTDIR)