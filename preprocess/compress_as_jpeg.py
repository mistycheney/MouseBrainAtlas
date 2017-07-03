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
args = parser.parse_args()

input_fp = args.input_fp
output_fp = args.output_fp

create_parent_dir_if_not_exists(output_fp)
download_from_s3(input_fp)
execute_command("convert \"%(input_fp)s\" -format jpg \"%(output_fp)s\"" % dict(input_fp=input_fp, output_fp=output_fp))
upload_to_s3(output_fp)