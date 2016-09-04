#!/usr/bin/env python

import time
import sys
import os
# import numpy as np

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from preprocess_utility import *
from metadata import *

from joblib import Parallel, delayed

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generate compressed JPEG and grayscale versions of images')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("-first_sec", type=int, help="first section")
parser.add_argument("-last_sec", type=int, help="last section")
# parser.add_argument("-i", "--input_dir", type=str, help="input dir", default=None)
# parser.add_argument("-ds", "--output_downscaled_dir", type=str, help="output downscaled dir", default=None)
# parser.add_argument("-gs", "--output_grayscale_dir", type=str, help="output grayscale dir", default=None)
args = parser.parse_args()

# hostids = detect_responsive_nodes(exclude_nodes=[33,42])
hostids = detect_responsive_nodes()

print hostids

n_hosts = len(hostids)

DATAPROC_DIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_processed'

stack = args.stack_name
# if args.input_dir is None:
# 	input_dir = DATAPROC_DIR + '/' + stack+'_lossless_aligned_cropped'
# if args.output_downscaled_dir is None:
# 	output_downscaled_dir = DATAPROC_DIR + '/' + stack+'_lossless_aligned_cropped_downscaled'
# if args.output_grayscale_dir is None:
# 	output_grayscale_dir = DATAPROC_DIR + '/' + stack+'_lossless_aligned_cropped_grayscale'
# first_sec = args.first_sec
# last_sec = args.last_sec

f0, l0 = section_range_lookup[stack]
first_sec = args.last_sec if args.last_sec is not None else l0
last_sec = args.first_sec if args.first_sec is not None else f0

# if not os.path.exists(output_downscaled_dir):
#     os.makedirs(output_downscaled_dir)

# if not os.path.exists(output_grayscale_dir):
#     os.makedirs(output_grayscale_dir)

script_dir = os.path.join(os.environ['REPO_DIR'], 'preprocess')

t = time.time()
sys.stderr.write('generating compressed JPEG and saturation as grayscale versions ...')
run_distributed3(command=script_dir + '/generate_other_versions.py',
				[(stack, f, l) for f, l in first_last_tuples_distribute_over(first_sec, last_sec, n_hosts)],
				stdout=open('/tmp/log', 'ab+'))
sys.stderr.write('done in %f seconds\n' % (time.time() - t))
