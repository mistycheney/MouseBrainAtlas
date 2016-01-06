#! /usr/bin/env python

import sys
import os
import numpy as np

from joblib import Parallel, delayed

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generate downscaled and grayscale versions of images')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("first_secind", type=int, help="first section")
parser.add_argument("last_secind", type=int, help="last section")
parser.add_argument("-i", "--input_dir", type=str, help="input dir", default=None)
parser.add_argument("-ds", "--output_downscaled_dir", type=str, help="output downscaled dir", default=None)
parser.add_argument("-gs", "--output_grayscale_dir", type=str, help="output grayscale dir", default=None)
args = parser.parse_args()

DATAPROC_DIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_processed'

stack = args.stack_name
if args.input_dir is None:
	input_dir = DATAPROC_DIR + '/' + stack+'_lossless_aligned_cropped'
if args.output_downscaled_dir is None:
	output_downscaled_dir = DATAPROC_DIR + '/' + stack+'_lossless_aligned_cropped_downscaled'
if args.output_grayscale_dir is None:
	output_grayscale_dir = DATAPROC_DIR + '/' + stack+'_lossless_aligned_cropped_grayscale'
first_secind = args.first_secind
last_secind = args.last_secind

if not os.path.exists(output_downscaled_dir):
    os.makedirs(output_downscaled_dir)

if not os.path.exists(output_grayscale_dir):
    os.makedirs(output_grayscale_dir)

def f(secind):
    d = {
        'input_fn': os.path.join(input_dir, stack + '_%04d'%secind + '_lossless_aligned_cropped.tif'),
        'output_downscaled_fn': os.path.join(output_downscaled_dir, stack + '_%04d'%secind + '_lossless_aligned_cropped_downscaled.jpg'),
        'output_grayscale_fn': os.path.join(output_grayscale_dir, stack + '_%04d'%secind + '_lossless_aligned_cropped_grayscale.tif')
        }

    # os.system("convert %(input_fn)s -format jpg %(output_downscaled_fn)s"%d)
    os.system("convert %(input_fn)s -type grayscale %(output_grayscale_fn)s"%d)

Parallel(n_jobs=8)(delayed(f)(secind) for secind in range(first_secind, last_secind+1))
