#! /usr/bin/env python

import sys
import os
sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from skimage.io import imread, imsave
from skimage.exposure import rescale_intensity
import argparse
import numpy as np

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Stretch contrast')

parser.add_argument("input_fp", type=str, help="input file path")
parser.add_argument("output_fp", type=str, help="output file path")
parser.add_argument("imin", type=int, help="Minimum clip value, same for all channels")
parser.add_argument("imax", type=int, help="Maximum clip value, same for all channels")
args = parser.parse_args()

im = imread(args.input_fp)[..., :3]
im_out = rescale_intensity(im, in_range=(args.imin, args.imax), out_range=np.uint8).astype(np.uint8)
imsave(args.output_fp, im_out)