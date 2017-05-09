#! /usr/bin/env python

import sys
import os

from skimage.io import imread, imsave
from skimage.exposure import rescale_intensity
import numpy as np

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from data_manager import *
from metadata import *
from multiprocess import Pool

import json
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Stretch contrast')

parser.add_argument("stack", type=str, help="stack")
parser.add_argument("filenames", type=str, help="Filenames")
#parser.add_argument("input_fp", type=str, help="input file path")
#parser.add_argument("output_fp", type=str, help="output file path")
parser.add_argument("imin", type=int, help="Minimum clip value, same for all channels")
parser.add_argument("imax", type=int, help="Maximum clip value, same for all channels")
args = parser.parse_args()

stack = args.stack
filenames = json.loads(args.filenames)

def worker(fn):
    if is_invalid(fn):
        return
    
    input_fp = DataManager.get_image_filepath(stack=stack, fn=fn, version='cropped_gray', resol='lossless')
    download_from_s3(input_fp)
    
    output_fp = DataManager.get_image_filepath(stack=stack, fn=fn, version='cropped_gray_contrast_stretched', resol='lossless')
    create_parent_dir_if_not_exists(output_fp)

    im = imread(input_fp)
    im_out = rescale_intensity(im, in_range=(args.imin, args.imax), out_range=np.uint8).astype(np.uint8)
    imsave(output_fp, im_out)
    upload_to_s3(output_fp)    
    
pool = Pool(NUM_CORES/2)
pool.map(worker, filenames)
pool.close()
pool.join()

# for fn in filenames:
    
#     if is_invalid(fn):
#         continue
    
#     input_fp = DataManager.get_image_filepath(stack=stack, fn=fn, version='cropped_gray', resol='lossless')
#     download_from_s3(input_fp)
    
#     output_fp = DataManager.get_image_filepath(stack=stack, fn=fn, version='cropped_gray_contrast_stretched', resol='lossless')
#     create_parent_dir_if_not_exists(output_fp)

#     im = imread(input_fp)
#     im_out = rescale_intensity(im, in_range=(args.imin, args.imax), out_range=np.uint8).astype(np.uint8)
#     imsave(output_fp, im_out)
#     upload_to_s3(output_fp)