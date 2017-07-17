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
    description='Stretch contrast. Keep only two channels: blue (neurotrace) and label.')

parser.add_argument("stack", type=str, help="stack")
parser.add_argument("filenames", type=str, help="Filenames")
parser.add_argument("-c", "--label_channel", type=int, help="label channel")
parser.add_argument("-max", "--label_clipmax", type=int, help="label clip max value")
args = parser.parse_args()

stack = args.stack
filenames = json.loads(args.filenames)
label_channel = args.label_channel
label_clipmax = args.label_clipmax

def worker(fn):
    if is_invalid(fn):
        return
    
    img = DataManager.load_image_v2(stack=stack, prep_id=2, resol='lossless', fn=fn)
    img_label_channel = img[:, :, label_channel]
    img_label_channel_contrast_stretched = img_as_ubyte(rescale_intensity(img_label_channel, in_range=(0, label_clipmax)))

    img_blue_normalized = DataManager.load_image_v2(stack=stack, prep_id=2, resol='lossless', fn=fn, version='gray')
    
    contrast_stretched_img = np.zeros(img.shape, np.uint8)
    contrast_stretched_img[..., label_channel] = img_label_channel_contrast_stretched
    contrast_stretched_img[..., 2] = rescale_intensity(-img_blue_normalized)
    
    output_fp = DataManager.get_image_filepath_v2(stack=stack, prep_id=2, resol='lossless', fn=fn, version='contrastStretched', ext='jpg')
    create_parent_dir_if_not_exists(output_fp)
    imsave(output_fp, contrast_stretched_img)
    upload_to_s3(output_fp, local_root=DATA_ROOTDIR)    
    
# pool = Pool(2)
# pool.map(worker, filenames)
# pool.close()
# pool.join()

for fn in filenames:
    worker(fn)