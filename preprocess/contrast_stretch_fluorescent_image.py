#! /usr/bin/env python

import sys
import os
import json
import argparse

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from data_manager import *
from metadata import *
from multiprocess import Pool

from skimage.exposure import rescale_intensity
import numpy as np
import cv2

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Stretch contrast. Keep only two channels: blue (neurotrace) and label.')

parser.add_argument("stack", type=str, help="stack")
parser.add_argument("filenames", type=str, help="Filenames")
parser.add_argument("-c", "--label_channel", type=int, help="label channel")
parser.add_argument("-max", "--label_clipmax", type=int, help="label clip max value")
parser.add_argument("-p", "--prep_id", type=int, help="prep id", default=2)
args = parser.parse_args()

stack = args.stack
filenames = json.loads(args.filenames)
label_channel = args.label_channel
label_clipmax = args.label_clipmax
prep_id = args.prep_id

def worker(fn):
    if is_invalid(fn):
        return
    
    img = DataManager.load_image_v2(stack=stack, prep_id=prep_id, resol='lossless', fn=fn)
    img_label_channel = img[:, :, label_channel]
    img_label_channel_contrast_stretched = img_as_ubyte(rescale_intensity(img_label_channel, in_range=(0, label_clipmax)))
    del img, img_label_channel

    img_blue_normalized = DataManager.load_image_v2(stack=stack, prep_id=prep_id, resol='lossless', fn=fn, version='gray')
    
    contrast_stretched_img = np.zeros(img_label_channel_contrast_stretched.shape + (3,), np.uint8)
    contrast_stretched_img[..., label_channel] = img_label_channel_contrast_stretched
    contrast_stretched_img[..., 2] = rescale_intensity(-img_blue_normalized)
    del img_blue_normalized
    
    output_fp = DataManager.get_image_filepath_v2(stack=stack, prep_id=prep_id, resol='lossless', fn=fn, version='contrastStretched', ext='jpg')
    create_parent_dir_if_not_exists(output_fp)
#     imsave(output_fp, contrast_stretched_img) #do not work for images as large as 20000x40000 (entire brain crop)
    cv2.imwrite(output_fp, contrast_stretched_img[..., ::-1])
    upload_to_s3(output_fp, local_root=DATA_ROOTDIR)    
    
# pool = Pool(2)
# pool.map(worker, filenames)
# pool.close()
# pool.join()

for fn in filenames:
    worker(fn)