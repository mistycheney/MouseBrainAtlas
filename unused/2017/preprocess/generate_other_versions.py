#! /usr/bin/env python

import sys
import os
import numpy as np

from joblib import Parallel, delayed

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generate other versions of images, such as compressed jpeg and saturation channel as grayscale')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("first_secind", type=int, help="first section")
parser.add_argument("last_secind", type=int, help="last section")
parser.add_argument("-i", "--input_dir", type=str, help="input dir", default=None)
parser.add_argument("-compress", "--output_compressed_dir", type=str, help="output compressed dir", default=None)
# parser.add_argument("-gs", "--output_grayscale_dir", type=str, help="output grayscale dir", default=None)
parser.add_argument("-sat", "--output_saturation_dir", type=str, help="output saturation dir", default=None)
args = parser.parse_args()

DATAPROC_DIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_processed'

stack = args.stack_name

if args.input_dir is None:
	input_dir = DATAPROC_DIR + '/' + stack+'_lossless_aligned_cropped'

if args.output_compressed_dir is None:
	output_compressed_dir = DATAPROC_DIR + '/' + stack+'_lossless_aligned_cropped_downscaled'
# if args.output_grayscale_dir is None:
# 	output_grayscale_dir = DATAPROC_DIR + '/' + stack+'_lossless_aligned_cropped_grayscale'

if args.output_saturation_dir is None:
	output_saturation_dir = DATAPROC_DIR + '/' + stack+'_lossless_aligned_cropped_saturation'

first_secind = args.first_secind
last_secind = args.last_secind

if not os.path.exists(output_compressed_dir):
    try:
        os.makedirs(output_compressed_dir)
    except:
        pass

if not os.path.exists(output_saturation_dir):
    try:
        os.makedirs(output_saturation_dir)
    except:
        pass

# if not os.path.exists(output_grayscale_dir):
#     try:
#         os.makedirs(output_grayscale_dir)
#     except:
#         pass

from skimage.io import imread
from skimage.util import img_as_ubyte
import cv2

def convert_to_saturation(fn, out_fn, rescale=True):
    """
    Generate saturation channel as a grayscale image.
    """

# ImageMagick 18s
#     execute_command('convert %(fn)s -colorspace HSL -channel G %(out_fn)s' % {'fn': fn, 'out_fn': out_fn})

#     t = time.time()
    img = imread(fn)
#     sys.stderr.write('Read image: %.2f seconds\n' % (time.time() - t)) # ~4s

#     t1 = time.time()
    ma = img.max(axis=-1)
    mi = img.min(axis=-1)
#     sys.stderr.write('compute min and max color components: %.2f seconds\n' % (time.time() - t1)) # ~5s

#     t1 = time.time()
    s = np.nan_to_num(mi/ma.astype(np.float))
#     sys.stderr.write('min oiver max: %.2f seconds\n' % (time.time() - t1)) # ~2s

#     t1 = time.time()
    if rescale:
        pmax = s.max()
        pmin = s.min()
        s = (s - pmin) / (pmax - pmin)
#     sys.stderr.write('rescale: %.2f seconds\n' % (time.time() - t1)) # ~3s

#     t1 = time.time()
    cv2.imwrite(out_fn, img_as_ubyte(s))
#     imsave(out_fn, s)
#     sys.stderr.write('Compute saturation: %.2f seconds\n' % (time.time() - t1)) # skimage 6.5s; opencv 5s


def generate_versions(secind, which=['compressed', 'saturation']):
    d = {
        'input_fn': os.path.join(input_dir, stack + '_%04d'%secind + '_lossless_aligned_cropped.tif'),
        'output_compressed_fn': os.path.join(output_compressed_dir, stack + '_%04d'%secind + '_lossless_aligned_cropped_downscaled.jpg'),
        # 'output_grayscale_fn': os.path.join(output_grayscale_dir, stack + '_%04d'%secind + '_lossless_aligned_cropped_grayscale.tif'),
        'output_saturation_fn': os.path.join(output_saturation_dir, stack + '_%04d'%secind + '_lossless_aligned_cropped_saturation.jpg'),
        }

    if 'compressed' in which:
        os.system("convert %(input_fn)s -format jpg %(output_compressed_fn)s"%d)

    if 'saturation' in which:
        convert_to_saturation(d['input_fn'], d['output_saturation_fn'], rescale=True)
        # convert_to_saturation(d['output_compressed_fn'], d['output_saturation_fn'], rescale=True)
    # os.system("convert %(input_fn)s -type grayscale %(output_grayscale_fn)s"%d)

Parallel(n_jobs=4)(delayed(generate_versions)(secind, which=['saturation'])
                            for secind in range(first_secind, last_secind+1))
