#! /usr/bin/env python

import sys
import os
import numpy as np
import cPickle as pickle

from joblib import Parallel, delayed

stack = sys.argv[1]
input_dir = sys.argv[2]
output_downscaled_dir = sys.argv[3]
output_grayscale_dir = sys.argv[4]
first_secind = int(sys.argv[5])
last_secind = int(sys.argv[6])

if not os.path.exists(output_downscaled_dir):
    os.makedirs(output_downscaled_dir)

if not os.path.exists(output_grayscale_dir):
    os.makedirs(output_grayscale_dir)

# all_files = dict(sorted([(int(img_fn[:-4].split('_')[1]), img_fn) for img_fn in os.listdir(input_dir) if suffix in img_fn]))

# for secind in range(first_secind, last_secind+1):

#     d = {
#         'input_fn': os.path.join(input_dir, stack + '_%04d'%secind + '_lossless_aligned_cropped.tif'),
#         'output_downscaled_fn': os.path.join(output_downscaled_dir, stack + '_%04d'%secind + '_lossless_aligned_cropped_downscaled.jpg'),
#         'output_grayscale_fn': os.path.join(output_grayscale_dir, stack + '_%04d'%secind + '_lossless_aligned_cropped_grayscale.tif')
#         }

#     os.system("convert %(input_fn)s -format jpg %(output_downscaled_fn)s"%d)
#     os.system("convert %(input_fn)s -type grayscale %(output_grayscale_fn)s"%d)

def f(secind):
    d = {
        'input_fn': os.path.join(input_dir, stack + '_%04d'%secind + '_lossless_aligned_cropped.tif'),
        'output_downscaled_fn': os.path.join(output_downscaled_dir, stack + '_%04d'%secind + '_lossless_aligned_cropped_downscaled.jpg'),
        'output_grayscale_fn': os.path.join(output_grayscale_dir, stack + '_%04d'%secind + '_lossless_aligned_cropped_grayscale.tif')
        }

    os.system("convert %(input_fn)s -format jpg %(output_downscaled_fn)s"%d)
    os.system("convert %(input_fn)s -type grayscale %(output_grayscale_fn)s"%d)

Parallel(n_jobs=16)(delayed(f)(secind) for secind in range(first_secind, last_secind+1))
