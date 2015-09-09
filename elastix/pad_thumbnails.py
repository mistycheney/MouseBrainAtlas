#! /usr/bin/env python

import sys
import os
from subprocess import check_output
from skimage.util import pad
from skimage.io import imread, imsave
import numpy as np
import cPickle as pickle
import cv2
import time

from joblib import Parallel, delayed

sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], 'notebooks'))
from utilities2014 import execute_command

stack = sys.argv[1]
first_sec = int(sys.argv[2])
last_sec = int(sys.argv[3])

bg_color = (230,232,235)
ext = 'tif'

def pad_IM(stack, secind, suffix, ext, input_dir, output_dir):
    
    img_fn = '%s_%04d_%s.%s'%(stack, secind, suffix, ext)
            
    warped_fn = img_fn[:-4] + '_padded_warped.' + ext
    padded_fn = img_fn[:-4] + '_padded.' + ext
    
    d = {'input_fn': os.path.join(input_dir, img_fn),
        'output_fn': os.path.join(output_dir, padded_fn),
         'bg_r': bg_color[0],
         'bg_g': bg_color[1],
         'bg_b': bg_color[2],
         'width': canvas_width,
         'height': canvas_height,
         }
        
    convert_cmd = 'convert %(input_fn)s -background "rgb(%(bg_r)d,%(bg_g)d,%(bg_b)d)" -gravity center -extent %(width)dx%(height)d -compress lzw %(output_fn)s'%d
    execute_command(convert_cmd)
    
input_dir = '/home/yuncong/csd395/CSHL_data/' + stack + '_renamed'
filenames = os.listdir(input_dir)
all_files = dict(sorted([(int(img_fn[:-4].split('_')[1]), img_fn) 
                         for img_fn in filenames if img_fn.endswith('_thumbnail.tif')]))
    
def identify_shape(img_fn):
    return map(int, check_output("identify -format %%Wx%%H %s" % os.path.join(input_dir, img_fn), 
                                 shell=True).split('x'))

img_shapes_arr = np.array(Parallel(n_jobs=16)(delayed(identify_shape)(img_fn) for img_fn in all_files.itervalues()))
max_width = img_shapes_arr[:,0].max()
max_height = img_shapes_arr[:,1].max()

margin = 0

canvas_width = max_width + 2 * margin
canvas_height = max_height + 2 * margin

padded_dir = '/home/yuncong/csd395/CSHL_data/' + stack + '_thumbnail_padded'
if not os.path.exists(padded_dir):
    os.makedirs(padded_dir)

Parallel(n_jobs=16)(delayed(pad_IM)(stack, secind, suffix='thumbnail', ext='tif', input_dir=input_dir, output_dir=padded_dir) for secind in range(first_sec, last_sec+1))
