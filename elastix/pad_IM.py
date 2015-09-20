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

stack = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]
first_sec = int(sys.argv[4])
last_sec = int(sys.argv[5])
suffix = sys.argv[6]

DATAPROC_DIR = '/home/yuncong/csd395/CSHL_data_processed'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

all_files = dict(sorted([(int(img_fn[:-4].split('_')[1]), img_fn) for img_fn in os.listdir(input_dir) if suffix in img_fn]))
all_files = dict([(i, all_files[i]) for i in range(first_sec, last_sec+1)])
print all_files
    
def identify_shape(img_fn):
    return map(int, check_output("identify -format %%Wx%%H %s" % os.path.join(input_dir, img_fn), shell=True).split('x'))

img_shapes_map = dict(zip(all_files.keys(), Parallel(n_jobs=16)(delayed(identify_shape)(img_fn) for img_fn in all_files.values())))

img_shapes_arr = np.array(img_shapes_map.values())
max_width = img_shapes_arr[:,0].max()
max_height = img_shapes_arr[:,1].max()

largest_sec = img_shapes_map.keys()[np.argmax(img_shapes_arr[:,0] * img_shapes_arr[:,1])]

print 'Largest section is', largest_sec
with open(os.path.join(DATAPROC_DIR, 'tmp', stack+'_largest_sec'), 'w') as f:
    f.write(str(largest_sec))

margin = 0

canvas_width = max_width + 2 * margin
canvas_height = max_height + 2 * margin

with open(os.path.join(DATAPROC_DIR, 'tmp', stack + '_bgColor.pkl'), 'r') as f:
    bg_map = pickle.load(f)

def pad_IM(input_fn, bg_color):
    
    d = {'input_fn': os.path.join(input_dir, input_fn),
        'output_fn': os.path.join(output_dir, input_fn[:-4]+'_padded.tif'),
         'bg_r': bg_color[0],
         'bg_g': bg_color[1],
         'bg_b': bg_color[2],
         'width': canvas_width,
         'height': canvas_height,
         }
         
    os.system('convert %(input_fn)s -background "rgb(%(bg_r)d,%(bg_g)d,%(bg_b)d)" -gravity center -extent %(width)dx%(height)d -compress lzw %(output_fn)s'%d)
    

Parallel(n_jobs=16)(delayed(pad_IM)(all_files[secind], bg_map[secind]) for secind in range(first_sec, last_sec+1))