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
from utilities2014 import execute_command

stack = 'MD585'

bg_color = (230,232,235)

input_dir = '/home/yuncong/csd395/CSHL_data/' + stack + '_renamed'
filenames = os.listdir(input_dir)
# suffix = 'thumbnail'
suffix = 'lossless'
ext = 'tif'
all_files = dict(sorted([(int(img_fn[:-4].split('_')[1]), img_fn) 
                         for img_fn in filenames if img_fn.endswith('_' + suffix + '.' + ext)]))

# the resolution of lossless is "scale factor" times the resolution on which transform parameters are obtained
if suffix == 'lossless':
    scale_factor = 32
elif suffix == 'lossy':
    scale_factor = 16
elif suffix == 'thumbnail':
    scale_factor = 1
    
# img_shapes = {}
# for i, img_fn in all_files.iteritems():
#     img_shapes[i] = np.array(check_output("identify %s" % os.path.join(input_dir, img_fn), shell=True).split()[2].split('x'),
#                      np.int)
    
def identify_shape(img_fn):
    return map(int, check_output("identify -format %%Wx%%H %s" % os.path.join(input_dir, img_fn), 
                                 shell=True).split('x'))

img_shapes_arr = np.array(Parallel(n_jobs=16)(delayed(identify_shape)(img_fn) for img_fn in all_files.itervalues()))
max_width = img_shapes_arr[:,0].max()
max_height = img_shapes_arr[:,1].max()

margin = 0

canvas_width = max_width + 2 * margin
canvas_height = max_height + 2 * margin

padded_dir = '/home/yuncong/csd395/CSHL_data/' + stack + '_' + suffix + '_padded'
if not os.path.exists(padded_dir):
    os.makedirs(padded_dir)

warped_dir = '/home/yuncong/csd395/CSHL_data/' + stack + '_' + suffix + '_warped'
if not os.path.exists(warped_dir):
    os.makedirs(warped_dir)
    
cropped_dir = '/home/yuncong/csd395/CSHL_data/' + stack + '_' + suffix + '_cropped'
if not os.path.exists(cropped_dir):
    os.makedirs(cropped_dir)

def pad_IM(img_fn):
    
    img_fn = os.path.basename(img_fn)
        
    warped_fn = img_fn[:-4] + '_padded_warped.' + ext
    padded_fn = img_fn[:-4] + '_padded.' + ext
    
    d = {'input_fn': os.path.join(input_dir, img_fn),
        'output_fn': os.path.join(padded_dir, padded_fn),
         'bg_r': bg_color[0],
         'bg_g': bg_color[1],
         'bg_b': bg_color[2],
         'width': canvas_width,
         'height': canvas_height,
         }
        
    convert_cmd = 'convert %(input_fn)s -background "rgb(%(bg_r)d,%(bg_g)d,%(bg_b)d)" -gravity center -extent %(width)dx%(height)d -compress lzw %(output_fn)s'%d
    execute_command(convert_cmd)
    
def warp_IM(img_fn, T):
    
    warped_fn = img_fn[:-4] + '_padded_warped.' + ext
    padded_fn = img_fn[:-4] + '_padded.' + ext

    d = {'sx':T[0,0],
         'sy':T[1,1],
         'rx':T[1,0],
         'ry':T[0,1],
         'tx':T[0,2],
         'ty':T[1,2],
         'input_fn': os.path.join(padded_dir, padded_fn),
         'output_fn': os.path.join(warped_dir, warped_fn)
        }

    affine_cmd = "convert %(input_fn)s -distort AffineProjection '%(sx)f,%(rx)f,%(ry)f,%(sy)f,%(tx)f,%(ty)f' %(output_fn)s"%d
    execute_command(affine_cmd)
    
def crop_IM(img_fn, x, y, w, h):
    
    warped_fn = img_fn[:-4] + '_padded_warped.' + ext
    cropped_fn = img_fn[:-4] + '_padded_warped_cropped.' + ext

    d = {
    'x': x * scale_factor,
    'y': y * scale_factor,
    'w': w * scale_factor,
    'h': h * scale_factor,
    'input_fn': os.path.join(warped_dir, warped_fn),
    'output_fn': os.path.join(cropped_dir, cropped_fn)
    }

    crop_cmd = 'convert %(input_fn)s -crop %(w)dx%(h)d+%(x)d+%(y)d %(output_fn)s'%d
    execute_command(crop_cmd)
    
if not os.path.exists('/home/yuncong/csd395/CSHL_data/%s_thumbnail_finalTransfParams.pkl'%stack):
    execute_command('../elastix/successive_registration.py %s %d %d thumbnail' % (stack, first_sec, last_sec))
    
with open('/home/yuncong/csd395/CSHL_data/%s_thumbnail_finalTransfParams.pkl'%stack, 'r') as f:
    Ts = pickle.load(f)
    
Ts_lossless = {}
for sec, T in Ts.iteritems():
    T_lossless = T.copy()
    T_lossless[:2, 2] = T[:2, 2] * scale_factor
    Ts_lossless[sec] = T_lossless
    

t = time.time()

secind = int(sys.argv[1])

pad_IM(all_files[secind])
warp_IM(all_files[secind], np.linalg.inv(Ts_lossless[secind]))
crop_IM(all_files[secind],543,440,489,248)

print time.time() - t