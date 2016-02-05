#! /usr/bin/env python

import sys
import os
import numpy as np
import cPickle as pickle

from joblib import Parallel, delayed


import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Process after having bounding box')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("input_dir", type=str, help="input dir")
parser.add_argument("output_dir", type=str, help="output dir")
parser.add_argument("first_sec", type=int, help="first section")
parser.add_argument("last_sec", type=int, help="last section")
parser.add_argument("suffix", type=str, help="suffix")
parser.add_argument("x", type=int, help="x on thumbnail", default=0)
parser.add_argument("y", type=int, help="y on thumbnail", default=0)
parser.add_argument("w", type=int, help="w on thumbnail", default=2000)
parser.add_argument("h", type=int, help="h on thumbnail", default=1500)
args = parser.parse_args()


# stack = sys.argv[1]
# input_dir = sys.argv[2]
# output_dir = sys.argv[3]
# first_secind = int(sys.argv[4])
# last_secind = int(sys.argv[5])
# suffix = sys.argv[6]
# crop = bool(int(sys.argv[7]))

stack = args.stack_name
input_dir = args.input_dir
output_dir = args.output_dir
first_sec = args.first_sec
last_sec = args.last_sec
suffix = args.suffix

# if args.x is not None:
x = args.x
y = args.y
w = args.w
h = args.h
crop = not (x == 0 and y == 0 and w == 2000 and h == 1500)
# else:
#     x = 0
#     y = 0
#     w = 2000
#     h = 1500    

# if crop:
#     x,y,w,h = map(int, sys.argv[8:12])
# else:
#     x = 0
#     y = 0
#     w = 2000
#     h = 1500

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# DATAPROC_DIR = '/home/yuncong/csd395/CSHL_data_processed'

all_files = dict(sorted([(int(img_fn[:-4].split('_')[1]), img_fn) for img_fn in os.listdir(input_dir) if suffix in img_fn]))

if suffix == 'thumbnail':
    scale_factor = 1
elif suffix == 'lossy':
    scale_factor = 16
elif suffix == 'lossless':
    scale_factor = 32

# with open(os.path.join(DATAPROC_DIR, stack + '_finalTransfParams.pkl'), 'r') as f:
with open(os.path.join(os.environ['DATA_DIR'], stack + '_finalTransfParams.pkl'), 'r') as f:
    Ts = pickle.load(f)
    
for secind in range(first_sec, last_sec+1):

    T2 = Ts[secind].copy()
    T2[:2,2] = Ts[secind][:2, 2] * scale_factor
    T = np.linalg.inv(T2)

    d = {'sx':T[0,0],
         'sy':T[1,1],
         'rx':T[1,0],
         'ry':T[0,1],
         'tx':T[0,2],
         'ty':T[1,2],
         'input_fn': os.path.join(input_dir, all_files[secind]),
         'output_fn': os.path.join(output_dir, all_files[secind][:-4] + '_aligned'+ ('_cropped.tif' if crop else '.tif')),
         'x': '+' + str(x * scale_factor) if int(x)>=0 else str(x * scale_factor),
         'y': '+' + str(y * scale_factor) if int(y)>=0 else str(y * scale_factor),
         'w': str(w * scale_factor),
         'h': str(h * scale_factor),
        }

    os.system("convert %(input_fn)s -virtual-pixel background +distort AffineProjection '%(sx)f,%(rx)f,%(ry)f,%(sy)f,%(tx)f,%(ty)f' -crop %(w)sx%(h)s%(x)s%(y)s\! -flatten -compress lzw %(output_fn)s"%d)