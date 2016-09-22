#! /usr/bin/env python

import sys
import os
import numpy as np
import cPickle as pickle
import json

from joblib import Parallel, delayed

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Warp and crop images.')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("input_dir", type=str, help="input dir")
parser.add_argument("output_dir", type=str, help="output dir")
parser.add_argument("transform_fp", type=str, help="transform file path")
parser.add_argument("filenames", type=str, help="filenames")
parser.add_argument("suffix", type=str, help="suffix")
parser.add_argument("x", type=int, help="x on thumbnail", default=0)
parser.add_argument("y", type=int, help="y on thumbnail", default=0)
parser.add_argument("w", type=int, help="w on thumbnail", default=2000)
parser.add_argument("h", type=int, help="h on thumbnail", default=1500)
args = parser.parse_args()

stack = args.stack_name
input_dir = args.input_dir
output_dir = args.output_dir
transform_fp = args.transform_fp
filenames = json.loads(args.filenames)[0]['filenames']

print filenames

suffix = args.suffix

x = args.x
y = args.y
w = args.w
h = args.h
crop = not (x == 0 and y == 0 and w == 2000 and h == 1500)


if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
    except:
        pass

if suffix == 'thumbnail':
    scale_factor = 1
elif suffix == 'lossy':
    scale_factor = 16
elif suffix == 'lossless':
    scale_factor = 32

with open(transform_fp, 'r') as f:
    Ts = pickle.load(f)

for secind in range(len(filenames)):

    if secind not in Ts:
        sys.stderr.write('No transform is computed for section %d.\n' % secind)
        continue

    T2 = Ts[secind].copy()
    T2[:2,2] = Ts[secind][:2, 2] * scale_factor
    T = np.linalg.inv(T2)

    d = {'sx':T[0,0],
         'sy':T[1,1],
         'rx':T[1,0],
         'ry':T[0,1],
         'tx':T[0,2],
         'ty':T[1,2],
         'input_fn': os.path.join(input_dir, filenames[secind] + '.tif'),
         'output_fn': os.path.join(output_dir, filenames[secind][:-4] + '_aligned'+ ('_cropped.tif' if crop else '.tif')),
         'x': '+' + str(x * scale_factor) if int(x)>=0 else str(x * scale_factor),
         'y': '+' + str(y * scale_factor) if int(y)>=0 else str(y * scale_factor),
         'w': str(w * scale_factor),
         'h': str(h * scale_factor),
        }

    os.system("convert %(input_fn)s -virtual-pixel background +distort AffineProjection '%(sx)f,%(rx)f,%(ry)f,%(sy)f,%(tx)f,%(ty)f' -crop %(w)sx%(h)s%(x)s%(y)s\! -flatten -compress lzw %(output_fn)s"%d)
