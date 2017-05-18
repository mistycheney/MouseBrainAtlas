#! /usr/bin/env python

import sys
import os
import numpy as np
import json

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import execute_command

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Warp and crop images.')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("input_dir", type=str, help="input dir")
parser.add_argument("output_dir", type=str, help="output dir")
parser.add_argument("transform", type=str, help="thumbnail transform")
parser.add_argument("filename", type=str, help="filename")
parser.add_argument("output_fn", type=str, help="output filename")
parser.add_argument("suffix", type=str, help="suffix")
parser.add_argument("x", type=int, help="x on thumbnail", default=0)
parser.add_argument("y", type=int, help="y on thumbnail", default=0)
parser.add_argument("w", type=int, help="w on thumbnail", default=2000)
parser.add_argument("h", type=int, help="h on thumbnail", default=1500)
parser.add_argument("background_color", type=str, help="background color (black or white)", default='white')
args = parser.parse_args()

stack = args.stack_name
input_dir = args.input_dir
output_dir = args.output_dir
output_fn = args.output_fn
transform = np.reshape(map(np.float, args.transform.split(',')), (3,3))
filename = args.filename
background_color = args.background_color

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

#if os.path.exists(os.path.join(output_dir, output_fn)):
#    sys.stderr.write('Output image already exists: %s\n' % output_fn)
#    sys.exit(0)

sys.stderr.write(output_fn + '\n')

T2 = transform.copy()
T2[:2,2] = transform[:2, 2] * scale_factor
T = np.linalg.inv(T2)

d = {'sx':T[0,0],
     'sy':T[1,1],
     'rx':T[1,0],
     'ry':T[0,1],
     'tx':T[0,2],
     'ty':T[1,2],
     'input_fn': os.path.join(input_dir, filename),
     'output_fn': os.path.join(output_dir, output_fn),
     'x': '+' + str(x * scale_factor) if int(x)>=0 else str(x * scale_factor),
     'y': '+' + str(y * scale_factor) if int(y)>=0 else str(y * scale_factor),
     'w': str(w * scale_factor),
     'h': str(h * scale_factor),
    }

sys.stderr.write("Background color: %s\n" % background_color)

if background_color == 'black':
    execute_command("convert %(input_fn)s -virtual-pixel background -background black +distort AffineProjection '%(sx)f,%(rx)f,%(ry)f,%(sy)f,%(tx)f,%(ty)f' -crop %(w)sx%(h)s%(x)s%(y)s\! -flatten -compress lzw %(output_fn)s"%d)
elif background_color == 'white':
    execute_command("convert %(input_fn)s -virtual-pixel background -background white +distort AffineProjection '%(sx)f,%(rx)f,%(ry)f,%(sy)f,%(tx)f,%(ty)f' -crop %(w)sx%(h)s%(x)s%(y)s\! -flatten -compress lzw %(output_fn)s"%d)