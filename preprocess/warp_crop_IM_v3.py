#! /usr/bin/env python

import sys
import os
import numpy as np
import json

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import execute_command, create_parent_dir_if_not_exists

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Warp and crop images.')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("input_fp", type=str, help="input filename")
parser.add_argument("output_fp", type=str, help="output filename")
parser.add_argument("transform", type=str, help="thumbnail transform")
parser.add_argument("suffix", type=str, help="suffix")
parser.add_argument("x", type=int, help="x on thumbnail", default=0)
parser.add_argument("y", type=int, help="y on thumbnail", default=0)
parser.add_argument("w", type=int, help="w on thumbnail", default=2000)
parser.add_argument("h", type=int, help="h on thumbnail", default=1500)
parser.add_argument("background_color", type=str, help="background color (black or white)", default='white')
args = parser.parse_args()

stack = args.stack_name
input_fp = args.input_fp
output_fp = args.output_fp
transform = np.reshape(map(np.float, args.transform.split(',')), (3,3))
background_color = args.background_color
sys.stderr.write("Background color: %s\n" % background_color)
suffix = args.suffix

x = args.x
y = args.y
w = args.w
h = args.h

if suffix == 'thumbnail':
    scale_factor = 1
elif suffix == 'lossy':
    scale_factor = 16
elif suffix == 'lossless':
    scale_factor = 32

T2 = transform.copy()
T2[:2,2] = transform[:2, 2] * scale_factor
T = np.linalg.inv(T2)

create_parent_dir_if_not_exists(output_fp)

execute_command("convert \"%(input_fp)s\" -virtual-pixel background -background %(bg_color)s +distort AffineProjection '%(sx)f,%(rx)f,%(ry)f,%(sy)f,%(tx)f,%(ty)f' -crop %(w)sx%(h)s%(x)s%(y)s\! -flatten -compress lzw \"%(output_fp)s\"" % \
                {'sx':T[0,0],
     'sy':T[1,1],
     'rx':T[1,0],
     'ry':T[0,1],
     'tx':T[0,2],
     'ty':T[1,2],
     'input_fp': input_fp,
     'output_fp': output_fp,
     'x': '+' + str(x * scale_factor) if int(x) >= 0 else str(x * scale_factor),
     'y': '+' + str(y * scale_factor) if int(y) >= 0 else str(y * scale_factor),
     'w': str(w * scale_factor),
     'h': str(h * scale_factor),
     'bg_color': background_color
    })