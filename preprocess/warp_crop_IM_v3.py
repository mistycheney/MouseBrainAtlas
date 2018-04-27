#! /usr/bin/env python

import sys
import os
import numpy as np
import json

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import execute_command, create_parent_dir_if_not_exists
from metadata import orientation_argparse_str_to_imagemagick_str

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Warp and crop images.')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("input_fp", type=str, help="input filename")
parser.add_argument("output_fp", type=str, help="output filename")
parser.add_argument("transform", type=str, help="transform for thumbnail resolution")
parser.add_argument("suffix", type=str, help="resolution, thumbnail or raw")
parser.add_argument("x_tb", type=int, help="x on thumbnail", default=0)
parser.add_argument("y_tb", type=int, help="y on thumbnail", default=0)
parser.add_argument("w_tb", type=int, help="w on thumbnail", default=2000)
parser.add_argument("h_tb", type=int, help="h on thumbnail", default=1500)
parser.add_argument("background_color", type=str, help="background color (black or white)", default='white')
parser.add_argument("-r", "--init_rotate", type=str, help="escaped imagemagick convert option string for initial flipping and rotation", default='')
args = parser.parse_args()

stack = args.stack_name
input_fp = args.input_fp
output_fp = args.output_fp
transform = np.reshape(map(np.float, args.transform.split(',')), (3,3))
background_color = args.background_color
sys.stderr.write("Background color: %s\n" % background_color)
suffix = args.suffix

if args.init_rotate == '':
    init_rotate = ''
else:
    init_rotate = orientation_argparse_str_to_imagemagick_str[args.init_rotate]
    
x_tb = args.x_tb
y_tb = args.y_tb
w_tb = args.w_tb
h_tb = args.h_tb

if suffix == 'thumbnail':
    scale_factor = 1
# elif suffix == 'lossy':
#     scale_factor = 16
elif suffix == 'raw':
    scale_factor = 32
else:
    raise "Unknown resolution: %s" % suffix

T2 = transform.copy()
T2[:2,2] = transform[:2, 2] * scale_factor
T = np.linalg.inv(T2)

create_parent_dir_if_not_exists(output_fp)


execute_command("convert \"%(input_fp)s\" %(init_rotate)s +repage -virtual-pixel background -background %(bg_color)s +distort AffineProjection '%(sx)f,%(rx)f,%(ry)f,%(sy)f,%(tx)f,%(ty)f' -crop %(w)sx%(h)s%(x)s%(y)s\! -flatten -compress lzw \"%(output_fp)s\"" % \
                {'init_rotate':init_rotate,
                    'sx':T[0,0],
     'sy':T[1,1],
     'rx':T[1,0],
     'ry':T[0,1],
     'tx':T[0,2],
     'ty':T[1,2],
     'input_fp': input_fp,
     'output_fp': output_fp,
     'x': '+' + str(x_tb * scale_factor) if int(x_tb) >= 0 else str(x_tb * scale_factor),
     'y': '+' + str(y_tb * scale_factor) if int(y_tb) >= 0 else str(y_tb * scale_factor),
     'w': str(w_tb * scale_factor),
     'h': str(h_tb * scale_factor),
     'bg_color': background_color
    })