#! /usr/bin/env python

import sys
import os

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from data_manager import *
from metadata import *
from distributed_utilities import *

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Rescale image')

parser.add_argument("input_spec", type=str, help="Input image name")
parser.add_argument('out_resol', type=str, help='')
parser.add_argument("-f", "--rescale_factor", type=float, help="Rescale factor")
parser.add_argument("-W", "--width", type=int, help="Width")
parser.add_argument("-H", "--height", type=int, help="Height")
args = parser.parse_args()

input_spec = load_ini(args.input_spec)
image_name_list = input_spec['image_name_list']
stack = input_spec['stack']
prep_id = input_spec['prep_id']
if prep_id == 'None':
    prep_id = None
resol = input_spec['resol']
version = input_spec['version']
if version == 'None':
    version = None

for img_name in image_name_list:

    t = time.time()

    in_fp = DataManager.get_image_filepath_v2(stack=stack, prep_id=prep_id, resol=resol, version=version, fn=img_name)
    out_fp = DataManager.get_image_filepath_v2(stack=stack, prep_id=prep_id, resol=args.out_resol, version=version, fn=img_name)
    create_parent_dir_if_not_exists(out_fp)
    
    img = imread(in_fp)
    print in_fp
    print img.dtype
    
    img_tb = img[::int(1./args.rescale_factor), ::int(1./args.rescale_factor)]
    imsave(out_fp, img_tb)

    # Alternative: ImageMagick introduces an artificial noisy stripe in the output image.
#     cmd = 'convert %(in_fp)s -scale 3.125%% %(out_fp)s' % {'in_fp': in_fp, 'out_fp': out_fp}
#     execute_command(cmd)
        
    sys.stderr.write("Rescale: %.2f seconds.\n" % (time.time() - t)) # ~20s / image



