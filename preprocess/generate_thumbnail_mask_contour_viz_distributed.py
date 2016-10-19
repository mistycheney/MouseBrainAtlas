#!/usr/bin/env python

import sys
import os
import cPickle as pickle

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from preprocess_utility import *
import time

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generate mask contour viz for thumbnail images, distributed. Run this on Gordon.')
parser.add_argument("stack_name", type=str, help="stack name")
args = parser.parse_args()

stack = args.stack_name

t = time.time()
sys.stderr.write('generating mask contour viz ...')

exclude_nodes = [33, 47]

image_dir = '/home/yuncong/CSHL_data/%(stack)s' % dict(stack=stack)
mask_dir = '/home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_mask_unsorted' % dict(stack=stack)
output_dir = create_if_not_exists('/home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_maskContourViz_unsorted' % dict(stack=stack))

from glob import glob
all_fns = [os.path.splitext(os.path.basename(fn))[0] for fn in glob(image_dir+'/*.tif')]

run_distributed4(command='%(script_path)s %(stack)s %(image_dir)s %(mask_dir)s \'%%(filenames)s\' %(output_dir)s' % \
                {'script_path': os.path.join(os.environ['REPO_DIR'], 'preprocess') + '/generate_thumbnail_mask_contour_viz.py',
                'stack': stack,
                'image_dir': image_dir,
                'mask_dir': mask_dir,
                'output_dir': output_dir},
                kwargs_list=dict(filenames=all_fns),
                exclude_nodes=exclude_nodes,
                argument_type='list2')

sys.stderr.write('done in %f seconds\n' % (time.time() - t))
