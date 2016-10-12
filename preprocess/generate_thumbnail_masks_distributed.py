#!/usr/bin/env python

import sys
import os

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from preprocess_utility import *
import time

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generate mask for thumbnail images, distributed')
parser.add_argument("stack_name", type=str, help="stack name")
args = parser.parse_args()

stack = args.stack_name

t = time.time()
sys.stderr.write('generating masks ...')

exclude_nodes = [33, 47]

input_dir = '/home/yuncong/CSHL_data/%(stack)s' % dict(stack=stack)
output_dir = create_if_not_exists('/home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_mask_unsorted' % dict(stack=stack))

from glob import glob
all_fns = [os.path.splitext(os.path.basename(fn))[0] for fn in glob(input_dir+'/*.tif')]

run_distributed4(command='%(script_path)s %(stack)s %(input_dir)s \'%%(filenames)s\' %(output_dir)s' % \
                {'script_path': os.path.join(os.environ['REPO_DIR'], 'preprocess') + '/generate_thumbnail_masks_v2.py',
                'stack': stack,
                'input_dir': input_dir,
                'output_dir': output_dir},
                kwargs_list=dict(filenames=all_fns),
                exclude_nodes=exclude_nodes,
                argument_type='list2')

sys.stderr.write('done in %f seconds\n' % (time.time() - t))
