#!/usr/bin/env python

import sys
import os
import cPickle as pickle

from preprocess_utility import *
import time

# stack = sys.argv[1]
# first_sec, last_sec = map(int, sys.argv[2:4])

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generate mask for thumbnail images, distributed')
parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("first_sec", type=int, help="first section")
parser.add_argument("last_sec", type=int, help="last section")
args = parser.parse_args()

stack = args.stack_name
first_sec = args.first_sec
last_sec = args.last_sec

t = time.time()
sys.stderr.write('generating mask ...')

exclude_nodes = [33]

run_distributed3(command='%(script_path)s %(stack)s %(input_dir)s %%(f)d %%(l)d'%\
                            {'script_path': os.path.join(os.environ['REPO_DIR'], 'elastix') + '/generate_thumbnail_masks.py', 
                            'stack': stack,
                            # 'input_dir': os.path.join(DATAPROC_DIR, stack+'_thumbnail_aligned_cropped')
                            'input_dir': os.path.join(os.environ['DATA_DIR'], stack+'_thumbnail_aligned')
                            }, 
                first_sec=first_sec,
                last_sec=last_sec,
                exclude_nodes=exclude_nodes,
                take_one_section=False)

sys.stderr.write('done in %f seconds\n' % (time.time() - t))
