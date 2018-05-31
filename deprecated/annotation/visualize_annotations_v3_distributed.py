#! /usr/bin/env python

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generate mask for thumbnail images, distributed')
parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("--first_sec", type=int, help="first section (default: first brainstem section)")
parser.add_argument("--last_sec", type=int, help="last section (default: last brainstem section)")
args = parser.parse_args()

import sys
import os
sys.path.append(os.environ['REPO_DIR'] + '/utilities/')
from preprocess_utility import *
from metadata import *
import time

stack = args.stack_name
f0, l0 = section_range_lookup[stack]
l = args.last_sec if args.last_sec is not None else l0
f = args.first_sec if args.first_sec is not None else f0

t = time.time()
sys.stderr.write('Generating annotation visualization ...')

exclude_nodes = [33]

run_distributed3(command='%(script_path)s %(stack)s %%(secind)d'%\
                            {'script_path': os.path.join(os.environ['REPO_DIR'], 'annotation', 'visualize_annotations_v3.py'),
                            'stack': stack,
                            },
                first_sec=f,
                last_sec=l,
                exclude_nodes=exclude_nodes,
                take_one_section=True)

sys.stderr.write('done in %f seconds\n' % (time.time() - t)) # 500s / stack
