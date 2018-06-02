#! /usr/bin/env python

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Generate mask for thumbnail images, distributed')
parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("--first_sec", type=int, help="first section (default: first detection section)")
parser.add_argument("--last_sec", type=int, help="last section (default: last detection section)")
parser.add_argument("--interval", type=int, help="interval (default: %(default)d)", default=1)
parser.add_argument("--train", dest='train', help="train", action='store_true')
parser.add_argument("--test", dest='test', help="ROI", action='store_true')
args = parser.parse_args()

import sys
import os
sys.path.append(os.environ['REPO_DIR'] + '/utilities/')
from preprocess_utility import *
from metadata import *
import time

stack = args.stack_name
f0, l0 = detect_bbox_range_lookup[stack]
l = args.last_sec if args.last_sec is not None else l0
f = args.first_sec if args.first_sec is not None else f0

t = time.time()
sys.stderr.write('Computing SPM histograms ...')

exclude_nodes = [33]

run_distributed3(command='%(script_path)s %(stack)s %%(secind)d %(train)s %(test)s'%\
                            {'script_path': os.path.join(os.environ['REPO_DIR'], 'spm', 'compute_sift_labelmap.py'),
                            'stack': args.stack_name,
                            'train': '--train' if args.train else '',
                            'test': '--test' if args.test else ''},
                section_list=range(f, l+1, args.interval),
                exclude_nodes=exclude_nodes,
                take_one_section=True)

sys.stderr.write('done in %f seconds\n' % (time.time() - t))
