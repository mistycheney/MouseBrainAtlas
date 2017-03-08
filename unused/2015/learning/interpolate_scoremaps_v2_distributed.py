#!/usr/bin/env python

import sys
import os

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from preprocess_utility import *
from data_manager import *
from metadata import *

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

exclude_nodes = [33]

first_sec, last_sec = DataManager.load_cropbox(stack)[4:]
# anchor_fn = DataManager.load_anchor_filename(stack)
# filenames_to_sections, sections_to_filenames = DataManager.load_sorted_filenames(stack)
#
# fn_list = [sections_to_filenames[sec] for sec in range(first_sec, last_sec+1)]
# fn_list = [fn for fn in fn_list if fn not in ['Nonexisting', 'Rescan', 'Placeholder']]

run_distributed4(command='%(script_path)s %(stack)s %%(first_sec)d %%(last_sec)d' % \
                {'script_path': os.path.join(os.environ['REPO_DIR'], 'learning') + '/interpolate_scoremaps_v2.py',
                'stack': stack},
                kwargs_list=dict(sections=range(first_sec, last_sec+1)),
                exclude_nodes=exclude_nodes,
                argument_type='partition')

sys.stderr.write('done in %f seconds\n' % (time.time() - t))
