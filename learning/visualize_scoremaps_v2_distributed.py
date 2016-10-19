#! /usr/bin/env python

"""Visualize scoremaps, with annotation overlays."""

import argparse

parser = argparse.ArgumentParser(description="Generate visualizations of score maps.")
parser.add_argument("stack", help="stack name, e.g. MD593")
parser.add_argument("-a", help="add annotation", action='store_true')

args = parser.parse_args()

import os
import sys
import time

stack = args.stack
add_annotation = args.a

##################################################

t = time.time()

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from preprocess_utility import *
from data_manager import *
from metadata import *

exclude_nodes = [33]
# first_sec, last_sec = DataManager.load_cropbox(stack)[4:]
first_sec, last_sec = (200, 210)

run_distributed4(command='%(script_path)s %(stack)s -b %%(first_sec)d -e %%(last_sec)d %(add_annotation)s' % \
                {'script_path': os.path.join(os.environ['REPO_DIR'], 'learning') + '/visualize_scoremaps_v2.py',
                'stack': stack,
                'add_annotation': '-a' if add_annotation else ''},
                kwargs_list=dict(sections=range(first_sec, last_sec+1)),
                exclude_nodes=exclude_nodes,
                argument_type='partition')

sys.stderr.write('done in %f seconds\n' % (time.time() - t))
