#! /usr/bin/env python

"""Visualize scoremaps, with annotation overlays."""

import argparse

parser = argparse.ArgumentParser(description="Generate visualizations of score maps.")
parser.add_argument("stack", help="stack name, e.g. MD593")
parser.add_argument("-b", type=int, help="beginning slide (default: first_detect_sec)", default=0)
parser.add_argument("-e", type=int, help="ending slide (default: last_detect_sec)", default=-1)
parser.add_argument("-a", help="add annotation", action='store_true')

args = parser.parse_args()

stack = args.stack
first_sec = first_detect_sec if args.b == 0 else args.b
last_sec = last_detect_sec if args.e == -1 else args.e
add_annotation = args.a

import os
import sys
sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *

from visualization_utilities import *

import time

##################################################

if not add_annotation:
    export_filepath_fmt = scoremapViz_rootdir + '/%(name)s/%(stack)s/%(stack)s_%(sec)04d_roi1_scoremapViz_%(name)s.jpg'
    export_scoremaps('original', stack, range(first_sec, last_sec+1),
                     set(labels_unsided) - {'outerContour'}, 8,
                     export_filepath_fmt=export_filepath_fmt, label_text=True)
else:
    outputViz_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_scoremapPlusAnnotationViz'
    export_filepath_fmt = outputViz_rootdir + '/%(name)s/%(stack)s/%(stack)s_%(sec)04d_roi1_scoremapPlusAnnotationViz_%(name)s_%(annofn)s.jpg'
    export_scoremapPlusAnnotationVizs('original', stack, range(first_sec, last_sec+1),
                                      set(labels_unsided) - {'outerContour'}, 8, export_filepath_fmt=export_filepath_fmt,
                                      users=['yuncong', 'localAdjusted', 'autoAnnotate', 'globalAligned'])
