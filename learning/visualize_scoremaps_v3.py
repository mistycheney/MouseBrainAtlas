#! /usr/bin/env python

"""Visualize scoremaps, with annotation overlays."""

import argparse

parser = argparse.ArgumentParser(description="Generate visualizations of score maps.")
parser.add_argument("stack", help="stack name, e.g. MD593")
parser.add_argument("setting", help="setting", type=int)
parser.add_argument("-b", type=int, help="First section")
parser.add_argument("-e", type=int, help="Last section")
parser.add_argument("-a", help="add annotation", action='store_true')
parser.add_argument("-d", help="downscale", type=int, default=8)

args = parser.parse_args()

stack = args.stack
setting = args.setting
add_annotation = args.a
downscale = args.d

import os
import sys
sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *
from data_manager import *

limits = metadata_cache['section_limits'][stack]
first_sec = args.b if args.b is not None else limits[0]
last_sec = args.e if args.e is not None else limits[1]

from visualization_utilities import *
from multiprocess import Pool

import time

################################################

for sec in range(first_sec, last_sec+1):

    t = time.time()

    bg = imread(DataManager.get_image_filepath(stack=stack, section=sec, resol='lossless', version='compressed'))

    def f(structure):
        viz_fp = DataManager.get_scoremap_viz_filepath(stack=stack, section=sec, structure=structure, setting=setting)
        try:
            viz = scoremap_overlay_on(bg=bg, stack=stack, sec=sec, structure=structure, downscale=downscale, label_text=add_annotation, setting=setting)
            create_if_not_exists(os.path.dirname(viz_fp))
            imsave(viz_fp, img_as_ubyte(viz))
        except Exception as e:
            sys.stderr.write('%s\n' % e)
            return

    pool = Pool(8)
    pool.map(f, all_known_structures)
    pool.close()
    pool.join()

    sys.stderr.write('Visualize scoremaps: %.2f seconds.\n' % (time.time() - t))
    # 7s for one structure, one section, single process
    # 20s for all structures, one section, 8 processes
