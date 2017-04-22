#! /usr/bin/env python

"""Visualize scoremaps, with annotation overlays."""

import argparse

parser = argparse.ArgumentParser(description="Generate visualizations of score maps.")
parser.add_argument("stack", help="stack name")
parser.add_argument("setting", help="setting", type=int)
parser.add_argument("-b", type=int, help="First section")
parser.add_argument("-e", type=int, help="Last section")
parser.add_argument("-a", help="whether to add label text", action='store_true')
parser.add_argument("-d", help="downscale", type=int, default=32)

args = parser.parse_args()

stack = args.stack
classifier_id = args.setting
add_label_text = args.a
downscale = args.d

###############################################

import os
import sys
import time

from multiprocess import Pool

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *
from data_manager import *
from visualization_utilities import *

limits = metadata_cache['section_limits'][stack]
first_sec = args.b if args.b is not None else limits[0]
last_sec = args.e if args.e is not None else limits[1]

################################################

for sec in range(first_sec, last_sec+1):

    if is_invalid(stack=stack, sec=sec):
        continue

    actual_setting = resolve_actual_setting(setting=classifier_id, stack=stack, sec=sec)

    t = time.time()

    def save_scoremap(structure):
        viz_fp = DataManager.get_scoremap_viz_filepath(stack=stack, downscale=downscale, section=sec, structure=structure, setting=actual_setting)
        create_parent_dir_if_not_exists(viz_fp)
        try:
            if add_label_text:
                label_text = str(structure)
            else:
                label_text = None

            viz = scoremap_overlay_on(bg='original', stack=stack, sec=sec, structure=structure,
                                downscale=downscale, label_text=label_text, setting=actual_setting)
            imsave(viz_fp, img_as_ubyte(viz))
            upload_from_ec2_to_s3(viz_fp)
        except Exception as e:
            sys.stderr.write('%s\n' % e)
            return

    pool = Pool(NUM_CORES)
    pool.map(save_scoremap, all_known_structures)
    pool.close()
    pool.join()

    sys.stderr.write('Visualize scoremaps: %.2f seconds.\n' % (time.time() - t))
    # 7s for one structure, one section, single process
    # 20s for all structures, one section, 8 processes
