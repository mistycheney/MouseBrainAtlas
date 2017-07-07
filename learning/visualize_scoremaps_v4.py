#! /usr/bin/env python

"""Visualize scoremaps, with annotation overlays."""

import json
import argparse

parser = argparse.ArgumentParser(description="Generate visualizations of score maps.")
parser.add_argument("stack", type=str, help="stack")
parser.add_argument("filenames", type=str, help="Filenames")
parser.add_argument("classifier_id", type=int, help="classifier id")
parser.add_argument("downscale", type=int, help="downscale factor")
parser.add_argument("--cmap", type=str, help="colormap name", default='hot')
parser.add_argument("-a", help="whether to add label text", action='store_true')

args = parser.parse_args()

stack = args.stack
filenames = json.loads(args.filenames)
classifier_id = args.classifier_id
downscale = args.downscale
add_label_text = args.a
cmap_name = args.cmap

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

################################################

for fn in filenames:

    if is_invalid(stack=stack, fn=fn):
        continue

    actual_setting = resolve_actual_setting(setting=classifier_id, stack=stack, fn=fn)

    t = time.time()

    def save_scoremap(structure):
        viz_fp = DataManager.get_scoremap_viz_filepath(stack=stack, downscale=downscale, fn=fn, structure=structure, classifier_id=actual_setting)
        create_parent_dir_if_not_exists(viz_fp)
        
        try:
            if add_label_text:
                label_text = str(structure)
            else:
                label_text = None
            viz = scoremap_overlay_on(bg='original', stack=stack, fn=fn, structure=structure,
                                out_downscale=downscale, label_text=label_text, classifier_id=actual_setting,
                                     cmap_name=cmap_name)
            imsave(viz_fp, img_as_ubyte(viz))
            upload_to_s3(viz_fp)
        except Exception as e:
            # raise e
            sys.stderr.write('%s\n' % e.message)
            return

    # for s in all_known_structures:
        # save_scoremap(s)

    pool = Pool(NUM_CORES)
    pool.map(save_scoremap, all_known_structures)
    pool.close()
    pool.join()

    sys.stderr.write('Visualize scoremaps: %.2f seconds.\n' % (time.time() - t))
    # 7s for one structure, one section, single process
    # 20s for all structures, one section, 8 processes
