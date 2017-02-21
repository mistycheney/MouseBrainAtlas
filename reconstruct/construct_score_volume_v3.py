#! /usr/bin/env python

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *
from data_manager import *

from multiprocess import Pool
from conversion import images_to_volume

stack = sys.argv[1]
structure = sys.argv[2]
setting = int(sys.argv[3])

downscale = 32
voxel_z_size = SECTION_THICKNESS/(XY_PIXEL_DISTANCE_LOSSLESS * downscale)

def load_scoremap_worker(stack, sec, structure, downscale, setting):
    try:
        scoremap = DataManager.load_scoremap(stack=stack, section=sec, structure=structure, downscale=downscale, setting=setting)
        return (sec-1, scoremap)
    except:
        pass

def load_scoremaps_multiple_sections_parallel(sections, stack, structure, downscale, setting):
    pool = Pool(12)
    index_scoremap_tuples = pool.map(lambda sec: load_scoremap_worker(stack, sec, structure, downscale, setting=setting),
                                     sections)
    return dict(filter(None, index_scoremap_tuples))

first_sec, last_sec = metadata_cache['section_limits'][stack]

print structure

t = time.time()
scoremaps = load_scoremaps_multiple_sections_parallel(stack=stack, sections=range(first_sec, last_sec+1), structure=structure, downscale=downscale, setting=setting)
if len(scoremaps) < 2:
    sys.stderr.write('Number of valid scoremaps for %s is less than 2.\n' % structure)
    sys.exit(1)
sys.stderr.write('Load scoremaps: %.2f seconds\n' % (time.time() - t)) # 10-40s (down=32, 12 processes)

t = time.time()
score_volume, score_volume_bbox = images_to_volume(images=scoremaps, voxel_size=(1, 1, voxel_z_size), first_sec=first_sec-1, last_sec=last_sec-1)
sys.stderr.write('Create score volume: %.2f seconds\n' % (time.time() - t)) # 2s

output_dir = create_if_not_exists(os.path.join(VOLUME_ROOTDIR, stack, 'score_volumes'))

score_volume_filepath = DataManager.get_score_volume_filepath(stack=stack, downscale=downscale, structure=structure, setting=setting)
bp.pack_ndarray_file(score_volume.astype(np.float16), score_volume_filepath)

score_volume_bbox_filepath = DataManager.get_score_volume_bbox_filepath(stack=stack, downscale=downscale, structure=structure)
np.savetxt(score_volume_bbox_filepath, np.array(score_volume_bbox)[None], fmt='%d')

del score_volume, scoremaps
