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

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='')
parser.add_argument("stack", type=str, help="Fixed stack name")
parser.add_argument("structure", type=str, help="Structure")
parser.add_argument("classifier_setting", type=int, help="classifier_setting")
args = parser.parse_args()

stack = args.stack
structure = args.structure
classifier_setting = args.classifier_setting

downscale = 32
voxel_z_size = SECTION_THICKNESS/(XY_PIXEL_DISTANCE_LOSSLESS * downscale)

def load_scoremap_worker(stack, sec, structure, downscale, classifier_setting):
    try:
        actual_setting = resolve_actual_setting(setting=classifier_setting, stack=stack, sec=sec)
        sm = DataManager.load_scoremap(stack=stack, section=sec, structure=structure, downscale=downscale, setting=actual_setting)
        return sm
    except:
        pass

def load_scoremaps_multiple_sections_parallel(sections, stack, structure, downscale, classifier_setting):
    pool = Pool(12)
    scoremaps = pool.map(lambda sec: load_scoremap_worker(stack, sec, structure, downscale, classifier_setting=classifier_setting),
                                     sections)
    pool.close()
    pool.join()
    return {sec: sm for sec, sm in zip(sections, scoremaps) if sm is not None}

first_sec, last_sec = metadata_cache['section_limits'][stack]

nissl_sections = []
for sec in range(first_sec, last_sec):
    fn = metadata_cache['sections_to_filenames'][stack][sec]
    if not is_invalid(fn) and fn.split('-')[1][0] == 'N':
        nissl_sections.append(sec)

print structure

t = time.time()
scoremaps = load_scoremaps_multiple_sections_parallel(stack=stack, sections=nissl_sections,
                                                      structure=structure, downscale=downscale,
                                                      classifier_setting=classifier_setting)

if len(scoremaps) < 2:
    sys.stderr.write('Number of valid scoremaps for %s is less than 2.\n' % structure)
    sys.exit(1)

sys.stderr.write('Load scoremaps: %.2f seconds\n' % (time.time() - t)) # 10-40s (down=32, 12 processes)

t = time.time()
score_volume, score_volume_bbox = images_to_volume(images=scoremaps, voxel_size=(1, 1, voxel_z_size*2),
                                                   first_sec=np.min(nissl_sections)-1,
                                                   last_sec=np.max(nissl_sections)-1)
sys.stderr.write('Create score volume: %.2f seconds\n' % (time.time() - t)) # 2s

#         t = time.time()

output_dir = create_if_not_exists(os.path.join(VOLUME_ROOTDIR, stack, 'score_volumes'))

score_volume_filepath = DataManager.get_score_volume_filepath(stack=stack, downscale=downscale,
                                                              structure=structure,
                                                              classifier_setting=classifier_setting)
create_if_not_exists(os.path.dirname(score_volume_filepath))
bp.pack_ndarray_file(score_volume.astype(np.float16), score_volume_filepath)

score_volume_bbox_filepath = DataManager.get_score_volume_bbox_filepath(stack=stack, downscale=downscale, structure=structure,
                                                                       classifier_setting=classifier_setting)
np.savetxt(score_volume_bbox_filepath, np.array(score_volume_bbox)[None], fmt='%d')

del score_volume, scoremaps
