#! /usr/bin/env python

import numpy as np

import sys
import os

sys.path.append('/shared/MouseBrainAtlas/utilities')
from utilities2015 import *
from metadata import *
from data_manager import *

import time

from multiprocess import Pool

from conversion import images_to_volume

stack = sys.argv[1]
label = sys.argv[2]

train_sample_scheme = 1
# svm_suffix = 'trainSampleScheme_%d'%train_sample_scheme

paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', 'SNC', 'SNR', '3N', '4N',
                    'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
singular_structures = ['AP', '12N', 'RtTg', 'SC', 'IC']
structures = paired_structures + singular_structures

downscale_factor = 32
voxel_z_size = section_thickness/(xy_pixel_distance_lossless * downscale_factor)

def load_scoremap_worker(sec, stack, label, downscale_factor):
    try:
        scoremap = DataManager.load_scoremap(stack=stack, section=sec, label=label,
                                            downscale_factor=downscale_factor,
                                            train_sample_scheme=train_sample_scheme)
        return (sec-1, scoremap)
    except:
        pass

def load_scoremaps_parallel(sections, stack, label, downscale_factor):
    pool = Pool(4)
    index_scoremap_tuples = pool.map(lambda sec: load_scoremap_worker(sec, stack, label, downscale_factor),
                                     sections)
    return dict(filter(None, index_scoremap_tuples))

first_sec, last_sec = metadata_cache['section_limits'][stack]


t = time.time()
scoremaps = load_scoremaps_parallel(stack=stack, sections=range(first_sec, last_sec+1), label=label, downscale_factor=downscale_factor)
sys.stderr.write('Load scoremaps: %.2f seconds\n' % (time.time() - t))

score_volume, score_volume_bbox = images_to_volume(images=scoremaps, voxel_size=(1, 1, voxel_z_size), first_sec=first_sec-1, last_sec=last_sec-1)

output_dir = create_if_not_exists(os.path.join(VOLUME_ROOTDIR, stack, 'score_volumes'))

score_volume_filepath = DataManager.get_score_volume_filepath(stack=stack, downscale=downscale_factor, label=label, train_sample_scheme=train_sample_scheme)
bp.pack_ndarray_file(score_volume.astype(np.float16), score_volume_filepath)

score_volume_bbox_filepath = DataManager.get_score_volume_bbox_filepath(stack=stack, downscale=downscale_factor, label=label)
np.savetxt(score_volume_bbox_filepath, np.array(score_volume_bbox)[None])

sys.stderr.write('Create score volume: %.2f seconds\n' % (time.time() - t)) # 40 seconds - load scoremap 2 processes

del score_volume, scoremaps
