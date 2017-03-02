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

try:
    volume = DataManager.load_score_volume(stack=stack, structure=structure, downscale=downscale,
                                           classifier_setting=classifier_setting)

    t = time.time()

    gy_gx_gz = np.gradient(volume.astype(np.float32), 3, 3, 3)
    # 3.3 second - re-computing is much faster than loading
    # .astype(np.float32) is important;
    # Otherwise the score volume is type np.float16, np.gradient requires np.float32 and will have to convert which is very slow
    # 2s (float32) vs. 20s (float16)

    sys.stderr.write('Gradient %s: %f seconds\n' % (structure, time.time() - t))

    t = time.time()

    gx_fp = DataManager.get_score_volume_gradient_filepath(stack=stack, structure=structure, downscale=downscale, classifier_setting=classifier_setting, suffix='gx')
    gy_fp = DataManager.get_score_volume_gradient_filepath(stack=stack, structure=structure, downscale=downscale, classifier_setting=classifier_setting, suffix='gy')
    gz_fp = DataManager.get_score_volume_gradient_filepath(stack=stack, structure=structure, downscale=downscale, classifier_setting=classifier_setting, suffix='gz')

    create_if_not_exists(os.path.dirname(gx_fp))

    bp.pack_ndarray_file(gy_gx_gz[1], gx_fp)
    bp.pack_ndarray_file(gy_gx_gz[0], gy_fp)
    bp.pack_ndarray_file(gy_gx_gz[2], gz_fp)

    del gy_gx_gz

    sys.stderr.write('save %s: %f seconds\n' % (structure, time.time() - t))

except Exception as e:
    sys.stderr.write('%s\n' % e)
    sys.stderr.write('Error computing gradient for %s.\n' % structure)
