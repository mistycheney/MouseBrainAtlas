#! /usr/bin/env python

import sys
import os
import time

import numpy as np
from multiprocess import Pool

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *
from data_manager import *
from conversion import images_to_volume

##################################################

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='')
parser.add_argument("stack", type=str, help="Fixed stack name")
parser.add_argument("structure", type=str, help="Structure")
parser.add_argument("classifier_id", type=int, help="classifier_id")
parser.add_argument("downscale", type=int, help="volume downscale")
args = parser.parse_args()

stack = args.stack
structure = args.structure
classifier_id = args.classifier_id
downscale = args.downscale

##################################################

try:
    volume = DataManager.load_original_volume(stack=stack, structure=structure, downscale=downscale,
                                     classifier_setting=classifier_id, volume_type='score')

    t = time.time()

    gy_gx_gz = np.gradient(volume.astype(np.float32), 3, 3, 3)
    # 3.3 second - re-computing is much faster than loading
    # .astype(np.float32) is important;
    # Otherwise the score volume is type np.float16, np.gradient requires np.float32 and will have to convert which is very slow
    # 2s (float32) vs. 20s (float16)

    sys.stderr.write('Gradient %s: %f seconds\n' % (structure, time.time() - t))

    t = time.time()

    gx_fp = DataManager.get_volume_gradient_filepath(stack=stack, structure=structure, downscale=downscale, classifier_setting=classifier_id, suffix='gx')
    gy_fp = DataManager.get_volume_gradient_filepath(stack=stack, structure=structure, downscale=downscale, classifier_setting=classifier_id, suffix='gy')
    gz_fp = DataManager.get_volume_gradient_filepath(stack=stack, structure=structure, downscale=downscale, classifier_setting=classifier_id, suffix='gz')

    create_parent_dir_if_not_exists(gx_fp)

    bp.pack_ndarray_file(gy_gx_gz[1], gx_fp)
    bp.pack_ndarray_file(gy_gx_gz[0], gy_fp)
    bp.pack_ndarray_file(gy_gx_gz[2], gz_fp)
    
    upload_from_ec2_to_s3(gx_fp)
    upload_from_ec2_to_s3(gy_fp)
    upload_from_ec2_to_s3(gz_fp)

    del gy_gx_gz

    sys.stderr.write('save %s: %f seconds\n' % (structure, time.time() - t))

except Exception as e:
    sys.stderr.write('%s\n' % e)
    sys.stderr.write('Error computing gradient for %s.\n' % structure)
