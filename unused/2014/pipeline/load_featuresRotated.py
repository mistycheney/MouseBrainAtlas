#! /usr/bin/env python

import os
import argparse
import sys
import time

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Rotate features using Gabor filters')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("slice_ind", type=int, help="slice index")
parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
# parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
# parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
args = parser.parse_args()


from joblib import Parallel, delayed

sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], 'utilities'))
from utilities2015 import *

dm = DataManager(gabor_params_id=args.gabor_params_id, 
                 # segm_params_id=args.segm_params_id, 
                 # vq_params_id=args.vq_params_id,
                 stack=args.stack_name, 
                 section=args.slice_ind)

#============================================================

t = time.time()
sys.stderr.write('load rotated features ...')

features_rotated = np.empty((54396576*2, 99), np.half)

features_rotated[:54396576,] = bp.unpack_ndarray_file(os.environ['GORDON_RESULT_DIR']+'/featuresRotated_0.bp')
features_rotated[54396576:,] = bp.unpack_ndarray_file(os.environ['GORDON_RESULT_DIR']+'/featuresRotated_1.bp')

# bp.pack_ndarray_file(features_rotated, os.environ['GORDON_RESULT_DIR']+'/featuresRotated.bp')

dm.save_pipeline_result(features_rotated, 'featuresRotated')

sys.stderr.write('done in %f seconds\n' % (time.time() - t))