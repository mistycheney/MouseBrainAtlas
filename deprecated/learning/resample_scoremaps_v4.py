#! /usr/bin/env python

import sys
import os
import time

from multiprocess import Pool
import numpy as np
from scipy.ndimage.interpolation import map_coordinates

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *
from data_manager import *
from learning_utilities import *

############################

import json
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Resample scoremaps')

parser.add_argument("stack", type=str, help="stack")
parser.add_argument("filenames", type=str, help="Filenames")
parser.add_argument("classifier_id", type=int, help="classifier id")
parser.add_argument("downscale", type=int, help="downscale factor")
args = parser.parse_args()

############################

stack = args.stack
filenames = json.loads(args.filenames)
classifier_id = args.classifier_id
downscale = args.downscale

classifier_properties = classifier_settings.loc[classifier_id]
input_img_version = classifier_properties['input_img_version']
cnn_model = dataset_settings.loc[int(classifier_settings.loc[classifier_id]['train_set_id'].split('/')[0])]['network_model']

patch_size, spacing, w, h = get_default_gridspec(stack)
half_size = patch_size/2

def resample(fn):

    if is_invalid(stack=stack, fn=fn):
        return

#         t = time.time()

    try:
        _, sample_locations_roi = DataManager.load_dnn_feature_locations(stack=stack, 
                                        model_name=cnn_model, fn=fn, input_img_version=input_img_version)
    except Exception as e:
        sys.stderr.write('Error loading patch locations for %s: %s.\n' % (fn, str(e)))
        return

    actual_setting = resolve_actual_setting(setting=classifier_id, stack=stack, fn=fn)

    downscaled_grid_y = np.arange(0, h, downscale)
    downscaled_grid_x = np.arange(0, w, downscale)
    downscaled_ny = len(downscaled_grid_y)
    downscaled_nx = len(downscaled_grid_x)

    for structure in all_structures_with_classifiers:
        try:
            sparse_scores = DataManager.load_sparse_scores(stack, fn=fn,
                                                           structure=structure, 
                                                           classifier_id=actual_setting)
        except Exception as e:
            sys.stderr.write('Error loading sparse scores for %s: %s\n' % (structure, str(e)))
            continue

        f_grid = np.zeros(((h-half_size)/spacing+1, (w-half_size)/spacing+1))
        a = (sample_locations_roi - half_size)/spacing
        f_grid[a[:,1], a[:,0]] = sparse_scores

        yinterps = (downscaled_grid_y - half_size)/float(spacing)
        xinterps = (downscaled_grid_x - half_size)/float(spacing)
        
        points_y, points_x = np.broadcast_arrays(yinterps.reshape(-1,1), xinterps)
        coord = np.c_[points_y.flat, points_x.flat]
        f_interp = map_coordinates(f_grid, coord.T, order=1)
        f_interp_2d = f_interp.reshape((downscaled_ny, downscaled_nx))                       

        scoremap_bp_filepath = \
        DataManager.get_downscaled_scoremap_filepath(stack=stack, fn=fn, 
                                                     structure=structure, 
                                                     classifier_id=actual_setting,
                                                    downscale=downscale)

        create_parent_dir_if_not_exists(scoremap_bp_filepath)
        bp.pack_ndarray_file(f_interp_2d.astype(np.float16), scoremap_bp_filepath)
        upload_to_s3(scoremap_bp_filepath)

#         sys.stderr.write('interpolate %d: %.2f seconds\n' % (sec, time.time() - t)) 

t = time.time()

pool = Pool(NUM_CORES/2)
# pool = Pool(1)
pool.map(resample, filenames)
pool.close()
pool.join()

sys.stderr.write('Resample: %.2f seconds\n' % (time.time() - t)) 