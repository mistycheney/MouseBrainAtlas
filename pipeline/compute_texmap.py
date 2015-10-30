#! /usr/bin/env python

import os
import argparse
import sys
import time

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Compute texton map')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("slice_ind", type=int, help="slice index")
parser.add_argument("-t", "--texton_path", type=str, help="path to textons.npy", default='')
parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
args = parser.parse_args()

from joblib import Parallel, delayed

sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], 'utilities'))
from utilities2015 import *

dm = DataManager(data_dir=os.environ['GORDON_DATA_DIR'], 
                 repo_dir=os.environ['GORDON_REPO_DIR'], 
                 result_dir=os.environ['GORDON_RESULT_DIR'], 
                 labeling_dir=os.environ['GORDON_LABELING_DIR'],
                 gabor_params_id=args.gabor_params_id, 
                 segm_params_id=args.segm_params_id, 
                 vq_params_id=args.vq_params_id,
                 stack=args.stack_name, 
                 section=args.slice_ind)

#==================================================

if dm.check_pipeline_result('texMap') and dm.check_pipeline_result('texMapViz'):
# if False:
    print "texMap.npy already exists, skip"

    textonmap = dm.load_pipeline_result('texMap')
    n_texton = textonmap.max() + 1

else:

    print 'loading centroids and features ...',
    t = time.time()

    if args.texton_path == '':
        centroids = dm.load_pipeline_result('textons')
    else:
        centroids = np.load(args.texton_path)

    features_rotated = dm.load_pipeline_result('featuresRotated')
    print 'done in', time.time() - t, 'seconds'

    n_texton = len(centroids)

    print 'assign textons ...',
    t = time.time()

    from scipy.spatial.distance import cdist

    label_list = []
    for i in range(0, len(features_rotated), 5000000):
        Ds = Parallel(n_jobs=16)(delayed(cdist)(fs, centroids) 
                                for fs in np.array_split(features_rotated[i:i+5000000], 16))
        for D in Ds:
            label_list.append(np.argmin(D, axis=1))

    labels = np.concatenate(label_list)

    print 'done in', time.time() - t, 'seconds'

    textonmap = -1 * np.ones((dm.image_height, dm.image_width), dtype=np.int8)
    textonmap[dm.mask] = labels

    dm.save_pipeline_result(textonmap, 'texMap')

    colors = (np.loadtxt(dm.repo_dir + '/visualization/100colors.txt') * 255).astype(np.uint8)
    
    textonmap_viz = np.zeros((dm.image_height, dm.image_width, 3), np.uint8)
    textonmap_viz[dm.mask] = colors[textonmap[dm.mask]]
    dm.save_pipeline_result(textonmap_viz, 'texMapViz')