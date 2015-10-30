#! /usr/bin/env python

import os
import argparse
import sys
import time

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Compute texton histograms')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("slice_ind", type=int, help="slice index")
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

# if dm.check_pipeline_result('texHist'):
if False:
	print "texHist.npy already exists, skip"

else:
	
    print 'computing histograms ...',
    t = time.time()

    segmentation = dm.load_pipeline_result('segmentation')
    n_superpixels = segmentation.max() + 1

    def texton_histogram_worker(i):
        # return np.bincount(textonmap[(segmentation == i)&(textonmap != -1)], minlength=n_texton)
        return np.bincount(textonmap[segmentation == i], minlength=n_texton)

    r = Parallel(n_jobs=8)(delayed(texton_histogram_worker)(i) for i in range(n_superpixels))
    sp_texton_hist = np.array(r)
    sp_texton_hist_normalized = sp_texton_hist.astype(np.float) / sp_texton_hist.sum(axis=1)[:, np.newaxis] # denom might be invalid

    dm.save_pipeline_result(sp_texton_hist_normalized, 'texHist')

    print 'done in', time.time() - t, 'seconds'

