import os
import argparse
import sys
from joblib import Parallel, delayed

sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], 'pipeline_scripts'))

if os.environ['DATASET_VERSION'] == '2014':
	from utilities2014 import *
elif os.environ['DATASET_VERSION'] == '2015':
	from utilities import *

parser = argparse.ArgumentParser(
formatter_class=argparse.RawDescriptionHelpFormatter,
description='Execute feature extraction pipeline',
epilog="""
The following command processes image RS141_x5_0001.tif using the specified parameters.
python %s RS141 1 -g blueNisslWide -s blueNisslRegular -v blueNissl
"""%(os.path.basename(sys.argv[0]), ))

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("slice_ind", type=int, help="slice index")
parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
args = parser.parse_args()

dm = DataManager(generate_hierarchy=False, stack=args.stack_name, resol='x5', section=args.slice_ind,
	gabor_params_id=args.gabor_params_id, segm_params_id=args.segm_params_id, 
	vq_params_id=args.vq_params_id)

#============================================================

if dm.check_pipeline_result('texHist', 'npy'):
	print "texHist.npy already exists, skip"

else:
	
	segmentation = dm.load_pipeline_result('segmentation', 'npy')
	n_superpixels = len(np.unique(segmentation)) - 1

	textonmap = dm.load_pipeline_result('texMap', 'npy')

	n_texton = len(np.unique(textonmap)) - 1


	def texton_histogram_worker(i):
	    return np.bincount(textonmap[(segmentation == i)&(textonmap != -1)], minlength=n_texton)

	r = Parallel(n_jobs=16)(delayed(texton_histogram_worker)(i) for i in range(n_superpixels))
	sp_texton_hist = np.array(r)
	sp_texton_hist_normalized = sp_texton_hist.astype(np.float) / sp_texton_hist.sum(axis=1)[:, np.newaxis] # denom might be invalid

	dm.save_pipeline_result(sp_texton_hist_normalized, 'texHist', 'npy')