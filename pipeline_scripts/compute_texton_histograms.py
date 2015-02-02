#============================================================
from utilities import *
import os
import argparse
import sys

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

dm = DataManager(data_dir=os.environ['GORDON_DATA_DIR'], 
  repo_dir=os.environ['GORDON_REPO_DIR'], 
  result_dir=os.environ['GORDON_RESULT_DIR'], labeling_dir=os.environ['GORDON_LABELING_DIR'])

dm.set_gabor_params(gabor_params_id='blueNisslWide')
dm.set_segmentation_params(segm_params_id='blueNisslRegular')
dm.set_vq_params(vq_params_id='blueNissl')

#============================================================

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