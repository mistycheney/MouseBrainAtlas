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

import itertools
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
    
centroids = dm.load_pipeline_result('textons', 'npy')
features_rotated = dm.load_pipeline_result('features_rotated', 'npy')

n_texton = len(centroids)

from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(n_clusters=n_texton, batch_size=1000, init=centroids)
kmeans.fit(features_rotated)
final_centroids = kmeans.cluster_centers_
labels = kmeans.labels_

    
textonmap = -1 * np.ones_like(dm.image, dtype=np.int)
textonmap[dm.mask] = labels

dm.save_pipeline_result(textonmap, 'texMap', 'npy')


hc_colors = np.loadtxt('../visualization/100colors.txt')

vis = label2rgb(textonmap, colors=hc_colors, alpha=1.)

dm.save_pipeline_result(vis, 'texMap', 'png')