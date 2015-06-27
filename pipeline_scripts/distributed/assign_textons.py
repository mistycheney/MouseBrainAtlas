
import os
import argparse
import sys

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

try:
	textonmap = dm.load_pipeline_result('texMap', 'npy')
	print "texMap.npy already exists, skip"

except Exception as e:

	centroids = dm.load_pipeline_result('textons', 'npy')
	features_rotated = dm.load_pipeline_result('features_rotated', 'npy')

	n_texton = len(centroids)

	from sklearn.cluster import MiniBatchKMeans

	kmeans = MiniBatchKMeans(n_clusters=n_texton, batch_size=1000, init=centroids)
	kmeans.fit(features_rotated)
	final_centroids = kmeans.cluster_centers_
	labels = kmeans.labels_

	dm._load_image()
	textonmap = -1 * np.ones_like(dm.image, dtype=np.int)
	textonmap[dm.mask] = labels

	dm.save_pipeline_result(textonmap, 'texMap', 'npy')

hc_colors = np.loadtxt(dm.repo_dir + '/visualization/100colors.txt')
vis = label2rgb(textonmap, colors=hc_colors, alpha=1.)
dm.save_pipeline_result(vis, 'texMap', 'png')