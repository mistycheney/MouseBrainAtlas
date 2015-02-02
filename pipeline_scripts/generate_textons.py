#============================================================
from utilities import *
import os
import argparse
import sys
from joblib import Parallel, delayed


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

dm.set_image(args.stack_name, 'x5', args.slice_ind)

#============================================================


if dm.check_pipeline_result('textons', 'npy'):
	print "textons.npy already exists, skip"

else:
	
	n_texton = 100
	# n_texton = 10

	features_rotated = dm.load_pipeline_result('features_rotated', 'npy')

	try:
	    centroids = dm.load_pipeline_result('original_centroids', 'npy')

	except:
	    
	    from sklearn.cluster import MiniBatchKMeans
	    kmeans = MiniBatchKMeans(n_clusters=n_texton, batch_size=1000)
	    # kmeans.fit(features_rotated_pca)
	    kmeans.fit(features_rotated)
	    centroids = kmeans.cluster_centers_
	    # labels = kmeans.labels_

	    dm.save_pipeline_result(centroids, 'original_centroids', 'npy')

	from scipy.cluster.hierarchy import fclusterdata
	cluster_assignments = fclusterdata(centroids, 1.15, method="complete", criterion="inconsistent")
	# cluster_assignments = fclusterdata(centroids, 80., method="complete", criterion="distance")

	reduced_centroids = np.array([centroids[cluster_assignments == i].mean(axis=0) for i in set(cluster_assignments)])

	n_reduced_texton = len(reduced_centroids)
	print n_reduced_texton, 'reduced textons'

	from sklearn.cluster import MiniBatchKMeans
	kmeans = MiniBatchKMeans(n_clusters=n_reduced_texton, batch_size=1000, init=reduced_centroids)
	# kmeans.fit(features_rotated_pca)
	kmeans.fit(features_rotated)
	final_centroids = kmeans.cluster_centers_
	# labels = kmeans.labels_

	dm.save_pipeline_result(reduced_centroids, 'textons', 'npy')