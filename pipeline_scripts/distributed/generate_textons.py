
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
parser.add_argument("interval", type=int, help="use every interval'th images to learn textons (default: %(default)s)", default=5)
parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
args = parser.parse_args()

dm = DataManager(generate_hierarchy=False, stack=args.stack_name, resol='x5',
	gabor_params_id=args.gabor_params_id, segm_params_id=args.segm_params_id, 
	vq_params_id=args.vq_params_id)

#============================================================

if dm.check_pipeline_result('textons', 'npy'):
	print "textons.npy already exists, skip"

else:

	import random
	from subprocess import check_output

	features_rotated_list = []

	s = check_output("ls %s" % os.path.join(os.environ['GORDON_DATA_DIR'], args.stack_name, 'x5'), shell=True)
	slide_indices = [int(f) for f in s.split('\n') if len(f) > 0]
	section_num = len(slide_indices)

	for i in range(0, section_num, args.interval):

		dm.set_slice(i)
		dm._load_image()

		features_rotated_one_image = dm.load_pipeline_result('features_rotated', 'npy')
		features_rotated_list.append(features_rotated_one_image[np.random.randint(features_rotated_one_image.shape[0], size=1000000), :])
	
	features_rotated = np.vstack(features_rotated_list)

	del features_rotated_list

	n_texton = 100

	# try:
	#     centroids = dm.load_pipeline_result('original_centroids', 'npy')

	# except:

	from sklearn.cluster import MiniBatchKMeans
	kmeans = MiniBatchKMeans(n_clusters=n_texton, batch_size=1000)
	# kmeans.fit(features_rotated_pca)
	kmeans.fit(features_rotated)
	centroids = kmeans.cluster_centers_
    # labels = kmeans.labels_

    # dm.save_pipeline_result(centroids, 'original_centroids', 'npy')

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