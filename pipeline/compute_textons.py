#! /usr/bin/env python

import os
import argparse
import sys
import time

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Compute texton histograms')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("first_sec", type=int, help="first slice")
parser.add_argument("last_sec", type=int, help="last slice")
parser.add_argument("interval", type=int, help="slice interval to take feature samples")
parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
args = parser.parse_args()


from joblib import Parallel, delayed

sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], 'notebooks'))
from utilities2015 import *

os.environ['GORDON_DATA_DIR'] = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_processed'
os.environ['GORDON_REPO_DIR'] = '/oasis/projects/nsf/csd395/yuncong/Brain'
os.environ['GORDON_RESULT_DIR'] = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_results'

dm = DataManager(data_dir=os.environ['GORDON_DATA_DIR'], 
  repo_dir=os.environ['GORDON_REPO_DIR'], 
  result_dir=os.environ['GORDON_RESULT_DIR'], labeling_dir=os.environ['GORDON_LABELING_DIR'],
  stack=args.stack_name)

#===================================================

if dm.check_pipeline_result('textons', 'npy'):
	print "textons.npy already exists, skip"

else:
	# print 'reading centroids ...',
	# t = time.time()

	# s = check_output("ls %s" % os.path.join(os.environ['GORDON_RESULT_DIR'], args.stack_name), shell=True)

	# slide_indices = [int(f) for f in s.split('\n') if len(f) > 0]
	# section_num = len(slide_indices)
	
	# all_centroids = []
	# for i in range(args.first_sec, args.last_sec+1, args.interval):	
	# 	dm.set_slice(i)
	# 	centroids = dm.load_pipeline_result('centroids', 'npy')
	# 	all_centroids.append(centroids)
	# centroids = np.vstack(all_centroids)

 #   	print 'done in', time.time() - t, 'seconds'


	print 'reading features ...',
	t = time.time()

	import random
	# from subprocess import check_output

	features_rotated_list = []

	# s = check_output("ls %s" % os.path.join(os.environ['GORDON_RESULT_DIR'], args.stack_name), shell=True)

	# slide_indices = [int(f) for f in s.split('\n') if len(f) > 0]
	# section_num = len(slide_indices)
	
	for i in range(args.first_sec, args.last_sec+1, args.interval):

		dm.set_slice(i)
		dm._load_image()

		features_rotated_one_image = dm.load_pipeline_result('featuresRotated', 'npy')
		features_rotated_list.append(features_rotated_one_image[np.random.randint(features_rotated_one_image.shape[0], size=1000000), :])
	
	features_rotated = np.vstack(features_rotated_list)

	del features_rotated_list

	print 'done in', time.time() - t, 'seconds'




	print 'clustering features ...',
	t = time.time()

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
	print 'done in', time.time() - t, 'seconds'

	print 'merging centroids ...',
	t = time.time()

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

	print 'done in', time.time() - t, 'seconds'

	dm.save_pipeline_result(reduced_centroids, 'textons', 'npy')
