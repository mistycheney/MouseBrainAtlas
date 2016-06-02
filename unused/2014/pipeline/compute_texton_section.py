#! /usr/bin/env python

import os
import argparse
import sys
import time

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Compute textons')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("slice_ind", type=int, help="slice index")
parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
args = parser.parse_args()

from joblib import Parallel, delayed

sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], 'utilities'))
from utilities2015 import *

dm = DataManager(stack=args.stack_name,
                 section=args.slice_ind,
                 gabor_params_id=args.gabor_params_id, 
                 # segm_params_id=args.segm_params_id, 
                 vq_params_id=args.vq_params_id,
                 result_dir='/scratch/yuncong/CSHL_data_results')

#===================================================

# if dm.check_pipeline_result('textons'):
# 	print "textons.npy already exists, skip"

# else:

print 'reading features ...',
t = time.time()

import random
# from subprocess import check_output

features_rotated_list = []

# dm = DataManager(gabor_params_id=args.gabor_params_id, 
#              stack=args.stack_name, 
#              section=args.slice_ind,
#              result_dir='/scratch/yuncong/CSHL_data_results')

block_size = 4000

for col, xmin in enumerate(range(dm.xmin, dm.xmax, block_size)):
    for row, ymin in enumerate(range(dm.ymin, dm.ymax, block_size)):

		xmax = xmin + block_size - 1
		ymax = ymin + block_size - 1

		t = time.time()
		sys.stderr.write('load featuresRotated ...')

		if not dm.check_pipeline_result('featuresMaskedRotatedRow%dCol%d'%(row, col)):
			continue

		features_rotated_one_image = dm.load_pipeline_result('featuresMaskedRotatedRow%dCol%d'%(row, col))

		features_rotated_list.append(features_rotated_one_image[np.random.randint(features_rotated_one_image.shape[0], size=10000), :])

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

dm.save_pipeline_result(centroids, 'centroids')

# print 'merging centroids ...',
# t = time.time()

# from scipy.cluster.hierarchy import fclusterdata
# cluster_assignments = fclusterdata(centroids, 1.15, method="complete", criterion="inconsistent")
# # cluster_assignments = fclusterdata(centroids, 80., method="complete", criterion="distance")

# reduced_centroids = np.array([centroids[cluster_assignments == i].mean(axis=0) for i in set(cluster_assignments)])
# n_reduced_texton = len(reduced_centroids)
# print n_reduced_texton, 'reduced textons'

# kmeans = MiniBatchKMeans(n_clusters=n_reduced_texton, batch_size=1000, init=reduced_centroids)
# # kmeans.fit(features_rotated_pca)
# kmeans.fit(features_rotated)
# final_centroids = kmeans.cluster_centers_
# labels = kmeans.labels_

# print 'done in', time.time() - t, 'seconds'

# dm.save_pipeline_result(reduced_centroids, 'textons')
