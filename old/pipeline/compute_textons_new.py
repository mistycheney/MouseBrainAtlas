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
parser.add_argument("-i", "--interval", type=int, help="slice interval to take feature samples (default: %(default)s)", default=1)
parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
# parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
args = parser.parse_args()


from joblib import Parallel, delayed

import numpy as np

#===================================================

print 'reading centroids ...',
t = time.time()

centroid_list = []
for i in range(args.first_sec, args.last_sec+1, args.interval):	
	centroids = np.load(os.environ['GORDON_RESULT_DIR']+'/MD594_centroids/'+ args.stack_name+'_'+'%04d'%i+'_lossless_gabor-'+args.gabor_params_id+'_centroids.npy')
	centroid_list.append(centroids)
centroids = np.vstack(centroid_list)

print 'done in', time.time() - t, 'seconds'

print 'merging centroids ...',
t = time.time()

print centroids.shape

from scipy.cluster.hierarchy import fclusterdata
# cluster_assignments = fclusterdata(centroids, 1.15, method="complete", criterion="inconsistent")
cluster_assignments = fclusterdata(centroids, .4, method="complete", criterion="distance")

reduced_centroids = np.array([centroids[cluster_assignments == i].mean(axis=0) for i in set(cluster_assignments)])
n_reduced_texton = len(reduced_centroids)
print n_reduced_texton, 'reduced textons'

# from sklearn.cluster import MiniBatchKMeans
# kmeans = MiniBatchKMeans(n_clusters=n_reduced_texton, batch_size=1000, init=reduced_centroids)
# # kmeans.fit(features_rotated_pca)
# kmeans.fit(features_rotated)
# final_centroids = kmeans.cluster_centers_
# # labels = kmeans.labels_

print 'done in', time.time() - t, 'seconds'

np.save(os.environ['GORDON_RESULT_DIR']+'/'+args.stack_name+'_lossless_gabor-'+args.gabor_params_id+'-vq-'+args.vq_params_id+'_textons.npy', reduced_centroids)