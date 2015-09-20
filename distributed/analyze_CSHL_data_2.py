#! /usr/bin/env python

import numpy as np

from joblib import Parallel, delayed

from scipy.cluster.hierarchy import fclusterdata

import matplotlib.pyplot as plt

from utilities2015 import *

import os, sys
import time

stack = sys.argv[1]
secind = 100

os.environ['GORDON_DATA_DIR'] = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_processed'
os.environ['GORDON_REPO_DIR'] = '/oasis/projects/nsf/csd395/yuncong/Brain'
os.environ['GORDON_RESULT_DIR'] = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_results'

dm = DataManager(data_dir=os.environ['GORDON_DATA_DIR'],
                 repo_dir=os.environ['GORDON_REPO_DIR'],
                 result_dir=os.environ['GORDON_RESULT_DIR'],
                 stack=stack, section=secind)

t = time.time()
print 'loading features data ...',

# features_rotated = dm.load_pipeline_result('featuresRotated', 'npy')
features_rotated = dm.load_pipeline_result('featuresRotated', 'hdf')

print 'done in', time.time() - t, 'seconds'

del features_rotated

t = time.time()
print 'quantize feature vectors ...',

n_texton = 100
    
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=n_texton, batch_size=1000)
kmeans.fit(features_rotated[::10])
centroids = kmeans.cluster_centers_

cluster_assignments = fclusterdata(centroids, 1.15, method="complete", criterion="inconsistent")
# cluster_assignments = fclusterdata(centroids, 80., method="complete", criterion="distance"
# cluster_assignments = fclusterdata(centroids, 1.1, method="complete", criterion="inconsistent")

centroids = np.array([centroids[cluster_assignments == i].mean(axis=0) for i in set(cluster_assignments)])

n_texton = len(centroids)
print n_texton, 'reduced textons'

print 'done in', time.time() - t, 'seconds'

del kmeans

dm.save_pipeline_result(centroids, 'textons', 'npy')

