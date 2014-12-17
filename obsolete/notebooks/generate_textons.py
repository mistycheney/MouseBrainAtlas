# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

# <codecell>

from utilities import *

if 'SSH_CONNECTION' in os.environ:
    DATA_DIR = '/home/yuncong/DavidData'
    REPO_DIR = '/home/yuncong/Brain'
else:
    DATA_DIR = '/home/yuncong/BrainLocal/DavidData_v3'
    REPO_DIR = '/home/yuncong/Brain'

dm = DataManager(DATA_DIR, REPO_DIR)

class args:
    stack_name = 'RS141'
    resolution = 'x5'
    slice_indices = range(0, 23, 5)
#     slice_indices = [1]

# <codecell>

dm.set_stack(args.stack_name, args.resolution)
dm.set_gabor_params(gabor_params_id='blueNisslWide')
dm.set_segmentation_params(segm_params_id='blueNissl')
dm.set_vq_params(vq_params_id='blueNissl')

# <codecell>

import random

features_fullstack = []

sample_per_slice = 1000000/len(args.slice_indices)

for slice_ind in args.slice_indices:
    print slice_ind
    dm.set_slice(slice_ind)
    
    cropped_features = dm.load_pipeline_result('cropFeatures', 'npy')    
    cropped_mask = dm.load_pipeline_result('cropMask', 'npy')
    
    n_pixels = cropped_features.shape[0]*cropped_features.shape[1]
    features_fullstack.append(random.sample(cropped_features[cropped_mask], min(sample_per_slice, n_pixels)))
    
features_fullstack_all = np.vstack(features_fullstack)

n_feature = features_fullstack_all.shape[-1]
print n_feature

# <codecell>

n_components = 5

from sklearn.decomposition import PCA
pca = PCA(n_components=n_components)
pca.fit(features_fullstack_all)
print(pca.explained_variance_ratio_)

features_pca = pca.transform(features_fullstack_all)

# <codecell>

import random
import itertools
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist

print '=== compute rotation-invariant texton map using K-Means ==='

# n_texton = int(dm.vq_params['n_texton'])


theta_interval = dm.gabor_params['theta_interval']
n_angle = int(180/theta_interval)
freq_step = dm.gabor_params['freq_step']
freq_max = 1./dm.gabor_params['min_wavelen']
freq_min = 1./dm.gabor_params['max_wavelen']
n_freq = int(np.log(freq_max/freq_min)/np.log(freq_step)) + 1

def compute_dist_per_proc(X_partial, c_all_rot):
    D = cdist(X_partial, c_all_rot, 'sqeuclidean')
    ci, ri = np.unravel_index(D.argmin(axis=1), (n_components, n_angle))
    return np.c_[ci, ri]

n_data = features_pca.shape[0]
n_splits = 1000
n_sample = min(int(dm.vq_params['n_sample']), n_data)

centroids = np.array(random.sample(features_pca, n_components))

n_iter = int(dm.vq_params['n_iter'])

for iteration in range(n_iter):

    data = random.sample(features_pca, n_sample)

    print 'iteration', iteration
    centroid_all_rotations = np.vstack([np.concatenate(np.roll(np.split(c, n_freq), i)) 
                            for c,i in itertools.product(centroids, range(n_angle))])

    r = Parallel(n_jobs=16)(delayed(compute_dist_per_proc)(x,c) 
                    for x, c in zip(np.array_split(data, n_splits, axis=0), 
                                    itertools.repeat(centroid_all_rotations, n_splits)))
    res = np.vstack(r)

    labels = res[:,0]
    rotations = res[:,1]

    centroids_new = np.zeros((n_components, n_feature))
    for d, l, r in itertools.izip(data, labels, rotations):
        rot = np.concatenate(np.roll(np.split(d, n_freq), i))
        centroids_new[l] += rot

    counts = np.bincount(labels, minlength=n_components)
    centroids_new /= counts[:, np.newaxis] # denominator might be zero
    centroids_new[counts==0] = centroids[counts==0]
    print np.sqrt(np.sum((centroids - centroids_new)**2, axis=1)).mean()

    centroids = centroids_new

    
print centroids.shape
# dm.save_pipeline_result(centroids, 'textons', 'npy')


print 'kmeans completes'

# <codecell>


