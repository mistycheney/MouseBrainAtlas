# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
from sklearn.mixture import GMM
from scipy.io import loadmat
import matplotlib.pyplot as plt

# <codecell>

g = GMM(n_components=200, covariance_type='full', 
        random_state=None, thresh=0.01, 
        min_covar=0.001, n_iter=100, 
        n_init=1, params='wmc', init_params='wmc')

# <codecell>

data = loadmat('/home/yuncong/data_train.mat')
data_train_T = data['data_train'].T

# <codecell>

from sklearn.cluster import KMeans, MiniBatchKMeans

kmeans = MiniBatchKMeans(n_clusters=200, init='random', compute_labels=True,
       n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, batch_size=10000)

# kmeans = KMeans(n_clusters=10, init='k-means++', n_init=10, 
#        max_iter=300, tol=0.0001, precompute_distances=False, 
#        verbose=0, random_state=None, copy_x=True, n_jobs=16)

# <codecell>

kmeans.fit(data_train_T)

# <markdowncell>

# Show patches that belong to cluster 3

# <codecell>

fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(20,20))
for i, ax in zip(where(kmeans.labels_==111)[0][:20], axes.flat):
    patch = data_train_T[i].reshape((8,8))
    ax.imshow(patch, cmap=plt.cm.Greys_r)

# <codecell>

valid = np.unique(kmeans.labels_)
counts = np.bincount(kmeans.labels_)
valid2 = [valid[i] for i in range(len(valid)) if counts[i] > 1000]

# plt.bar(np.arange(len(counts)), counts);
# plt.show()
# plt.hist(counts, bins=np.arange(0, 10000, 100));

# <codecell>

fig, axes = plt.subplots(nrows=len(valid)/10, ncols=10, figsize=(20,20))
for i, (im, ax) in enumerate(zip(kmeans.cluster_centers_[valid], axes.flat)):
    patch = im.reshape((8,8))
    ax.imshow(patch, cmap=plt.cm.Greys_r)
    ax.set_title(str(i))
    ax.axis('off')

