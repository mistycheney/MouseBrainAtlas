# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
CACHE_DIR = '/home/yuncong/my_csd181_scratch/'
img_name = 'PMD1305_244_reduce0_region0'
features = np.load(CACHE_DIR + img_name + '_features.npy')
features = features[200:-200, 200:-200, :]
# features = features[2500:2700, 2500:2700, :]
X = features.reshape(-1, 72)

# <codecell>

# %%time
from sklearn.cluster import MiniBatchKMeans 

num_textons = 20
kmeans_model = MiniBatchKMeans(num_textons, batch_size=100)
kmeans_model.fit(b)

# <codecell>

def kmean(X, num_clusters):
    num_data = X.shape[0]
    centroids = X[np.random.choice(range(num_data), size=num_clusters)]
    for iteration in range(10):
        D = cdist(X, centroids)
        print D.shape
        labels = D.argmin(axis=1)
        centroids = np.array([X[labels==i].mean(axis=0) for i in range(num_clusters)])
    return centroids

# <codecell>

# for i in range(72):
plt.imshow(features[...,-1], cmap=plt.cm.coolwarm_r);
plt.show()

# <codecell>

%%time

from scipy.spatial.distance import euclidean, cdist
import random
from multiprocessing import Pool
import itertools

n_data = X.shape[0]
n_freq = 4
n_angle = 18
n_texton = 20
n_feature = 72
n_splits = 1000

def compute_dist_per_proc((X_partial, c_all_rot)):
    D = cdist(X_partial, c_all_rot, 'sqeuclidean')
    ci, ri = np.unravel_index(D.argmin(axis=1), (n_texton, n_angle))
    return np.column_stack((ci, ri))

pool = Pool(16)

n_sample = 10000
data = random.sample(X, n_sample)
centroids = data[:n_texton]

n_iter = 10
for iteration in range(n_iter):

    centroid_all_rotations = np.vstack([np.concatenate(np.roll(np.split(c, n_freq), i)) 
                                for c,i in itertools.product(centroids, range(n_angle))])
    
    print centroid_all_rotations.shape
    
    res = np.vstack(pool.map(compute_dist_per_proc, 
                             zip(np.array_split(data, n_splits, axis=0), 
                                 itertools.repeat(centroid_all_rotations, n_splits))))
    labels = res[:,0]
    rotations = res[:,1]
    
    centroids_new = np.zeros((n_texton, n_feature))
    for d, l, r in itertools.izip(data, labels, rotations):
        rot = np.concatenate(np.roll(np.split(d, n_freq), i))
        centroids_new[l] += rot
        
    counts = np.bincount(labels)
    centroids_new /= counts[:, np.newaxis]
    print np.sqrt(np.sum((centroids - centroids_new)**2, axis=1)).mean()
    
    centroids = centroids_new

# <codecell>

%%time
centroid_all_rotations = np.vstack([np.concatenate(np.roll(np.split(c, n_freq), i)) 
                            for c,i in itertools.product(centroids, range(n_angle))])

res = np.vstack(pool.map(compute_dist_per_proc, 
                         zip(np.array_split(X, n_splits, axis=0), itertools.repeat(centroid_all_rotations, n_splits))))
labels = res[:,0]
rotations = res[:,1]

# <codecell>

pool.close()

# <codecell>

%%time
textonmap = labels.reshape(features.shape[:2])

# <codecell>

from skimage.color import gray2rgb, label2rgb
textonmap_rgb = label2rgb(textonmap, image=None, colors=None, alpha=0.3, image_alpha=1)

# cv2.imwrite(img_name+'_textonmap_with_boundary_text.jpg', 
#             img_as_ubyte(.4*textonmap_rgb + .6*img_superpixelized_text))
# Image(img_name+'_textonmap_with_boundary_text.jpg')

# <codecell>

import cv2
from skimage.util import img_as_ubyte
from IPython.display import FileLink, Image, FileLinks

cv2.imwrite(img_name + '_textonmap.png', img_as_ubyte(textonmap_rgb))
FileLink(CACHE_DIR + img_name + '_textonmap.png')

# <codecell>

sim_mat = 

# <codecell>

plt.scatter(data[:,0], data[:,1])
plt.scatter(centroids[:,0], centroids[:,1], c='r')

# <codecell>

# %%time
# X = np.random.random((30,2))
# Xnormalized = (X - X.mean(axis=1)[:,np.newaxis])/np.sqrt((X.var(axis=1)[:,np.newaxis]))

centroids = kmean(X[:100,:], num_clusters=20)
# plt.scatter(X[:,0], X[:,1])
# plt.scatter(centroids[:,0], centroids[:,1], c='r')
# print centroids

# <codecell>

a = np.random.random((30,10))
plt.scatter(a[:,0], a[:,1], c='r');

b = (a - a.mean(axis=1)[:,np.newaxis])/np.sqrt((a.var(axis=1)[:,np.newaxis]))
plt.scatter(b[:,0], b[:,1]);
plt.show()

# <codecell>

q = np.random.randint(X.shape[0], size=(10000,))
plt.scatter(X[q,60], X[q,50], s=.2);

