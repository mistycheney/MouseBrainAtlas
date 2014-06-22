# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
CACHE_DIR = '/home/yuncong/my_csd181_scratch/'
img_name = 'PMD1305_244_reduce0_region0'
features = np.load(CACHE_DIR + img_name + '_features.npy')
features = features[200:-200, 200:-200, :]
X = features.reshape(-1, 72)

from scipy.spatial.distance import euclidean, cdist

# w = np.random.randint(X.shape[0], size=(20,))

n_sample = 100

q = np.random.randint(X.shape[0], size=(n_sample,))
data = X[q,:]

centroids = data[:20]

for iteration in range(5):
    ci = np.zeros((n_sample,), dtype=np.int)
    ri = np.zeros((n_sample,), dtype=np.int)
    for data_i, x in enumerate(data):
        D = np.zeros((20,18))
        for k, c in enumerate(centroids):
            b = np.vstack([np.concatenate([np.roll(c[0:18], i),
                           np.roll(c[18:36], i),
                           np.roll(c[36:54], i),
                           np.roll(c[54:72], i)]) for i in range(18)])
            D[k,:] = cdist(x[np.newaxis,:], b, 'sqeuclidean')
        ci[data_i], ri[data_i] = np.unravel_index(D.argmin(), D.shape)

    centroids_new = np.zeros((20,72))
    counts = np.zeros((20,), dtype=np.int)
    for data_i in range(n_sample):
        d = data[data_i]
        a = np.concatenate([np.roll(d[0:18], ri[data_i]),
                           np.roll(d[18:36], ri[data_i]),
                           np.roll(d[36:54], ri[data_i]),
                           np.roll(d[54:72], ri[data_i])])
        centroids_new[ci[data_i]] += a
        counts[ci[data_i]] += 1
    centroids_new /= counts[:, np.newaxis]
    
    
    
    
#     print np.sqrt(np.sum((centroids - centroids_new)**2, axis=1)).mean()
    
    centroids = centroids_new

    
# %%time
n_data = X.shape[0]

b = [None for i in range(20)]
for k, c in enumerate(centroids):
    b[k] = np.vstack([np.concatenate([np.roll(c[0:18], i),
                   np.roll(c[18:36], i),
                   np.roll(c[36:54], i),
                   np.roll(c[54:72], i)]) for i in range(18)])
    
bb = np.reshape(b, (18*20, 72))

# %%time
# ci = -1*np.ones((n_data,), dtype=np.int)
# ri = np.zeros((n_data,), dtype=np.int)
from multiprocessing import Pool

def compute_dist_per_proc(i):
    D = cdist(X[1000*i:1000*(i+1)], bb, 'sqeuclidean')
    ci, ri = np.unravel_index(D.argmin(axis=1), (20,18))
    return ci, ri

pool = Pool(4)
res = pool.map(compute_dist_per_proc, range(4))

