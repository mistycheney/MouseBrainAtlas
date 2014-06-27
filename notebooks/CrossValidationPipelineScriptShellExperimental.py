# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import cv2
import matplotlib.pyplot as plt

import random, itertools, sys, os
from multiprocessing import Pool
import json

from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops
from skimage.util import img_as_ubyte
from skimage.color import hsv2rgb, label2rgb, gray2rgb
from skimage.morphology import disk
from skimage.filter.rank import gradient
from skimage.filter import gabor_kernel
from skimage.transform import rescale, resize

from scipy.ndimage import gaussian_filter, measurements
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, squareform, euclidean, cdist
from scipy.signal import fftconvolve

from IPython.display import FileLink, Image, FileLinks

from utilities import *
import manager_utilities

from joblib import Parallel, delayed

import glob, re, os, sys, subprocess, argparse

# <codecell>

img_dir = '../data/PMD1305_reduce2_region0'
img_name_full = 'PMD1305_reduce2_region0_0244.tif'

# img_dir = '../data/PMD1305_reduce2_region1'
# img_name_full = 'PMD1305_reduce2_region1_0160.tif'

img_path = os.path.join(img_dir, img_name_full)
img_name, ext = os.path.splitext(img_name_full)

param_id = 10

class args:
    param_file = '../params/param%d.json'%param_id
    img_file = img_path
    output_dir = '../output'

# <codecell>

def load_array(suffix):
    return manager_utilities.load_array(suffix, img_name, 
                                 params['param_id'], args.output_dir)

def save_array(arr, suffix):
    manager_utilities.save_array(arr, suffix, img_name, 
                                 params['param_id'], args.output_dir)
        
def save_img(img, suffix):
    manager_utilities.save_img(img, suffix, img_name, params['param_id'], 
                               args.output_dir, overwrite=True)

def get_img_filename(suffix, ext='tif'):
    return manager_utilities.get_img_filename(suffix, img_name, params['param_id'], args.output_dir, ext=ext)

# <codecell>

params = json.load(open(args.param_file))

p, ext = os.path.splitext(args.img_file)
img_dir, img_name = os.path.split(p)
img = cv2.imread(args.img_file, 0)
im_height, im_width = img.shape[:2]

print 'read %s' % args.img_file

result_name = img_name + '_param' + str(params['param_id'])
result_dir = os.path.join(args.output_dir, result_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# <codecell>

%%time
print '=== finding foreground mask ==='
# mask = foreground_mask(rescale(img, .5**3), min_size=100)
# mask = resize(mask, img.shape) > .5
mask = foreground_mask(img, min_size=2000)
mask = mask > .5
plt.imshow(mask, cmap=plt.cm.Greys_r);

# <codecell>

theta_interval = params['theta_interval'] #10
n_angle = 180/theta_interval
# n_freq = params['n_freq']
freq_step = params['freq_step']
freq_max = 1./params['min_wavelen'] #1./5.
freq_min = 1./params['max_wavelen']
n_freq = int(np.log(freq_max/freq_min)/np.log(freq_step)) + 1
frequencies = freq_max/freq_step**np.arange(n_freq)

kernels = [gabor_kernel(f, theta=t, bandwidth=2.) for f in frequencies 
          for t in np.arange(0, np.pi, np.deg2rad(theta_interval))]
kernels = map(np.real, kernels)
n_kernel = len(kernels)

print '=== filter using Gabor filters ==='
print 'num. of kernels: %d' % (n_kernel)
print 'frequencies:', frequencies
print 'wavelength (pixels):', 1/frequencies

max_kern_size = np.max([kern.shape[0] for kern in kernels])
print 'max kernel matrix size:', max_kern_size

# <codecell>

%%time

try:
    raise IOError
    features = load_array('features')
except IOError:
    def convolve_per_proc(i):
        return fftconvolve(img, kernels[i], 'same').astype(np.half)

    filtered = Parallel(n_jobs=16)(delayed(convolve_per_proc)(i) 
                            for i in range(n_kernel))

    features = np.empty((im_height, im_width, n_kernel), dtype=np.half)
    for i in range(n_kernel):
        features[...,i] = filtered[i]

    del filtered
    
    save_array(features, 'features')

n_feature = features.shape[-1]

# <codecell>

print 'crop border where filters show border effects'
features = features[max_kern_size/2:-max_kern_size/2, max_kern_size/2:-max_kern_size/2, :]
img = img[max_kern_size/2:-max_kern_size/2, max_kern_size/2:-max_kern_size/2]
mask = mask[max_kern_size/2:-max_kern_size/2, max_kern_size/2:-max_kern_size/2]

# features = features[:-500, :, :]
# img = img[:-500, :]
# mask = mask[:-500, :]

im_height, im_width = img.shape[:2]

# <codecell>

%%time

print '=== compute rotation-invariant texton map using K-Means ==='

n_texton = params['n_texton']

try: 
    raise IOError
    textonmap = load_array('textonmap')
except IOError:
    
    X = features.reshape(-1, n_feature)
    n_data = X.shape[0]
    n_splits = 1000
    n_sample = 10000
    centroids = np.array(random.sample(X, n_texton))
    
    n_iter = 5

    def compute_dist_per_proc(X_partial, c_all_rot):
        D = cdist(X_partial, c_all_rot, 'sqeuclidean')
        ci, ri = np.unravel_index(D.argmin(axis=1), (n_texton, n_angle))
        return np.c_[ci, ri]
    
    for iteration in range(n_iter):
        
        data = np.array(random.sample(X, n_sample))
        
        print 'iteration', iteration
        centroid_all_rotations = np.vstack([np.concatenate(np.roll(np.split(c, n_freq), i)) 
                                for c,i in itertools.product(centroids, range(n_angle))])

        r = Parallel(n_jobs=16)(delayed(compute_dist_per_proc)(x,c) 
                        for x, c in zip(np.array_split(data, n_splits, axis=0), 
                                        itertools.repeat(centroid_all_rotations, n_splits)))
        res = np.vstack(r)        
        labels = res[:,0]
        rotations = res[:,1]

        centroids_new = np.zeros((n_texton, n_feature))
        for d, l, r in itertools.izip(data, labels, rotations):
            rot = np.concatenate(np.roll(np.split(d, n_freq), i))
            centroids_new[l] += rot

        counts = np.bincount(labels, minlength=n_texton)
#         if np.count_nonzero(counts) == len(counts):
        centroids_new /= counts[:, np.newaxis]
        centroids_new[counts==0] = centroids[counts==0]
        print 'average centroid movement', np.sqrt(np.sum((centroids - centroids_new)**2, axis=1)).mean()
        print 'average instance-centroid distance', np.sqrt(np.sum([(np.roll(data[i], rotations[i]) - centroids_new[labels[i]])**2 for i in range(n_sample)], axis=1)).mean()

        centroids = centroids_new

    print 'kmeans completes'
    centroid_all_rotations = np.vstack([np.concatenate(np.roll(np.split(c, n_freq), i)) 
                                for c,i in itertools.product(centroids, range(n_angle))])

    r = Parallel(n_jobs=16)(delayed(compute_dist_per_proc)(x,c) 
                            for x, c in zip(np.array_split(X, n_splits, axis=0), 
                                            itertools.repeat(centroid_all_rotations, n_splits)))
    res = np.vstack(r)
    
    labels = res[:,0]
    rotations = res[:,1]
#     print 'average instance-centroid distance', np.sqrt(np.sum([(np.roll(X[i], rotations[i]) - centroids_new[labels[i]])**2 for i in range(X.shape[0])], axis=1)).mean()
    
    textonmap = labels.reshape(features.shape[:2])
    textonmap[~mask] = -1
    
    save_array(textonmap, 'textonmap')
    
    textonmap_rgb = label2rgb(textonmap, image=None, colors=None, alpha=0.3, image_alpha=1)
    save_img(textonmap_rgb, 'textonmap')

# <codecell>

%%time

print '=== over-segment the image into superpixels based on color information ==='

img_rgb = gray2rgb(img)

try:
    raise IOError
    segmentation = load_array('segmentation')
    
except IOError:
    segmentation = slic(img_rgb, n_segments=params['n_superpixels'], max_iter=10, 
                        compactness=params['slic_compactness'], 
                        sigma=params['slic_sigma'], enforce_connectivity=True)
    print 'segmentation computed'
    
    save_array(segmentation, 'segmentation')
    
sp_props = regionprops(segmentation+1, intensity_image=img, cache=True)

def foo2(i):
    return sp_props[i].centroid, sp_props[i].area, sp_props[i].mean_intensity

r = Parallel(n_jobs=16)(delayed(foo2)(i) for i in range(len(sp_props)))

sp_centroids = np.array([s[0] for s in r])
sp_areas = np.array([s[1] for s in r])
sp_mean_intensity = np.array([s[2] for s in r])

    # sp_areas = np.array([s.area for s in sp_props])
    # sp_mean_intensity = np.array([s.mean_intensity for s in sp_props])

    # sp_centroids = np.array([s.centroid for s in sp_props])
    # sp_wcentroids = np.array([s.weighted_centroid for s in sp_props])
    # sp_centroid_dist = pdist(sp_centroids)
    # sp_centroid_dist_matrix = squareform(sp_centroid_dist)

n_superpixels = len(np.unique(segmentation))

img_superpixelized = mark_boundaries(img_rgb, segmentation)
sptext = img_as_ubyte(img_superpixelized)
for s in range(n_superpixels):
    sptext = cv2.putText(sptext, str(s), 
                      tuple(np.floor(sp_centroids[s][::-1]).astype(np.int)), 
                      cv2.FONT_HERSHEY_COMPLEX_SMALL,
                      .5, ((255,0,255)), 1)
save_img(sptext, 'segmentation')

# <codecell>

%%time
# superpixels_fg_count = [np.count_nonzero(mask[segmentation==i]) for i in range(n_superpixels)]

def foo(i):
    return np.count_nonzero(mask[segmentation==i])

r = Parallel(n_jobs=16)(delayed(foo)(i) for i in range(n_superpixels))
superpixels_fg_count = np.array(r)
bg_superpixels = np.nonzero((superpixels_fg_count/sp_areas) < 0.3)[0]
print '%d background superpixels'%len(bg_superpixels)

# pool = Pool(16)
# superpixels_fg_count = np.array(pool.map(foo, range(n_superpixels)))
# pool.close()
# pool.join()
# del pool

# <codecell>

%%time

print '=== compute texton and directionality histogram of each superpixel ==='

# sample_interval = 1
# gridy, gridx = np.mgrid[:img.shape[0]:sample_interval, :img.shape[1]:sample_interval]

# all_seg = segmentation[gridy.ravel(), gridx.ravel()]

try:
    raise IOError
    sp_texton_hist_normalized = load_array('sp_texton_hist_normalized')
except IOError:
#     all_texton = textonmap[gridy.ravel(), gridx.ravel()]

    def bar(i):
        return np.bincount(textonmap[(segmentation == i)&(textonmap != -1)], minlength=n_texton)

    r = Parallel(n_jobs=16)(delayed(bar)(i) for i in range(n_superpixels))
    sp_texton_hist = np.array(r)
    # sp_texton_hist = np.array([np.bincount(textonmap[(segmentation == s)&(textonmap != -1)], minlength=n_texton) 
    #                  for s in range(n_superpixels)])
    sp_texton_hist_normalized = sp_texton_hist.astype(np.float) / sp_texton_hist.sum(axis=1)[:, np.newaxis]
    save_array(sp_texton_hist_normalized, 'sp_texton_hist_normalized')

# <codecell>

D = pdist(sp_texton_hist_normalized)
D = squareform(D)

# <codecell>

%%time

try:
    raise IOError
    sp_dir_hist_normalized = load_array('sp_dir_hist_normalized')
except IOError:
    f = np.reshape(features, (features.shape[0], features.shape[1], n_freq, n_angle))
    dir_energy = np.sum(abs(f), axis=2)

    def bar2(i):
        segment_dir_energies = dir_energy[segmentation == i].astype(np.float_).mean(axis=0)
        return segment_dir_energies    

    r = Parallel(n_jobs=16)(delayed(bar2)(i) for i in range(n_superpixels))
    
    sp_dir_hist = np.vstack(r)
    sp_dir_hist_normalized = sp_dir_hist/sp_dir_hist.sum(axis=1)[:,np.newaxis]
    save_array(sp_dir_hist_normalized, 'sp_dir_hist_normalized')

# pool = Pool(16)
# r = pool.map(bar2, range(n_superpixels))
# try:
#     sp_dir_hist_normalized = load_array('sp_dir_hist_normalized')
# except IOError:
# sp_dir_hist_normalized = np.empty((n_superpixels, n_angle))
# for i in range(n_superpixels):
#     segment_dir_energies = dir_energy[segmentation == i].astype(np.float_).sum(axis=0)
#     sp_dir_hist_normalized[i,:] = segment_dir_energies/segment_dir_energies.sum()    
# save_array(sp_dir_hist_normalized, 'sp_dir_hist_normalized')

# <codecell>

%%time

def chi2(u,v):
    return np.sum(np.where(u+v!=0, (u-v)**2/(u+v), 0))

print '=== compute significance of each superpixel ==='

overall_texton_hist = np.bincount(textonmap[mask].flat)
overall_texton_hist_normalized = overall_texton_hist.astype(np.float) / overall_texton_hist.sum()

# individual_texton_saliency_score = np.zeros((n_superpixels, ))
# for i, sp_hist in enumerate(sp_texton_hist_normalized):
#     individual_texton_saliency_score[i] = chi2(sp_hist, overall_texton_hist_normalized)

# individual_texton_saliency_score = cdist(sp_texton_hist_normalized, overall_texton_hist_normalized[np.newaxis,:], chi2)
# individual_texton_saliency_score[bg_superpixels] = 0
individual_texton_saliency_score = np.array([chi2(sp_hist, overall_texton_hist_normalized) 
                                             if sp_hist not in bg_superpixels else 0 
                                             for sp_hist in sp_texton_hist_normalized])

# texton_saliency_score = individual_texton_saliency_score

texton_saliency_score = np.zeros((n_superpixels,))
for i, sp_hist in enumerate(sp_texton_hist_normalized):
    if i not in bg_superpixels:
        texton_saliency_score[i] = individual_texton_saliency_score[i]
        
texton_saliency_map = texton_saliency_score[segmentation]

# save_img(texton_saliency_map, 'texton_saliencymap')
# Image(get_img_filename('texton_saliencymap', 'png'))

texton_saliency_map = texton_saliency_score[segmentation]
plt.matshow(texton_saliency_map, cmap=cm.Greys_r)
plt.colorbar()
plt.show()

# <codecell>

x = np.linspace(0,1,100)
plot(x, np.exp(-100*x));

# <codecell>

%%time

# mean_diameter = np.sqrt(sp_areas).mean()
mean_unit = .5*(im_height + im_width)

sp_centroid_dist = pdist(sp_centroids)
sp_centroid_dist_matrix = squareform(sp_centroid_dist)

def bar4(i):
    if i not in bg_superpixels: 
        return np.mean([np.exp(-100*sp_centroid_dist_matrix[i,j]/mean_unit) * D[i,j] 
                            for j in range(n_superpixels) if j != i and j not in bg_superpixels])
    else:
        return 0

# texton_saliency_score = np.zeros((n_superpixels,))
r = Parallel(n_jobs=16)(delayed(bar4)(i) for i in range(n_superpixels))
texton_saliency_score = np.array(r)


# for i in range(n_superpixels):
#     if i not in bg_superpixels: 
#         texton_saliency_score[i] = np.sum([np.exp(-.1*sp_centroid_dist_matrix[i,j]) * D[i,j] 
#                             for j in range(n_superpixels) if j != i and j not in bg_superpixels])

texton_saliency_map = texton_saliency_score[segmentation]
plt.matshow(texton_saliency_map, cmap=cm.Greys_r)
plt.colorbar()
plt.show()

# <codecell>

%%time

from skimage.morphology import disk
from skimage.filter.rank import gradient
from scipy.sparse import coo_matrix

edge_map = gradient(segmentation.astype(np.uint8), disk(3))
neighbors = [set() for i in range(n_superpixels)]

# def bar3(y,x):
#     neighbors[segmentation[y,x]] |= set(segmentation[y-2:y+2,x-2:x+2].ravel())

# r = Parallel(n_jobs=16)(delayed(bar3)(y,x) for y,x in zip(*np.nonzero(edge_map)))
# sp_texton_hist = np.array(r)

for y,x in zip(*np.nonzero(edge_map)):
    neighbors[segmentation[y,x]] |= set(segmentation[y-2:y+2,x-2:x+2].ravel())

for i in range(n_superpixels):
    neighbors[i] -= set([i])
    
# connectivity_matrix = np.zeros((n_segmentation, n_segmentation), dtype=np.bool)
rows = np.hstack([s*np.ones((len(neighbors[s]),), dtype=np.int) for s in range(n_superpixels)])
cols = np.hstack([list(neighbors[s]) for s in range(n_superpixels)])
data = np.ones((cols.size, ), dtype=np.bool)
connectivity_matrix = coo_matrix((data, (rows, cols)), shape=(n_superpixels,n_superpixels))
connectivity_matrix = connectivity_matrix.transpose() * connectivity_matrix

# from skimage.draw import line
# superpixel_connectivity_img = img_superpixelized.copy()
# for i in range(n_superpixels):
#     for neig in neighbors[i]:
#         rr, cc = line(int(sp_centroids[i,0]), int(sp_centroids[i,1]),
#                       int(sp_centroids[neig,0]), int(sp_centroids[neig,1]))
#         superpixel_connectivity_img[rr, cc] = (0,0,1)

# <codecell>

q = sp_dir_hist_normalized.argsort(axis=1)

a = np.zeros((n_superpixels, ))
for i in range(n_superpixels):
    a[i] = sp_dir_hist_normalized[i,q[i,-1]]/sp_dir_hist_normalized[i,q[i,-2]]
# maxdir_rgb = label2rgb(maxdir_map)

w = [i for i in a.argsort() if i not in bg_superpixels]
dir_ratiomap = np.ones_like(segmentation, dtype=np.float)
maxdir_map = -1*np.ones_like(segmentation, dtype=np.int)
for i in w:
    dir_ratiomap[segmentation==i] = a[i]
    maxdir_map[segmentation==i] = q[i,-1]
# maxdir_map[~mask] = 1

# <codecell>

plt.matshow(kernels[9])

# <codecell>

plt.bar(np.arange(n_angle), sp_dir_hist[455]);

# <codecell>

plt.matshow(maxdir_map);
plt.axis('off')
plt.colorbar();
plt.show()

plt.matshow(dir_ratiomap);
plt.axis('off')
plt.colorbar();
plt.show()

# <codecell>

FileLink(get_img_filename('segmentation', 'png')[3:])

# <codecell>

FileLink(get_img_filename('textonmap', 'png')[3:])

# <codecell>

plt.matshow(texton_saliency_map, cmap=plt.cm.Greys_r);
plt.axis('off')
plt.colorbar();

# <codecell>

fg_superpixels = np.array([i for i in range(n_superpixels) if i not in bg_superpixels])
D_fg = D[ix_(fg_superpixels, fg_superpixels)]
# co = connectivity_matrix.todense()
# connectivity_fg = co[ix_(fg_superpixels, fg_superpixels)]
connectivity_fg = connectivity_matrix[ix_(fg_superpixels, fg_superpixels)]

# <markdowncell>

# Spectral Clustering

# <codecell>

from sklearn import cluster

for n_clusters, gamma in itertools.product(range(3,10), np.arange(1., 5., .5)):
    spectral = cluster.SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', assign_labels='kmeans', affinity='precomputed')
    affinity = np.exp(-gamma*D_fg**2)*connectivity_fg
    # labels = spectral.fit_predict(sp_texton_hist_normalized[fg_superpixels])
    labels = spectral.fit_predict(affinity)
    visualize_seg(labels, title='n=%d, gamma=%.2f'%(n_clusters, gamma))

# <markdowncell>

# KMeans; similar result to hierarchical clustering

# <codecell>

n_clusters = 4
kmeans = cluster.MiniBatchKMeans(n_clusters=n_clusters, init='k-means++')
labels = kmeans.fit_predict(sp_texton_hist_normalized[fg_superpixels])
visualize_seg(labels, title='n=%d'%(n_clusters))

# <markdowncell>

# Hierarchical Clustering, must use Euclidean distance between distributions

# <codecell>

c = connectivity_matrix[ix_(fg_superpixels, fg_superpixels)]

# <codecell>

connectivity_matrix[455,440]

# <codecell>

n_clusters = 6
ward = cluster.Ward(n_clusters=n_clusters, connectivity=connectivity_matrix[ix_(fg_superpixels, fg_superpixels)])
labels = ward.fit_predict(sp_texton_hist_normalized[fg_superpixels])
visualize_seg(labels, title='n=%d'%(n_clusters))

# <markdowncell>

# DBSCAN; does not consider connectivity, so not appropriate

# <codecell>

db = cluster.DBSCAN(eps=0.05, min_samples=3)
db.fit_predict(sp_texton_hist_normalized[fg_superpixels])
labels = db.labels_
visualize_seg(labels, title='')

# <markdowncell>

# Affinity Propagation

# <codecell>

gamma = 3
damping = 0.91

affinity = np.exp(-gamma*D_fg**2)*connectivity_fg
affinity_propagation = cluster.AffinityPropagation(damping=damping, affinity='precomputed')
affinity_propagation.fit(affinity)
labels = affinity_propagation.labels_

visualize_seg(labels, title='n=%d'%(n_clusters))

# <codecell>

def visualize_seg(labels, title):
    a = -1*np.ones_like(segmentation)
    for sp_i, c in enumerate(labels):
        a[segmentation==fg_superpixels[sp_i]] = c
    vis = label2rgb(a, img)
    plt.title(title)
    plt.imshow(vis);
    plt.show()

# <codecell>

plt.bar(range(n_texton), sp_texton_hist_normalized[465]);

# <codecell>

sig_sp_sorted = np.argsort(texton_saliency_score)[::-1]

print sig_sp_sorted[:10]
most_sal = sig_sp_sorted[0]
print most_sal
print np.argsort(D[most_sal])[:10]
print np.sort(D[most_sal])[:10]

# <codecell>

chosen_cluster1 = set(curr_cluster.tolist())
print chosen_cluster1

# <codecell>

sp_overall_dist = np.array([chi2(h, overall_texton_hist_normalized) for h in sp_texton_hist_normalized])

# <codecell>

curr_cluster

# <codecell>

plt.bar(np.arange(n_texton), sp_texton_hist_normalized[curr_cluster].mean(axis=0), color='r', width=.2);

# propagate_selection = np.equal.outer(segmentation, curr_cluster).any(axis=2)
# cluster_hist2 = np.bincount(textonmap[mask * propagate_selection].flat, minlength=n_texton)
# cluster_hist2 = cluster_hist2.astype(np.float)/cluster_hist2.sum()
# plt.bar(np.arange(n_texton)+.3, cluster_hist2, color='g', width=.3);

plt.bar(np.arange(n_texton)+.2, sp_texton_hist_normalized[seed], color='b', width=.2);
plt.bar(np.arange(n_texton)+.4, overall_texton_hist_normalized, color='g', width=.2);

plt.show()

# <codecell>

scatter(sp_texton_hist_normalized[curr_cluster,6],sp_texton_hist_normalized[curr_cluster,9],s=10, c='r');
q = np.random.randint(0,n_superpixels,100)
scatter(sp_texton_hist_normalized[q,0],sp_texton_hist_normalized[q,1],s=10,c='b');
plt.show()

# <codecell>

# thresh = .33
# vis = np.exp(-2*D[:,seed])[segmentation]


vis = -(D[:,seed]-sp_overall_dist)[segmentation]
# vis = np.exp(-2*(D[:,seed]-sp_overall_dist))[segmentation]
matched = (D[:, seed] < sp_overall_dist).nonzero()[0]
matched_selection = np.equal.outer(segmentation, matched).any(axis=2)
vis[~matched_selection] = 0

# decoys = np.setdiff1d(matched, curr_cluster)
# decoy_selection = np.equal.outer(segmentation, decoys).any(axis=2)
# vis[~decoy_selection] = 0

plt.matshow(vis, cmap=plt.cm.Greys_r);
plt.title('%.2f'%thresh)
plt.axis('off');
plt.colorbar();
plt.show()


vis = -(D[:,seed]-sp_overall_dist)[segmentation]
cluster_selection = np.equal.outer(segmentation, curr_cluster).any(axis=2)
vis[~cluster_selection] = 0

# decoys = np.setdiff1d(matched, curr_cluster)
# decoy_selection = np.equal.outer(segmentation, decoys).any(axis=2)
# vis[~decoy_selection] = 0

plt.matshow(vis, cmap=plt.cm.Greys_r);
plt.title('%.2f'%thresh)
plt.axis('off');
plt.colorbar();
plt.show()

# <codecell>

plt.matshow(texton_saliency_map, cmap=cm.Greys_r)
plt.colorbar()
plt.show()

# <codecell>

# thresh_choices = np.arange(.2,.5,0.01)
thresh_choices = [.34]
new_sig_scores = []
in_class_vars = []
seed_surround_dists = []
seed_cluster_dists = []
frontier_surround_dists = []

for thresh in thresh_choices:

    clusters = []
#     chosen_superpixels = chosen_cluster1.copy()
    chosen_superpixels = set([])

    for it in range(1):
        if it == 0:
#             seed = 1589
#             seed = 1407
            seed = 829
        else:
            seed = next((i for i in sig_sp_sorted if i not in chosen_superpixels\
                     and i not in bg_superpixels))

        print 'seed', seed, 'saliency score', texton_saliency_score[seed]

        seed_hist = sp_texton_hist_normalized[seed]
        curr_cluster = np.array([seed], dtype=np.int)
        chosen_superpixels.add(seed)
        frontier = [seed]
        while len(frontier) > 0:
            u = frontier.pop(-1)
            for v in neighbors[u]:
#                 and chi2(sp_texton_hist_normalized[v], cluster_hist) < thresh\                
                if v not in curr_cluster and v not in chosen_superpixels\
                and D[v, seed] < thresh\
                and v not in bg_superpixels:
    #                 print 'testing', v, D[v, seed]
                    curr_cluster = np.append(curr_cluster, v)
                    frontier.append(v)
                    chosen_superpixels.add(v)

#         print curr_cluster
        
#         cluster_hist = sp_texton_hist_normalized[curr_cluster,:].mean(axis=0)
#         propagate_selection = np.equal.outer(segmentation, curr_cluster).any(axis=2)
#         cluster_hist = np.bincount(textonmap[mask * propagate_selection].flat, minlength=n_texton)
#         cluster_hist = cluster_hist.astype(np.float)/cluster_hist.sum()
        
        centroid_hist = sp_texton_hist_normalized[curr_cluster,:].mean(axis=0)
        in_class_var = np.mean(np.sum((sp_texton_hist_normalized[curr_cluster] - centroid_hist)**2, axis=1))
        in_class_vars.append(in_class_var)
        
#         plt.bar(np.arange(n_texton), cluster_hist, color='r', width=.5);
#         plt.bar(np.arange(n_texton)+.5, sp_texton_hist_normalized[seed], color='b', width=.5);
#         plt.show()
        
        surround = set.union(*[neighbors[i] for i in curr_cluster])
        surround = surround - set(curr_cluster.tolist()) - set(bg_superpixels)

        frontier_in_cluster = set.union(*[neighbors[i] for i in surround]) & set(curr_cluster.tolist())
        
        frontier_surround_dist = np.max([np.max([D[i,j] for j in neighbors[i] if j not in bg_superpixels]) for i in frontier_in_cluster])
        frontier_surround_dists.append(frontier_surround_dist)
#         surround_hist = sp_texton_hist_normalized[list(surround),:].mean(axis=0)
#         propagate_selection = np.equal.outer(segmentation, list(surround)).any(axis=2)
#         surround_hist = np.bincount(textonmap[mask * propagate_selection].flat, minlength=n_texton)
#         surround_hist = surround_hist.astype(np.float)/surround_hist.sum()
        
#         new_sig_score = chi2(cluster_hist, surround_hist)
#         new_sig_scores.append(new_sig_score)
        
        seed_cluster_dist = np.max([chi2(seed_hist, sp_texton_hist_normalized[c]) for c in curr_cluster if c != seed]) if len(curr_cluster)> 1 else 0
#         seed_cluster_dist = chi2(seed_hist, cluster_hist)
        seed_cluster_dists.append(seed_cluster_dist)

        seed_surround_dist = np.max([chi2(seed_hist, sp_texton_hist_normalized[c]) for c in surround])
        seed_surround_dists.append(seed_surround_dist)
    #     overall_texton_hist = np.bincount(textonmap[mask].flat)
    #     overall_texton_hist_normalized = overall_texton_hist.astype(np.float) / overall_texton_hist.sum()

        clusters.append(curr_cluster)

        vis = np.exp(-2*D[:,seed])[segmentation]
        cluster_selection = np.equal.outer(segmentation, curr_cluster).any(axis=2)
#         surround_selection = np.equal.outer(segmentation, list(surround)).any(axis=2)
        vis[~cluster_selection] = 0
#         vis[surround_selection] = 1
        plt.matshow(vis, cmap=plt.cm.Greys_r);
        plt.title('%.2f'%thresh)
        plt.axis('off');
        plt.colorbar();
        plt.show()

# <codecell>

fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True);
ax[0].plot(thresh_choices[:len(in_class_vars)], in_class_vars);
ax[0].set_title('in class vars');
ax[1].plot(thresh_choices[:len(seed_surround_dists)], seed_surround_dists);
ax[1].set_title('seed vs. surround');
ax[2].plot(thresh_choices[:len(seed_cluster_dists)], seed_cluster_dists);
ax[2].set_title('seed vs. cluster');
ax[3].plot(thresh_choices[:len(frontier_surround_dists)], frontier_surround_dists);
ax[3].set_title('frontier vs. surround');

