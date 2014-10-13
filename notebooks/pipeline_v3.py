# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import cv2

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

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

import utilities
from utilities import chi2

import glob, re, os, sys, subprocess, argparse
import pprint

# <codecell>

# parse arguments
parser = argparse.ArgumentParser(
formatter_class=argparse.RawDescriptionHelpFormatter,
description='Execute feature extraction pipeline',
epilog="""
The following command processes image PMD1305_region0_reduce2_0244.tif using the parameter id nissl324.
python %s ../ParthaData/PMD1305_region0_reduce2/PMD1305_region0_reduce2_0244.tif nissl324

This script loads the parameters in params_dir. 
Results are stored in a sub-directory named <result name>_param_<parameter id>, under output_dir.
Details are in the GitHub README (https://github.com/mistycheney/BrainSaliencyDetection/blob/master/README.md)
"""%(os.path.basename(sys.argv[0]), ))

parser.add_argument("img_file", type=str, help="path to image file")
parser.add_argument("param_id", type=str, help="parameter identification name")
parser.add_argument("-o", "--output_dir", type=str, help="output directory (default: %(default)s)", default='/oasis/scratch/csd181/yuncong/output')
parser.add_argument("-p", "--params_dir", type=str, help="directory containing csv parameter files %(default)s)", default='/oasis/projects/nsf/csd181/yuncong/Brain/params')
args = parser.parse_args()

# <codecell>

def load_array(suffix):
    return utilities.load_array(suffix, img_name, param['param_id'], args.output_dir)

def save_array(arr, suffix):
    utilities.save_array(arr, suffix, img_name, param['param_id'], args.output_dir)
        
def save_img(img, suffix):
    utilities.save_img(img, suffix, img_name, param['param_id'], args.output_dir, overwrite=True)

def get_img_filename(suffix, ext='png'):
    return utilities.get_img_filename(suffix, img_name, param['param_id'], args.output_dir, ext=ext)

# <codecell>

# load parameter settings
params_dir = os.path.realpath(args.params_dir)
param_file = os.path.join(params_dir, 'param_%s.json'%args.param_id)
param_default_file = os.path.join(params_dir, 'param_default.json')
param = json.load(open(param_file, 'r'))
param_default = json.load(open(param_default_file, 'r'))

for k, v in param_default.iteritems():
    if not isinstance(param[k], basestring):
        if np.isnan(param[k]):
            param[k] = v

pprint.pprint(param)

# set image paths
img_file = os.path.realpath(args.img_file)
img_path, ext = os.path.splitext(img_file)
img_dir, img_name = os.path.split(img_path)

print img_file
img = cv2.imread(img_file, 0)
im_height, im_width = img.shape[:2]

# set output paths
output_dir = os.path.realpath(args.output_dir)

result_name = img_name + '_param_' + str(param['param_id'])
result_dir = os.path.join(output_dir, result_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# <codecell>

# Find foreground mask

print '=== finding foreground mask ==='
mask = utilities.foreground_mask(img, min_size=2500)
mask = mask > .5
# plt.imshow(mask, cmap=plt.cm.Greys_r);

# <codecell>

# Generate Gabor filter kernels

theta_interval = param['theta_interval']
n_angle = int(180/theta_interval)
freq_step = param['freq_step']
freq_max = 1./param['min_wavelen']
freq_min = 1./param['max_wavelen']
bandwidth = param['bandwidth']
n_freq = int(np.log(freq_max/freq_min)/np.log(freq_step)) + 1
frequencies = freq_max/freq_step**np.arange(n_freq)

kernels = [gabor_kernel(f, theta=t, bandwidth=bandwidth) for f in frequencies 
          for t in np.arange(0, n_angle)*np.deg2rad(theta_interval)]
kernels = map(np.real, kernels)

n_kernel = len(kernels)

print '=== filter using Gabor filters ==='
print 'num. of kernels: %d' % (n_kernel)
print 'frequencies:', frequencies
print 'wavelength (pixels):', 1/frequencies

max_kern_size = np.max([kern.shape[0] for kern in kernels])
print 'max kernel matrix size:', max_kern_size

# <codecell>

# Process the image using Gabor filters

try:
#     raise IOError
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

# Crop image border where filters show border effects

features = features[max_kern_size/2:-max_kern_size/2, max_kern_size/2:-max_kern_size/2, :]
img = img[max_kern_size/2:-max_kern_size/2, max_kern_size/2:-max_kern_size/2]
mask = mask[max_kern_size/2:-max_kern_size/2, max_kern_size/2:-max_kern_size/2]
im_height, im_width = img.shape[:2]

save_img(img, 'cropImg')

save_array(mask, 'mask')

# <codecell>

# Compute rotation-invariant texton map using K-Means

print '=== compute rotation-invariant texton map using K-Means ==='

n_texton = int(param['n_texton'])

try: 
#     raise IOError
    textonmap = load_array('textonmap')
except IOError:
    
    X = features.reshape(-1, n_feature)
    n_data = X.shape[0]
    n_splits = 1000
    n_sample = int(param['n_sample'])
    centroids = np.array(random.sample(X, n_texton))
    
    n_iter = int(param['n_iter'])

    def compute_dist_per_proc(X_partial, c_all_rot):
        D = cdist(X_partial, c_all_rot, 'sqeuclidean')
        ci, ri = np.unravel_index(D.argmin(axis=1), (n_texton, n_angle))
        return np.c_[ci, ri]

    for iteration in range(n_iter):
        
        data = random.sample(X, n_sample)
        
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
        centroids_new /= counts[:, np.newaxis]
        centroids_new[counts==0] = centroids[counts==0]
        print np.sqrt(np.sum((centroids - centroids_new)**2, axis=1)).mean()

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

    textonmap = labels.reshape(features.shape[:2])
    textonmap[~mask] = -1
    
    save_array(textonmap, 'textonmap')
    
textonmap_rgb = label2rgb(textonmap, image=None, colors=None, alpha=0.3, image_alpha=1)
save_img(textonmap_rgb, 'textonmap')

# <codecell>

# Over-segment the image into superpixels using SLIC (http://ivrg.epfl.ch/research/superpixels)

print '=== over-segment the image into superpixels based on color information ==='

img_rgb = gray2rgb(img)

try:
#     raise IOError
    segmentation = load_array('segmentation')
    
except IOError:
    segmentation = slic(img_rgb, n_segments=int(param['n_superpixels']), 
                        max_iter=10, 
                        compactness=float(param['slic_compactness']), 
                        sigma=float(param['slic_sigma']), 
                        enforce_connectivity=True)
    print 'segmentation computed'
    
    save_array(segmentation, 'segmentation')
    
sp_props = regionprops(segmentation+1, intensity_image=img, cache=True)

def obtain_props_worker(i):
    return sp_props[i].centroid, sp_props[i].area, sp_props[i].mean_intensity

r = Parallel(n_jobs=16)(delayed(obtain_props_worker)(i) for i in range(len(sp_props)))
sp_centroids = np.array([s[0] for s in r])
sp_areas = np.array([s[1] for s in r])
sp_mean_intensity = np.array([s[2] for s in r])

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

# Determine which superpixels are mostly background

def count_fg_worker(i):
    return np.count_nonzero(mask[segmentation==i])

r = Parallel(n_jobs=16)(delayed(count_fg_worker)(i) for i in range(n_superpixels))
superpixels_fg_count = np.array(r)
bg_superpixels = np.nonzero((superpixels_fg_count/sp_areas) < 0.3)[0]
# bg_superpixels = np.array(list(set(bg_superpixels.tolist()
#                           +[0,1,2,3,4,5,6,7,119,78,135,82,89,187,174,242,289]
#                           +[50,51,56,57,58,59,60,61,62,63,64,65,115,73,88,109,99,91,122,110,151,192,165,158,254,207,236,306]
#                           )))
fg_superpixels = np.array([i for i in range(n_superpixels) if i not in bg_superpixels])
print '%d background superpixels'%len(bg_superpixels)

save_array(fg_superpixels, 'fg')
save_array(bg_superpixels, 'bg')

# a = np.zeros((n_superpixels,), dtype=np.bool)
# a[fg_superpixels] = True
# plt.imshow(a[segmentation], cmap=plt.cm.Greys_r)
# plt.show()

# <codecell>

# Compute neighbor lists and connectivity matrix

from skimage.morphology import disk
from skimage.filter.rank import gradient
from scipy.sparse import coo_matrix

edge_map = gradient(segmentation.astype(np.uint8), disk(3))
neighbors = [set() for i in range(n_superpixels)]

for y,x in zip(*np.nonzero(edge_map)):
    neighbors[segmentation[y,x]] |= set(segmentation[y-2:y+2,x-2:x+2].ravel())

for i in range(n_superpixels):
    neighbors[i] -= set([i])
    
rows = np.hstack([s*np.ones((len(neighbors[s]),), dtype=np.int) for s in range(n_superpixels)])
cols = np.hstack([list(neighbors[s]) for s in range(n_superpixels)])
data = np.ones((cols.size, ), dtype=np.bool)
connectivity_matrix = coo_matrix((data, (rows, cols)), shape=(n_superpixels,n_superpixels))
connectivity_matrix = connectivity_matrix.transpose() * connectivity_matrix

# <codecell>

# compute texton histogram of every superpixel
print '=== compute texton histogram of each superpixel ==='

try:
#     raise IOError
    sp_texton_hist_normalized = load_array('texHist')
except IOError:
    def texton_histogram_worker(i):
        return np.bincount(textonmap[(segmentation == i)&(textonmap != -1)], minlength=n_texton)

    r = Parallel(n_jobs=16)(delayed(texton_histogram_worker)(i) for i in range(n_superpixels))
    sp_texton_hist = np.array(r)
    sp_texton_hist_normalized = sp_texton_hist.astype(np.float) / sp_texton_hist.sum(axis=1)[:, np.newaxis]
    save_array(sp_texton_hist_normalized, 'texHist')

# compute the null texton histogram
overall_texton_hist = np.bincount(textonmap[mask].flat)
overall_texton_hist_normalized = overall_texton_hist.astype(np.float) / overall_texton_hist.sum()

# <codecell>

# compute directionality histogram of every superpixel
print '=== compute directionality histogram of each superpixel ==='

try:
#     raise IOError
    sp_dir_hist_normalized = load_array('dirHist')
except IOError:
    f = np.reshape(features, (features.shape[0], features.shape[1], n_freq, n_angle))
    dir_energy = np.sum(abs(f), axis=2)

    def dir_histogram_worker(i):
        segment_dir_energies = dir_energy[segmentation == i].astype(np.float_).mean(axis=0)
        return segment_dir_energies    

    r = Parallel(n_jobs=16)(delayed(dir_histogram_worker)(i) for i in range(n_superpixels))
    
    sp_dir_hist = np.vstack(r)
    sp_dir_hist_normalized = sp_dir_hist/sp_dir_hist.sum(axis=1)[:,np.newaxis]
    save_array(sp_dir_hist_normalized, 'dirHist')

# compute the null directionality histogram
overall_dir_hist = sp_dir_hist_normalized[fg_superpixels].mean(axis=0)
overall_dir_hist_normalized = overall_dir_hist.astype(np.float) / overall_dir_hist.sum()

# <codecell>

# compute distance between every superpixel and the null
D_texton_null = np.squeeze(cdist(sp_texton_hist_normalized, [overall_texton_hist_normalized], chi2))
D_dir_null = np.squeeze(cdist(sp_dir_hist_normalized, [overall_dir_hist_normalized], chi2))

p = sp_texton_hist_normalized
q = sp_dir_hist_normalized

# <codecell>

re_thresh_min = 0.01
re_thresh_max = 0.8

def grow_cluster_relative_entropy(seed, debug=False, 
                                  frontier_contrast_diff_thresh = 0.1,
                                  max_cluster_size = 100):
    '''
    find the connected cluster of superpixels that have similar texture, starting from a superpixel as seed
    '''
    
    bg_set = set(bg_superpixels.tolist())
    
    if seed in bg_set:
        return [], -1

    prev_frontier_contrast = np.inf
    for re_thresh in np.arange(re_thresh_min, re_thresh_max, .01):
    
        curr_cluster = set([seed])
        frontier = [seed]

        while len(frontier) > 0:
            u = frontier.pop(-1)
            for v in neighbors[u]:
                if v in bg_superpixels or v in curr_cluster: 
                    continue

                if chi2(p[v], p[seed]) < re_thresh:
                    curr_cluster.add(v)
                    frontier.append(v)
        
        surround = set.union(*[neighbors[i] for i in curr_cluster]) - set.union(curr_cluster, bg_set)
        if len(surround) == 0:
            return curr_cluster, re_thresh

        frontier_in_cluster = set.intersection(set.union(*[neighbors[i] for i in surround]), curr_cluster)
        frontier_contrasts = [np.nanmax([chi2(p[i], p[j]) for j in neighbors[i] if j not in bg_set]) for i in frontier_in_cluster]
        frontier_contrast = np.max(frontier_contrasts)
        
        if debug:
            print 'frontier_contrast=', frontier_contrast, 'prev_frontier_contrast=', prev_frontier_contrast, 'diff=', frontier_contrast - prev_frontier_contrast
        
        if len(curr_cluster) > max_cluster_size or \
        frontier_contrast - prev_frontier_contrast > frontier_contrast_diff_thresh:
            return curr_cluster, re_thresh
        
        prev_frontier_contrast = frontier_contrast
        prev_cluster = curr_cluster
        prev_re_thresh = re_thresh
                                
    return curr_cluster, re_thresh
    

def grow_cluster_likelihood_ratio(seed, texton_model, dir_model, debug=False, lr_grow_thresh = 0.1):
    '''
    find the connected cluster of superpixels that are more likely to be explained by given model than by null, 
    starting from a superpixel as seed
    '''
    
    if seed in bg_superpixels:
        return [], -1

    curr_cluster = set([seed])
    frontier = [seed]
        
    while len(frontier) > 0:
        u = frontier.pop(-1)
        for v in neighbors[u]:
            if v in bg_superpixels or v in curr_cluster: 
                continue
            
            ratio_v = D_texton_null[v] - chi2(p[v], texton_model) +\
                        D_dir_null[v] - chi2(q[v], dir_model)
            if debug:  
                print 'u=', u, 'v=',v, 'ratio_v = ', ratio_v
                print D_texton_null[v],  chi2(p[v], texton_model), \
                        D_dir_null[v], chi2(q[v], dir_model)
            
            if ratio_v > lr_grow_thresh:
                curr_cluster.add(v)
                frontier.append(v)
                                
    return curr_cluster, lr_grow_thresh

def grow_cluster_likelihood_ratio_precomputed(seed, D_texton_model, D_dir_model, debug=False, lr_grow_thresh = 0.1):
    '''
    find the connected cluster of superpixels that are more likely to be explained by given model than by null, 
    starting from a superpixel as seed
    using pre-computed distances between model and superpixels
    '''

    if seed in bg_superpixels:
        return [], -1

    curr_cluster = set([seed])
    frontier = [seed]
        
    while len(frontier) > 0:
        u = frontier.pop(-1)
        for v in neighbors[u]:
            if v in bg_superpixels or v in curr_cluster: 
                continue
            
            ratio_v = D_texton_null[v] - D_texton_model[v] +\
                        D_dir_null[v] - D_dir_model[v]
            if debug:  
                print 'u=', u, 'v=',v, 'ratio_v = ', ratio_v
                print D_texton_null[v],  D_texton_model[v], \
                        D_dir_null[v], D_dir_model[v]
            
            if ratio_v > lr_grow_thresh:
                curr_cluster.add(v)
                frontier.append(v)
                                
    return curr_cluster, lr_grow_thresh


def visualize_cluster(scores, cluster='all', title='', filename=None):
    '''
    Generate black and white image with the cluster of superpixels highlighted
    '''
    
    vis = scores[segmentation]
    if cluster != 'all':
        cluster_selection = np.equal.outer(segmentation, cluster).any(axis=2)
        vis[~cluster_selection] = 0
    
    plt.matshow(vis, cmap=plt.cm.Greys_r);
    plt.axis('off');
    plt.title(title)
    if filename is not None:
        plt.savefig(os.path.join(result_dir, 'stages', filename + '.png'), bbox_inches='tight')
#     plt.show()
    plt.close();
    
    
def paint_cluster_on_img(cluster, title, filename=None):
    '''
    Highlight a cluster of superpixels on the real image
    '''    

    cluster_map = -1*np.ones_like(segmentation)
    for s in cluster:
        cluster_map[segmentation==s] = 1
    vis = label2rgb(cluster_map, image=img)
    plt.imshow(vis, cmap=plt.cm.Greys_r);
    plt.axis('off');
    plt.title(title)
    if filename is not None:
        plt.savefig(os.path.join(result_dir, 'stages', filename + '.png'), bbox_inches='tight')
#     plt.show()
    plt.close();

def paint_clusters_on_img(clusters, title, filename=None):
    '''
    Highlight multiple clusters with different colors on the real image
    '''
    
    cluster_map = -1*np.ones_like(segmentation)
    for i, cluster in enumerate(clusters):
        for j in cluster:
            cluster_map[segmentation==j] = i
    vis = label2rgb(cluster_map, image=img)
    plt.imshow(vis, cmap=plt.cm.Greys_r);
    plt.axis('off');
    plt.title(title)
    if filename is not None:
        plt.savefig(os.path.join(result_dir, 'stages', filename + '.png'), bbox_inches='tight')
#     plt.show()
    plt.close();

# <codecell>

# set up sigboost parameters

n_models = param['n_models']
frontier_contrast_diff_thresh = param['frontier_contrast_diff_thresh']
lr_grow_thresh = param['lr_grow_thresh']
beta = param['beta']
lr_decision_thresh = param['lr_decision_thresh']

# <codecell>

# compute RE-clusters of every superpixel
r = Parallel(n_jobs=16)(delayed(grow_cluster_relative_entropy)(i, frontier_contrast_diff_thresh=frontier_contrast_diff_thresh) 
                        for i in range(n_superpixels))
clusters = [list(c) for c, t in r]
print 'clusters computed'

# <codecell>

# create output directory
f = os.path.join(result_dir, 'stages')
if not os.path.exists(f):
    os.makedirs(f)

# initialize models
texton_models = np.zeros((n_models, n_texton))
dir_models = np.zeros((n_models, n_angle))

seed_indices = np.zeros((n_models,))

weights = np.ones((n_superpixels, ))/n_superpixels
weights[bg_superpixels] = 0

# begin boosting loop; learn one model at each iteration
for t in range(n_models):
    
    print 'model %d' % (t)
    
    # Compute significance scores for every superpixel;
    # the significance score is defined as the average log likelihood ratio in a superpixel's RE-cluster
    sig_score = np.zeros((n_superpixels, ))
    for i in fg_superpixels:
        cluster = clusters[i]
        sig_score[i] = np.mean(weights[cluster] * \
                               (D_texton_null[cluster] - np.array([chi2(p[j], p[i]) for j in cluster]) +\
                               D_dir_null[cluster] - np.array([chi2(q[j], q[i]) for j in cluster])))
 
    # Pick the most significant superpixel
    seed_sp = sig_score.argsort()[-1]
    print "most significant superpixel", seed_sp
    
    visualize_cluster(sig_score, 'all', title='significance score for each superpixel', filename='sigscore%d'%t)
    
    curr_cluster = clusters[seed_sp]
    visualize_cluster(sig_score, curr_cluster, title='distance cluster', filename='curr_cluster%d'%t)

    # models are the average of the distributions in the chosen superpixel's RE-cluster
    model_texton = sp_texton_hist_normalized[curr_cluster].mean(axis=0)
    model_dir = sp_dir_hist_normalized[curr_cluster].mean(axis=0)
    
    # Compute log likelihood ratio of this model against the null, for every superpixel
    
    # RE(pj|pm)
    D_texton_model = np.empty((n_superpixels,))
    D_texton_model[fg_superpixels] = np.array([chi2(sp_texton_hist_normalized[i], model_texton) for i in fg_superpixels])
    D_texton_model[bg_superpixels] = np.nan
    
    # RE(qj|qm)
    D_dir_model = np.empty((n_superpixels,)) 
    D_dir_model[fg_superpixels] = np.array([chi2(sp_dir_hist_normalized[i], model_dir) for i in fg_superpixels])
    D_dir_model[bg_superpixels] = np.nan
    
    # RE(pj|p0)-RE(pj|pm) + RE(qj|q0)-RE(qj|qm)
    match_scores = np.empty((n_superpixels,))
    match_scores[fg_superpixels] = D_texton_null[fg_superpixels] - D_texton_model[fg_superpixels] +\
                                    D_dir_model[fg_superpixels] - D_dir_model[fg_superpixels]
    match_scores[bg_superpixels] = 0

    visualize_cluster(match_scores, 'all', title='match score', filename='grow%d'%t)

    # Find the cluster growed from seed based on log likelihood ratio. Refer to this cluster as the LR-cluster
    matched, _ = grow_cluster_likelihood_ratio(seed_sp, model_texton, model_dir)
    matched = list(matched)

    visualize_cluster(match_scores, matched, title='growed cluster', filename='grow%d'%t)

    # Reduce the weights of superpixels in LR-cluster
    weights[matched] = weights[matched] * np.exp(-5*(D_texton_null[matched] - D_texton_model[matched] +\
                                                   D_dir_null[matched] - D_dir_model[matched])**beta)
    weights[bg_superpixels] = 0
    weights = weights/weights.sum()
    visualize_cluster((weights - weights.min())/(weights.max()-weights.min()), 'all', 
                      title='updated superpixel weights', filename='weight%d'%t)
    
    labels = -1*np.ones_like(segmentation)
    for i in matched:
        labels[segmentation == i] = 1
    real_image = label2rgb(labels, img)
    save_img(real_image, os.path.join('stage', 'real_image_model%d'%t))

    # record the model found at this round
    seed_indices[t] = seed_sp
    texton_models[t] = model_texton
    dir_models[t] = model_dir

# <codecell>

# Compute the distances between every model and every superpixel
D_texton_model = -1*np.ones((n_models, n_superpixels))
D_dir_model = -1*np.ones((n_models, n_superpixels))
D_texton_model[:, fg_superpixels] = cdist(sp_texton_hist_normalized[fg_superpixels], texton_models, chi2).T
D_dir_model[:, fg_superpixels] = cdist(sp_dir_hist_normalized[fg_superpixels], dir_models, chi2).T

# <codecell>

def find_best_model_per_proc(i):
    '''
    Worker function for finding the best models for every superpixel on the current image.
    Best model is the one with the highest likelihood ratio against the null distribution.
    '''
    
    model_score = np.empty((n_models, ))

    if i in bg_superpixels:
        return -1
    else:
        for m in range(n_models):
            matched, _ = grow_cluster_likelihood_ratio_precomputed(i, D_texton_model[m], D_dir_model[m], 
                                                                   lr_grow_thresh=lr_grow_thresh)
            matched = list(matched)
            model_score[m] = np.mean(D_texton_null[matched] - D_texton_model[m, matched] +\
                                     D_dir_null[matched] - D_dir_model[m, matched])

        best_sig = model_score.max()
        if best_sig > lr_decision_thresh: # sp whose sig is smaller than this is assigned null
          return model_score.argmax()    
    return -1


# Compute the likelihood ratio for every model on every superpixel, and return the model with the highest ratio
best_model = Parallel(n_jobs=16)(delayed(find_best_model_per_proc)(i) for i in range(n_superpixels))
labels = np.array(best_model, dtype=np.int)
save_array(labels, 'labels')

labelmap = labels[segmentation]
save_array(labelmap, 'labelmap')

labelmap_rgb = label2rgb(labelmap.astype(np.int), image=img)
save_img(labelmap_rgb, 'labelmap')

