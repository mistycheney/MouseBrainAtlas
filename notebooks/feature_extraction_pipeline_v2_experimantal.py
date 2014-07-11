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

import utilities

from joblib import Parallel, delayed

import glob, re, os, sys, subprocess, argparse
import pprint

%autosave 60

# <codecell>

# parser = argparse.ArgumentParser(
# formatter_class=argparse.RawDescriptionHelpFormatter,
# description='Execute feature extraction pipeline',
# epilog="""
# The following command processes image PMD1305_region0_reduce2_0244.tif using the parameter setting number 10.
# python %s ../data/PMD1305_region0_reduce2/PMD1305_region0_reduce2_0244.tif 10

# This script loads the parameters in ../params. 
# The meanings of all parameters are explained in GitHub README.

# The results are stored in a sub-directory under the output directory. 
# The sub-directory is named <dataset name>_reduce<reduce level>_<image index>_param<parameter id>.
# The content of this sub-directory are the .npy files or image files with different _<suffix>. See GitHub README for details of these files.

# * GitHub README *
# https://github.com/mistycheney/BrainSaliencyDetection/blob/master/README.md
# """%(os.path.basename(sys.argv[0]), ))

# parser.add_argument("img_file", type=str, help="path to image file")
# parser.add_argument("param_id", type=str, help="parameter identification name")
# parser.add_argument("-o", "--output_dir", type=str, help="output directory (default: %(default)s)", default='/oasis/scratch/csd181/yuncong/output')
# parser.add_argument("-p", "--params_dir", type=str, help="directory containing csv parameter files %(default)s)", default='/oasis/projects/nsf/csd181/yuncong/Brain/params')
# args = parser.parse_args()

class args:
    param_id = 'nissl324'
    img_file = '../ParthaData/PMD1305_region0_reduce2/PMD1305_region0_reduce2_0244.tif'
    output_dir = '/oasis/scratch/csd181/yuncong/output'
    params_dir = '/oasis/projects/nsf/csd181/yuncong/Brain/params'

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

img_file = os.path.realpath(args.img_file)
img_path, ext = os.path.splitext(img_file)
img_dir, img_name = os.path.split(img_path)

img = cv2.imread(img_file, 0)
im_height, im_width = img.shape[:2]

output_dir = os.path.realpath(args.output_dir)

result_name = img_name + '_param_' + str(param['param_id'])
result_dir = os.path.join(output_dir, result_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# <codecell>

print '=== finding foreground mask ==='
mask = utilities.foreground_mask(img, min_size=2000)
mask = mask > .5
plt.imshow(mask, cmap=plt.cm.Greys_r);

# <codecell>

theta_interval = param['theta_interval']
n_angle = int(180/theta_interval)
freq_step = param['freq_step']
freq_max = 1./param['min_wavelen']
freq_min = 1./param['max_wavelen']
bandwidth = param['bandwidth']
n_freq = int(np.log(freq_max/freq_min)/np.log(freq_step)) + 1
frequencies = freq_max/freq_step**np.arange(n_freq)

kernels = [gabor_kernel(f, theta=t, bandwidth=bandwidth) for f in frequencies 
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
im_height, im_width = img.shape[:2]

save_img(img, 'img_cropped')

# <codecell>

print '=== compute rotation-invariant texton map using K-Means ==='

n_texton = int(param['n_texton'])

try: 
    raise IOError
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

print '=== over-segment the image into superpixels based on color information ==='

img_rgb = gray2rgb(img)

try:
    raise IOError
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

def foo2(i):
    return sp_props[i].centroid, sp_props[i].area, sp_props[i].mean_intensity

r = Parallel(n_jobs=16)(delayed(foo2)(i) for i in range(len(sp_props)))
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

FileLink(get_img_filename('segmentation')[30:])

# <codecell>

FileLink(get_img_filename('textonmap')[30:])

# <codecell>

def foo(i):
    return np.count_nonzero(mask[segmentation==i])

r = Parallel(n_jobs=16)(delayed(foo)(i) for i in range(n_superpixels))
superpixels_fg_count = np.array(r)
bg_superpixels = np.nonzero((superpixels_fg_count/sp_areas) < 0.3)[0]
bg_superpixels = np.array(list(set(bg_superpixels.tolist()
                          +[0,1,2,3,4,5,6,7,119,78,135,82,89,187,174,242,289]
                          +[50,51,56,57,58,59,60,61,62,63,64,65,115,73,88,109,99,91,122,110,151,192,165,158,254,207,236,306]
                          )))
fg_superpixels = np.array([i for i in range(n_superpixels) if i not in bg_superpixels])
print '%d background superpixels'%len(bg_superpixels)

a = np.zeros((n_superpixels,), dtype=np.bool)
a[fg_superpixels] = True
plt.imshow(a[segmentation], cmap=plt.cm.Greys_r)
plt.show()

# <codecell>

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

print '=== compute texton histogram of each superpixel ==='

try:
    raise IOError
    sp_texton_hist_normalized = load_array('sp_texton_hist_normalized')
except IOError:
    def bar(i):
        return np.bincount(textonmap[(segmentation == i)&(textonmap != -1)], minlength=n_texton)

    r = Parallel(n_jobs=16)(delayed(bar)(i) for i in range(n_superpixels))
    sp_texton_hist = np.array(r)
    sp_texton_hist_normalized = sp_texton_hist.astype(np.float) / sp_texton_hist.sum(axis=1)[:, np.newaxis]
    save_array(sp_texton_hist_normalized, 'sp_texton_hist_normalized')
    
overall_texton_hist = np.bincount(textonmap[mask].flat)
overall_texton_hist_normalized = overall_texton_hist.astype(np.float) / overall_texton_hist.sum()

# <codecell>

print '=== compute directionality histogram of each superpixel ==='

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
    
overall_dir_hist = sp_dir_hist_normalized[fg_superpixels].mean(axis=0)
overall_dir_hist_normalized = overall_dir_hist.astype(np.float) / overall_dir_hist.sum()

# <codecell>

def chi2(u,v):
    r = np.nansum((u-v)**2/(u+v))
    return r

D_texton_null = np.squeeze(cdist(sp_texton_hist_normalized, [overall_texton_hist_normalized], chi2))
D_dir_null = np.squeeze(cdist(sp_dir_hist_normalized, [overall_dir_hist_normalized], chi2))
p = sp_texton_hist_normalized
q = sp_dir_hist_normalized

# <codecell>

re_thresh_min = 0.2
re_thresh_max = 0.8     

def grow_cluster_relative_entropy(seed, frontier_contrast_thresh,
                                  debug=False, 
                                  max_cluster_size = 200):
    
    bg_set = set(bg_superpixels.tolist())
    
    if seed in bg_set:
        return [], -1

#     prev_frontier_contrast = 0
    
#     re_thresh = 0.15
    
    for re_thresh in np.arange(re_thresh_min, re_thresh_max, .01):
    
        curr_cluster = set([seed])
        frontier = [seed]

        while len(frontier) > 0:
            u = frontier.pop(-1)
            for v in neighbors[u]:
                if v in bg_superpixels or v in curr_cluster: 
                    continue

    #             print u,v, chi2(p[v], p[u])
    #                 if chi2(p[v], p[u]) < re_thresh:
                if chi2(p[v], p[seed]) + chi2(q[v], q[seed]) < re_thresh:
                    curr_cluster.add(v)
                    frontier.append(v)

        surround = set.union(*[neighbors[i] for i in curr_cluster]) - set.union(curr_cluster, bg_set)
        assert len(surround) != 0, seed

        paint_cluster_on_img(surround, 'surround')
        
        frontier_in_cluster = set.intersection(set.union(*[neighbors[i] for i in surround]), curr_cluster)
        frontier_contrasts = [np.nanmin([chi2(p[i], p[j]) + chi2(q[i], q[j]) for j in neighbors[i] if j not in bg_set]) for i in frontier_in_cluster]
        frontier_contrast = np.min(frontier_contrasts)

        paint_cluster_on_img(frontier_in_cluster, 'frontier_in_cluster')
        
    #         visualize_cluster(np.ones((n_superpixels,)), list(curr_cluster), 
    #                       title='thresh = %.2f, frontier_contrast = %f, frontier_contrast_curr_prev = %f'%(re_thresh, frontier_contrast, frontier_contrast - prev_frontier_contrast))

#         title = 'thresh = %f, frontier_contrast = %f, frontier_contrast_curr_prev = %f'%(re_thresh, frontier_contrast, frontier_contrast - prev_frontier_contrast)
        title = 'thresh = %f, frontier_contrast = %f'%(re_thresh, frontier_contrast)
        print title
        #         paint_cluster_on_img(curr_cluster, title=title)

        if len(curr_cluster) > max_cluster_size or \
            frontier_contrast > frontier_contrast_thresh:
#         abs(frontier_contrast - prev_frontier_contrast) > frontier_contrast_thresh:
            return curr_cluster, re_thresh

#         prev_frontier_contrast = frontier_contrast
#         prev_cluster = curr_cluster
#         prev_re_thresh = re_thresh
                                
    return curr_cluster, re_thresh
    

def grow_cluster_likelihood_ratio(seed, texton_model, dir_model, debug, grow_thresh = -0.1):
    
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
            if ratio_v > grow_thresh:
                curr_cluster.add(v)
                frontier.append(v)
                                
    return curr_cluster, grow_thresh


def paint_clusters_on_img(clusters, title=''):
    labels = -1*np.ones_like(segmentation)
    for n, c in enumerate(clusters):
        for i in cluster:
            labels[segmentation == i] = n +1
    real_image = label2rgb(labels, img)
    plt.figure(figsize=(20,10))
    plt.imshow(real_image, cmap=plt.cm.Greys_r)
    plt.title(title)
    plt.axis('off')
    plt.show()

    

def paint_cluster_on_img(cluster, title='', label=1):
    labels = -1*np.ones_like(segmentation)
    for i in cluster:
        labels[segmentation == i] = label
    real_image = label2rgb(labels, img)
    plt.figure(figsize=(20,10))
    plt.imshow(real_image, cmap=plt.cm.Greys_r)
    plt.title(title)
    plt.axis('off')
    plt.show()


def visualize_cluster(scores, cluster='all', title='', filename=None):
    vis = scores[segmentation]
    if cluster != 'all':
        cluster_selection = np.equal.outer(segmentation, cluster).any(axis=2)
        vis[~cluster_selection] = 0
    
    plt.matshow(vis, cmap=plt.cm.Greys_r);
    plt.axis('off');
    plt.title(title)
    if filename is not None:
        plt.savefig(os.path.join(args.output_dir, 'stages', filename + '.png'), bbox_inches='tight')
    plt.show()
        #     plt.close();

# <codecell>

c, _ = grow_cluster_relative_entropy(1495, frontier_contrast_thresh=0.1, debug=False, max_cluster_size=300)
# c, _ = grow_cluster_likelihood_ratio(i, p[i], q[i], debug=False)
paint_cluster_on_img(c)

# <codecell>

r = Parallel(n_jobs=16)(delayed(grow_cluster_relative_entropy)(i, debug=False) 
                        for i in range(n_superpixels))
clusters = [list(c) for c, t in r]
print 'clusters computed'

plt.hist([len(c) for c in clusters], bins=200);
plt.title('distribution of cluster sizes')
plt.show()

# <codecell>

f = os.path.join(args.output_dir, 'stages')
if not os.path.exists(f):
    os.makedirs(f)

n_models = 10

texton_models = np.zeros((n_models, n_texton))
dir_models = np.zeros((n_models, n_angle))

seed_indices = np.zeros((n_models,))

weights = np.ones((n_superpixels, ))/n_superpixels
weights[bg_superpixels] = 0

for t in range(n_models):
    
    print 'model %d' % (t)
    
    sig_score = np.zeros((n_superpixels, ))
    for i in fg_superpixels:
        cluster = clusters[i]
        sig_score[i] = np.mean(weights[cluster] * \
                               (D_texton_null[cluster] - np.array([chi2(p[j], p[i]) for j in cluster]) +\
                               D_dir_null[cluster] - np.array([chi2(q[j], q[i]) for j in cluster])))
 
    seed_sp = sig_score.argsort()[-1]
#     seed_sp = 1622
    print "most significant superpixel", seed_sp
    
    visualize_cluster(sig_score, 'all', title='significance score for each superpixel', filename='sigscore%d'%t)
    
    curr_cluster = clusters[seed_sp]
    visualize_cluster(sig_score, curr_cluster, title='distance cluster', filename='curr_cluster%d'%t)
    
    model_texton = sp_texton_hist_normalized[curr_cluster].mean(axis=0)
    model_dir = sp_dir_hist_normalized[curr_cluster].mean(axis=0)
    
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

    
    matched, _ = grow_cluster_likelihood_ratio(seed_sp, model_texton, model_dir, debug=True)
    matched = list(matched)

    visualize_cluster(match_scores, matched, title='growed cluster', filename='grow%d'%t)
    
    beta = 1.
    weights[matched] = weights[matched] * np.exp(-5*(D_texton_null[matched] - D_texton_model[matched] +\
                                                   D_dir_null[matched] - D_dir_model[matched])**beta)
    weights[bg_superpixels] = 0
    weights = weights/weights.sum()
    visualize_cluster((weights - weights.min())/(weights.max()-weights.min()), 'all', title='updated superpixel weights', filename='weight%d'%t)
    
    labels = -1*np.ones_like(segmentation)
    for i in matched:
        labels[segmentation == i] = 1
    real_image = label2rgb(labels, img)
    plt.imshow(real_image, cmap=plt.cm.Greys_r)
    plt.axis('off')
    plt.show()
        
    seed_indices[t] = seed_sp
    texton_models[t] = model_texton
    dir_models[t] = model_dir

# <codecell>

%%time

D_texton_model = np.empty((n_models, n_superpixels))
D_dir_model = np.empty((n_models, n_superpixels))
D_texton_null = cdist(sp_texton_hist_normalized, [overall_texton_hist_normalized], chi2)
D_dir_null = cdist(sp_dir_hist_normalized, [overall_dir_hist_normalized], chi2)

for m in range(n_models):
    model_texton = texton_models[m]
    model_dir = dir_models[m]

    D_texton_model[m, fg_superpixels][:, np.newaxis] = cdist(sp_texton_hist_normalized[fg_superpixels], [model_texton], chi2)
    D_dir_model[m, fg_superpixels][:, np.newaxis] = cdist(sp_dir_hist_normalized[fg_superpixels], [model_dir], chi2)

# <codecell>

%%time

T = 0.3
model_score = np.empty((n_models, ))

# for i in range(n_superpixels):
#     print i
#     if i in bg_superpixels:
#         print -1
#     else:
#         for m in range(n_models):
#             matched, _ = grow_cluster_sig(i, D_texton_null, D_texton_model[m], 
#                                           D_dir_null, D_dir_model[m])
#             matched = list(matched)
#             model_score[m] = np.mean(D_texton_null[matched] - D_texton_model[m, matched] +\
#                                      D_dir_null[matched] - D_dir_model[m, matched])

#         best_sig = model_score.max()
#         if best_sig > T: # sp whose sig is smaller than this is assigned null
#           print model_score.argmax()

def f(i):
    if i in bg_superpixels:
        return -1
    else:
        for m in range(n_models):
            matched, _ = grow_cluster_sig(i, D_texton_null, D_texton_model[m], 
                                          D_dir_null, D_dir_model[m])
            matched = list(matched)
            model_score[m] = np.mean(D_texton_null[matched] - D_texton_model[m, matched] +\
                                     D_dir_null[matched] - D_dir_model[m, matched])

        best_sig = model_score.max()
        if best_sig > T: # sp whose sig is smaller than this is assigned null
          return model_score.argmax()    
    return -1

r = Parallel(n_jobs=16)(delayed(f)(i) for i in range(n_superpixels))
labels = np.array(r, dtype=np.int)
save_array(labels, 'labels')

# labelmap = labels[segmentation]
# save_array(labelmap, 'labelmap')

# labelmap_rgb = label2rgb(labelmap.astype(np.int), image=img)
# save_img(labelmap_rgb, 'labelmap')

