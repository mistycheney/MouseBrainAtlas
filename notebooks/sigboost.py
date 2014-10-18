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
from utilities import chi2

from joblib import Parallel, delayed

import glob, re, os, sys, subprocess, argparse
import pprint
import cPickle as pickle

# <codecell>

# parser = argparse.ArgumentParser(
# formatter_class=argparse.RawDescriptionHelpFormatter,
# description='Semi-supervised Sigboost',
# epilog="""%s
# """%(os.path.basename(sys.argv[0]), ))

# parser.add_argument("stack_name", type=str, help="stack name")
# parser.add_argument("resolution", type=str, help="resolution string")
# parser.add_argument("slice_num", type=str, help="slice number, zero-padded to 4 digits")
# parser.add_argument("param_id", type=str, help="parameter identification name")
# args = parser.parse_args()

class args:
    stack_name = 'RS141'
    resolution = 'x5'
    slice_num = '0002'
    params_name = 'redNissl'
    models_fn = '/home/yuncong/BrainLocal/DavidData/RS141/x5/0001/redNissl/labelings/RS141_x5_0001_redNissl_models.pkl'

data_dir = '/home/yuncong/BrainLocal/DavidData'
repo_dir = '/home/yuncong/BrainSaliencyDetection'
params_dir = os.path.join(repo_dir, 'params')

# <codecell>

# stack_name, resolution, slice_id, params_name, username, logout_time = os.path.basename(args.labeling_fn)[:-4].split('_')

stack_name = args.stack_name
resolution = args.resolution
slice_id = args.slice_num
params_name = args.params_name

results_dir = os.path.join(data_dir, stack_name, resolution, slice_id, params_name, 'pipelineResults')
labelings_dir = os.path.join(data_dir, stack_name, resolution, slice_id, params_name, 'labelings')

instance_name = '_'.join([stack_name, resolution, slice_id, params_name])
# parent_labeling_name = username + '_' + logout_time
parent_labeling_name = None

def full_object_name(obj_name, ext):
    return os.path.join(data_dir, stack_name, resolution, slice_id, params_name, 'pipelineResults', instance_name + '_' + obj_name + '.' + ext)

segmentation = np.load(full_object_name('segmentation', 'npy'))
n_superpixels = np.max(segmentation) + 1

# load parameter settings
params_dir = os.path.realpath(params_dir)
param_file = os.path.join(params_dir, 'param_%s.json'%params_name)
param_default_file = os.path.join(params_dir, 'param_default.json')
param = json.load(open(param_file, 'r'))
param_default = json.load(open(param_default_file, 'r'))

for k, v in param_default.iteritems():
    if not isinstance(param[k], basestring):
        if np.isnan(param[k]):
            param[k] = v

pprint.pprint(param)

# <codecell>

sp_texton_hist_normalized = np.load(full_object_name('texHist', 'npy'))
sp_dir_hist_normalized = np.load(full_object_name('dirHist', 'npy'))
p = sp_texton_hist_normalized
q = sp_dir_hist_normalized

# labellist = labeling['final_labellist']

models = pickle.load(open(args.models_fn, 'r'))
n_models = len(models)

texton_models = [model['texton_hist'] for model in models]
dir_models = [model['dir_hist'] for model in models]

mask = np.load(full_object_name('cropMask','npy'))
fg_superpixels = np.load(full_object_name('fg','npy'))
bg_superpixels = np.load(full_object_name('bg','npy'))
neighbors = np.load(full_object_name('neighbors','npy'))

D_texton_model = -1*np.ones((n_models, n_superpixels))
D_dir_model = -1*np.ones((n_models, n_superpixels))
D_texton_model[:, fg_superpixels] = cdist(sp_texton_hist_normalized[fg_superpixels], texton_models, chi2).T
D_dir_model[:, fg_superpixels] = cdist(sp_dir_hist_normalized[fg_superpixels], dir_models, chi2).T

textonmap = np.load(full_object_name('texMap', 'npy'))
overall_texton_hist = np.bincount(textonmap[mask].flat)

overall_texton_hist_normalized = overall_texton_hist.astype(np.float) / overall_texton_hist.sum()

overall_dir_hist = sp_dir_hist_normalized[fg_superpixels].mean(axis=0)

overall_dir_hist_normalized = overall_dir_hist.astype(np.float) / overall_dir_hist.sum()

D_texton_null = np.squeeze(cdist(sp_texton_hist_normalized, [overall_texton_hist_normalized], chi2))
D_dir_null = np.squeeze(cdist(sp_dir_hist_normalized, [overall_dir_hist_normalized], chi2))

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

# <codecell>

# set up sigboost parameters

# n_models = param['n_models']
n_models = None
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
stages_dir = os.path.join(results_dir, 'stages')
if not os.path.exists(stages_dir):
    os.makedirs(stages_dir)

# initialize models
texton_models = np.zeros((n_models, n_texton))
dir_models = np.zeros((n_models, n_angle))

seed_indices = np.zeros((n_models,))

weights = np.ones((n_superpixels, ))/n_superpixels
weights[bg_superpixels] = 0

# <codecell>



# begin boosting loop; learn one model at each iteration
for t in range(n_models):
    
    print 'model %d' % (t)
    
    # Compute significance scores for every superpixel;
    # i.e. the significance of using the appearance of superpixel i as model
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

