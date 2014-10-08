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

from joblib import Parallel, delayed

import glob, re, os, sys, subprocess, argparse
import pprint

import cPickle as pickle

# <codecell>

# # parse arguments
# parser = argparse.ArgumentParser(
# formatter_class=argparse.RawDescriptionHelpFormatter,
# description='Supervised learning',
# # epilog="""
# # """%(os.path.basename(sys.argv[0]))
# )

# parser.add_argument("labeling_fn", type=str, help="path to labeling file")
# parser.add_argument("-o", "--output_dir", type=str, help="output directory (default: %(default)s)", default='/oasis/scratch/csd181/yuncong/output')
# args = parser.parse_args()

class args(object):
    labeling_fn = 'output/RS141_x5_0001_redNissl/RS141_x5_0001_redNissl_yuncong_141002050817.pkl'
    output_dir = '/oasis/scratch/csd181/yuncong/output'
    params_dir = '/oasis/projects/nsf/csd181/yuncong/Brain/params'

# <codecell>

stack_name, resolution, slice_id, params_name, username, logout_time = os.path.basename(args.labeling_fn)[:-4].split('_')
labeling = pickle.load(open(args.labeling_fn, 'r'))

instance_name = '_'.join([stack_name, resolution, slice_id, params_name])
parent_labeling_name = username + '_' + logout_time

def fullname(obj_name, ext):
    return os.path.join(args.output_dir, instance_name, instance_name + '_' + obj_name + '.' + ext)

segmentation = np.load(fullname('segmentation', 'npy'))

n_superpixels = len(segmentation)

# <codecell>

sp_texton_hist_normalized = np.load(fullname('sp_texton_hist_normalized', 'npy'))
sp_dir_hist_normalized = np.load(fullname('sp_dir_hist_normalized', 'npy'))

# <codecell>

labellist = labeling['final_labellist']

texton_models = []
dir_models = []
for i in range(np.max(labellist)+1):
    sps = np.where(labellist == i)[0]
    if len(sps) > 0:
        texton_model = sp_texton_hist_normalized[sps, :].mean(axis=0)
        texton_models.append(texton_model)
        dir_model = sp_dir_hist_normalized[sps, :].mean(axis=0)
        dir_models.append(dir_model)
        
n_models = len(texton_models)

# <codecell>

mask = np.load(fullname('mask','npy'))
fg_superpixels = np.load(fullname('fg','npy'))
bg_superpixels = np.load(fullname('bg','npy'))

# <codecell>

D_texton_model = -1*np.ones((n_models, n_superpixels))
D_dir_model = -1*np.ones((n_models, n_superpixels))
D_texton_model[:, fg_superpixels] = cdist(sp_texton_hist_normalized[fg_superpixels], texton_models, chi2).T
D_dir_model[:, fg_superpixels] = cdist(sp_dir_hist_normalized[fg_superpixels], dir_models, chi2).T

textonmap = np.load(fullname('textonmap', 'npy'))
overall_texton_hist = np.bincount(textonmap[mask].flat)
overall_texton_hist_normalized = overall_texton_hist.astype(np.float) / overall_texton_hist.sum()
overall_dir_hist = sp_dir_hist_normalized[fg_superpixels].mean(axis=0)
overall_dir_hist_normalized = overall_dir_hist.astype(np.float) / overall_dir_hist.sum()
D_texton_null = np.squeeze(cdist(sp_texton_hist_normalized, [overall_texton_hist_normalized], chi2))
D_dir_null = np.squeeze(cdist(sp_dir_hist_normalized, [overall_dir_hist_normalized], chi2))

# <codecell>

# load parameter settings
params_dir = os.path.realpath(args.params_dir)
param_file = os.path.join(params_dir, 'param_%s.json'%params_name)
param_default_file = os.path.join(params_dir, 'param_default.json')
param = json.load(open(param_file, 'r'))
param_default = json.load(open(param_default_file, 'r'))

for k, v in param_default.iteritems():
    if not isinstance(param[k], basestring):
        if np.isnan(param[k]):
            param[k] = v

pprint.pprint(param)

lr_decision_thresh = param['lr_decision_thresh']
lr_grow_thresh = param['lr_grow_thresh']

def f(i):
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

r = Parallel(n_jobs=16)(delayed(f)(i) for i in range(n_superpixels))
labels = np.array(r, dtype=np.int)
save_array(labels, 'labels')

labelmap = labels[segmentation]
save_array(labelmap, 'labelmap')

labelmap_rgb = label2rgb(labelmap.astype(np.int), image=img)
save_img(labelmap_rgb, 'labelmap')


dt = datetime.datetime.now().strftime("%y%m%d%H%M%S")

new_labeling = {
'username': 'sigboost',
'parent_labeling_name': parent_labeling_name,
'logout_time': dt,
'init_labellist': labellist,
'final_labellist': labels,
'labelnames': None,
'history': history
}

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

pickle.dump(open(fullname('neighbors','pkl'), 'w'))

# <codecell>

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

