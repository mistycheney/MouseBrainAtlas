# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

import numpy as np
import cv2

import matplotlib.pyplot as plt

import random, itertools, sys, os
from multiprocessing import Pool
import json

from skimage.segmentation import slic, mark_boundaries
# from skimage.measure import regionprops
# from skimage.util import img_as_ubyte
from skimage.color import hsv2rgb, label2rgb, gray2rgb
# from skimage.morphology import disk
# from skimage.filter.rank import gradient
# from skimage.filter import gabor_kernel
# from skimage.transform import rescale, resize

# from scipy.ndimage import gaussian_filter, measurements
# from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, squareform, euclidean, cdist
# from scipy.signal import fftconvolve

# from IPython.display import FileLink, Image, FileLinks

import utilities
from utilities import chi2

from joblib import Parallel, delayed

import glob
import re
import os
import sys
import subprocess
import argparse
import pprint

import cPickle as pickle


data_dir = '/home/yuncong/BrainLocal/DavidData'
repo_dir = '/home/yuncong/BrainSaliencyDetection'
params_dir = os.path.join(repo_dir, 'params')

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
    models_fn = '/home/yuncong/BrainLocal/DavidData/RS141/x5/0001/redNissl/labelings/RS141_x5_0001_redNissl_models.pkl'
#     labeling_fn = '/home/yuncong/BrainLocal/DavidData/RS141/x5/0001/redNissl/labelings/RS141_x5_0001_redNissl_anon_10132014165928.pkl'

# <codecell>

def save_array(arr, suffix):
    utilities.save_array(arr, suffix, instance_name=instance_name, results_dir=results_dir)
        
def save_image(img, suffix):
    utilities.save_image(img, suffix, instance_name=instance_name, results_dir=results_dir, overwrite=True)

def load_image(suffix):
    return utilities.load_image(suffix, instance_name=instance_name, results_dir=results_dir)

# <codecell>

# stack_name, resolution, slice_id, params_name, username, logout_time = os.path.basename(args.labeling_fn)[:-4].split('_')

stack_name = 'RS141'
resolution = 'x5'
slice_id = '0002'
params_name = 'redNissl'

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

# <codecell>

# labellist = labeling['final_labellist']

models = pickle.load(open(args.models_fn, 'r'))
n_models = len(models)

texton_models = [model['texton_hist'] for model in models]
dir_models = [model['dir_hist'] for model in models]

# <codecell>

# plt.bar(range(100), texton_models[0])
# plt.show()

# <codecell>

mask = np.load(full_object_name('cropMask','npy'))
fg_superpixels = np.load(full_object_name('fg','npy'))
bg_superpixels = np.load(full_object_name('bg','npy'))
neighbors = np.load(full_object_name('neighbors','npy'))

# <codecell>

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

img = load_image('cropImg')

# lr_decision_thresh = param['lr_decision_thresh']
lr_decision_thresh = .2
lr_grow_thresh = param['lr_grow_thresh']

print lr_decision_thresh, lr_grow_thresh

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

# <codecell>

img = load_image('cropImg')

# lr_decision_thresh = param['lr_decision_thresh']
lr_decision_thresh = .2
lr_grow_thresh = param['lr_grow_thresh']

print lr_decision_thresh, lr_grow_thresh

def f(i):
    model_score = np.empty((n_models, ))

    if i in bg_superpixels:
        return -1
    else:
        for m in range(n_models):
            print 'model', m
            matched, _ = grow_cluster_likelihood_ratio_precomputed(i, D_texton_model[m], D_dir_model[m], 
                                                                   lr_grow_thresh=lr_grow_thresh)
            
            a = utilities.paint_superpixels_on_image(matched, segmentation, img=img)
            plt.imshow(a)
            plt.show()
            
            matched = list(matched)
            model_score[m] = np.mean(D_texton_null[matched] - D_texton_model[m, matched] +\
                                     D_dir_null[matched] - D_dir_model[m, matched])
            print model_score[m]

        best_sig = model_score.max()
        if best_sig > lr_decision_thresh: # sp whose sig is smaller than this is assigned null
          return model_score.argmax()    
    return -1


# f(1382) #0001
f(811) # axon bundles on 0002

# <codecell>

labelmap = labels[segmentation]

hc_colors = np.loadtxt('../visualization/high_contrast_colors.txt', skiprows=1)

img = load_image('cropImg')

labelmap_rgb = label2rgb(labelmap.astype(np.int), image=img, colors=hc_colors[1:]/255., alpha=0.1, 
                         image_alpha=1, bg_color=hc_colors[0]/255.)

import datetime
dt = datetime.datetime.now().strftime("%m%d%Y%H%M%S")

new_labeling = {
'username': 'sigboost',
'parent_labeling_name': None,
'login_time': dt,
'logout_time': dt,
'init_labellist': None,
'final_labellist': labels,
'labelnames': None,
'history': None
}

labelmap_rgb = utilities.regulate_img(labelmap_rgb)
new_preview_fn = os.path.join(labelings_dir, instance_name + '_sigboost_' + dt + '_preview.tif')
cv2.imwrite(new_preview_fn, labelmap_rgb)

new_labeling_fn = os.path.join(labelings_dir, instance_name + '_sigboost_' + dt + '.pkl')
pickle.dump(new_labeling, open(new_labeling_fn, 'w'))

