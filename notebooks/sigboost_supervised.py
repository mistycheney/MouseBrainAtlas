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
    models_fn = '/home/yuncong/BrainLocal/DavidData_v3/RS141/x5/RS141_x5_models.pkl'
    labeling_fn = '/home/yuncong/BrainLocal/DavidData_v3/RS141/x5/0000/labelings/RS141_x5_0000_anon_10202014204123.pkl'

data_dir = '/home/yuncong/BrainLocal/DavidData_v3'
repo_dir = '/home/yuncong/BrainSaliencyDetection'
params_dir = os.path.join(repo_dir, 'params')


# <codecell>

stack_name = args.stack_name
resolution = args.resolution
slice_id = args.slice_num
params_name = args.params_name

results_dir = os.path.join(data_dir, stack_name, resolution, slice_id, params_name+'_pipelineResults')
labelings_dir = os.path.join(data_dir, stack_name, resolution, slice_id, 'labelings')

image_name = '_'.join([stack_name, resolution, slice_id])
instance_name = '_'.join([stack_name, resolution, slice_id, params_name])

_, _, _, username, logout_time = os.path.basename(args.labeling_fn)[:-4].split('_')
parent_labeling_name = username + '_' + logout_time
# parent_labeling_name = None

def full_object_name(obj_name, ext):
    return os.path.join(data_dir, stack_name, resolution, slice_id, params_name+'_pipelineResults', instance_name + '_' + obj_name + '.' + ext)


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

# def circle_list_to_labeling_field(self, circle_list):
#     label_circles = []
#     for c in circle_list:
#         label = np.where(np.all(self.colors == c.get_facecolor()[:3], axis=1))[0][0] - 1
#         label_circles.append((int(c.center[0]), int(c.center[1]), c.radius, label))
#     return label_circles


def labeling_field_to_labelmap(labeling_field, size):
    
    labelmap = -1*np.ones(size, dtype=np.int)

    for cx,cy,cradius,label in labeling_field:
        for x in np.arange(cx-cradius, cx+cradius):
            for y in np.arange(cy-cradius, cy+cradius):
                if (cx-x)**2+(cy-y)**2 <= cradius**2:
                    labelmap[int(y),int(x)] = label
    return labelmap

    
def count_unique(keys):
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    return uniq_keys, np.bincount(bins)


def label_superpixels(labelmap, segmentation):
    n_superpixels = len(np.unique(segmentation))
    labellist = -1*np.ones((n_superpixels,), dtype=np.int)
    for sp in range(n_superpixels):
        in_sp_labels = labelmap[segmentation==sp]
        labels, counts = count_unique(in_sp_labels)
        dominant_label = int(labels[counts.argmax()])
        if dominant_label != -1:
            labellist[sp] = dominant_label
    return labellist
        
        
def generate_models(labellist, sp_texton_hist_normalized):
    
    models = []
    for i in range(np.max(labellist)+1):
        sps = np.where(labellist == i)[0]
        print i, sps
        model = {}
        if len(sps) > 0:
            texton_model = sp_texton_hist_normalized[sps, :].mean(axis=0)
            model['texton_hist'] = texton_model
#             dir_model = sp_dir_hist_normalized[sps, :].mean(axis=0)
#             model['dir_hist'] = dir_model
            models.append(model)

    n_models = len(models)
    print n_models, 'models'
    
    return models

#     labelmap = labellist[segmentation]

#     for l in range(n_models):
#         matched_rows, matched_cols = np.where(labelmap == l)
#         ymin = matched_rows.min()
#         ymax = matched_rows.max()
#         xmin = matched_cols.min()
#         xmax = matched_cols.max()
#         models[l]['bounding_box'] = (xmin, ymin, xmax-xmin+1, ymax-ymin+1)    

# <codecell>

def get_max_kernel_size(param):

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
    print max_kern_size
    return max_kern_size

def models_from_labeling(labeling_fn):
    stack, resol, slice, username, logout_time = os.path.basename(labeling_fn)[:-4].split('_')
    img_fn = os.path.join(data_dir, stack, resol, slice, '_'.join([stack, resol, slice])+'.tif')
    img = cv2.imread(img_fn, 0)
    
    cropImg_fn = os.path.join(data_dir, stack, resol, slice, params_name+'_pipelineResults', 
                 '_'.join([stack, resol, slice, params_name]) + '_cropImg.tif')
    cropImg = cv2.imread(cropImg_fn, 0)
    
    labeling = pickle.load(open(labeling_fn, 'r'))
    labelmap = labeling_field_to_labelmap(labeling['final_label_circles'], size=img.shape)
    
    max_kern_size = get_max_kernel_size(param)
    
    cropped_labelmap = labelmap[max_kern_size/2:-max_kern_size/2, max_kern_size/2:-max_kern_size/2]

#     show_labelmap(labelmap, img)

    
    segmentation_fn = os.path.join(data_dir, stack, resol, slice, params_name+'_pipelineResults', 
                 '_'.join([stack, resol, slice, params_name]) + '_segmentation.npy')
    print segmentation_fn
    segmentation = np.load(segmentation_fn)

    labellist = label_superpixels(cropped_labelmap, segmentation)


    segmentation_vis_fn = os.path.join(data_dir, stack, resol, slice, params_name+'_pipelineResults', 
                 '_'.join([stack, resol, slice, params_name]) + '_segmentation.tif')
    segvis = cv2.imread(segmentation_vis_fn, 0)

    
#     show_labelmap(labellist[segmentation], segvis)
    
    f = os.path.join(data_dir, stack, resol, slice, params_name+'_pipelineResults', 
                 '_'.join([stack, resol, slice, params_name]) + '_texHist.npy')
    print f
    
    tex_hists = np.load(f)
    
    models = generate_models(labellist, tex_hists)
    
    return models

# <codecell>

def show_labelmap(lm, im):
    
    hc_colors = np.loadtxt('../visualization/high_contrast_colors.txt', skiprows=1)

    labelmap_rgb = label2rgb(lm.astype(np.int), image=im, colors=hc_colors[1:]/255., alpha=0.1, 
                             image_alpha=1, bg_color=hc_colors[0]/255.)

    labelmap_rgb = utilities.regulate_img(labelmap_rgb)
    plt.imshow(labelmap_rgb)
    plt.show()    

# <codecell>

models = models_from_labeling(args.labeling_fn)
n_models = len(models)

# <codecell>


# <codecell>

texton_models = [model['texton_hist'] for model in models]

mask = np.load(full_object_name('cropMask','npy'))
fg_superpixels = np.load(full_object_name('fg','npy'))
bg_superpixels = np.load(full_object_name('bg','npy'))
neighbors = np.load(full_object_name('neighbors','npy'))

sp_texton_hist_normalized = np.load(full_object_name('texHist', 'npy'))

segmentation = np.load

segmentation = np.load(full_object_name('segmentation', 'npy'))
n_superpixels = len(np.unique(segmentation))

D_texton_model = -1*np.ones((n_models, n_superpixels))
D_texton_model[:, fg_superpixels] = cdist(sp_texton_hist_normalized[fg_superpixels], texton_models, chi2).T

textonmap = np.load(full_object_name('texMap', 'npy'))
overall_texton_hist = np.bincount(textonmap[mask].flat)

overall_texton_hist_normalized = overall_texton_hist.astype(np.float) / overall_texton_hist.sum()

D_texton_null = np.squeeze(cdist(sp_texton_hist_normalized, [overall_texton_hist_normalized], chi2))

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
    

def grow_cluster_likelihood_ratio(seed, texton_model, debug=False, lr_grow_thresh = 0.1):
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
            
            ratio_v = D_texton_null[v] - chi2(p[v], texton_model)
            if debug:  
                print 'u=', u, 'v=',v, 'ratio_v = ', ratio_v
                print D_texton_null[v],  chi2(p[v], texton_model)
            
            if ratio_v > lr_grow_thresh:
                curr_cluster.add(v)
                frontier.append(v)
                                
    return curr_cluster, lr_grow_thresh

def grow_cluster_likelihood_ratio_precomputed(seed, D_texton_model, debug=False, lr_grow_thresh = 0.1):
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
            
            ratio_v = D_texton_null[v] - D_texton_model[v]
            if debug:  
                print 'u=', u, 'v=',v, 'ratio_v = ', ratio_v
                print D_texton_null[v],  D_texton_model[v], \
            
            if ratio_v > lr_grow_thresh:
                curr_cluster.add(v)
                frontier.append(v)
                                
    return curr_cluster, lr_grow_thresh

# <codecell>

# lr_decision_thresh = param['lr_decision_thresh']
# lr_grow_thresh = param['lr_grow_thresh']

lr_grow_thresh = .01
lr_decision_thresh = .04

def find_best_model(i):
    model_score = np.empty((n_models, ))

    if i in bg_superpixels:
        return -1
    else:
        for m in range(n_models):
            matched, _ = grow_cluster_likelihood_ratio_precomputed(i, D_texton_model[m], lr_grow_thresh=lr_grow_thresh)         
            matched = list(matched)
            model_score[m] = np.mean(D_texton_null[matched] - D_texton_model[m, matched])

#             print matched, model_score[m]
            
        best_sig = model_score.max()
        if best_sig > lr_decision_thresh: # sp whose sig is smaller than this is assigned null
          return model_score.argmax()    
    return -1


def assign_models():

    print lr_decision_thresh, lr_grow_thresh

    r = Parallel(n_jobs=16)(delayed(find_best_model)(i) for i in range(n_superpixels))
    labels = np.array(r, dtype=np.int)
    
    return labels
    
    
# find_best_model(801)
# find_best_model(1360)
# find_best_model(1181)
# find_best_model(1435)

assigned_models = assign_models()

# <codecell>


labelmap = assigned_models[segmentation]

hc_colors = np.loadtxt('../visualization/high_contrast_colors.txt', skiprows=1)

labelmap_rgb = label2rgb(labelmap.astype(np.int), image=cropImg, colors=hc_colors[1:]/255., alpha=0.1, 
                         image_alpha=1, bg_color=hc_colors[0]/255.)

# import datetime
# dt = datetime.datetime.now().strftime("%m%d%Y%H%M%S")

# new_labeling = {
# 'username': 'sigboost',
# 'parent_labeling_name': None,
# 'login_time': dt,
# 'logout_time': dt,
# 'init_labellist': None,
# 'final_labellist': labels,
# 'labelnames': None,
# 'history': None
# }

labelmap_rgb = utilities.regulate_img(labelmap_rgb)
plt.imshow(labelmap_rgb)
plt.show()

new_preview_fn = os.path.join('/home/yuncong/BrainLocal/sigboost_outputs', image_name + '_sigboost.tif')
cv2.imwrite(new_preview_fn, labelmap_rgb)

