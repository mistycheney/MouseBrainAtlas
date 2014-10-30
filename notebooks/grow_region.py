# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

# <codecell>

from utilities import *

if 'SSH_CONNECTION' in os.environ:
    DATA_DIR = '/home/yuncong/DavidData'
    REPO_DIR = '/home/yuncong/Brain'
else:
    DATA_DIR = '/home/yuncong/BrainLocal/DavidData_4'
    REPO_DIR = '/home/yuncong/BrainSaliencyDetection'

dm = DataManager(DATA_DIR, REPO_DIR)

# import argparse
# import sys

# parser = argparse.ArgumentParser(
# formatter_class=argparse.RawDescriptionHelpFormatter,
# description='Execute feature extraction pipeline',
# epilog="""
# The following command processes image RS141_x5_0001.tif using blueNissl for both gabor parameters and segmentation parameters.
# python %s RS141 x5 1 -g blueNissl -s blueNissl -v blueNissl
# """%(os.path.basename(sys.argv[0]), ))

# parser.add_argument("stack_name", type=str, help="stack name")
# parser.add_argument("resolution", type=str, help="resolution string")
# parser.add_argument("slice_ind", type=int, help="slice index")
# parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNissl')
# parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNissl')
# parser.add_argument("-v", "--vq_params_id", type=str, help="vq parameters id (default: %(default)s)", default='blueNissl')
# args = parser.parse_args()

class args:
    stack_name = 'RS141'
    resolution = 'x5'
    slice_ind = 1
    gabor_params_id = 'blueNisslWide'
    segm_params_id = 'blueNissl'
    vq_params_id = 'blueNissl'

# <codecell>

dm.set_image(args.stack_name, args.resolution, args.slice_ind)
dm.set_gabor_params(gabor_params_id=args.gabor_params_id)
dm.set_segmentation_params(segm_params_id=args.segm_params_id)
dm.set_vq_params(vq_params_id=args.vq_params_id)

from joblib import Parallel, delayed

n_texton = int(dm.vq_params['n_texton'])

# <codecell>

texton_hists = dm.load_pipeline_result('texHist', 'npy')

# <codecell>

cropped_segmentation = dm.load_pipeline_result('cropSegmentation', 'npy')
n_superpixels = len(unique(cropped_segmentation)) - 1
cropped_mask = dm.load_pipeline_result('cropMask', 'npy')

textonmap = dm.load_pipeline_result('texMap', 'npy')
neighbors = dm.load_pipeline_result('neighbors', 'npy')

# <codecell>

neighbors[149]

# <codecell>

plt.bar(np.arange(n_texton), texton_hists[458], width=0.8, color='b', alpha=0.5)
plt.bar(np.arange(n_texton), texton_hists[478], width=0.8, color='r', alpha=0.5)

# <codecell>

distance2null_sp = D_texton_null[cropped_segmentation].copy()
distance2null_sp[~cropped_mask] = 0
plt.matshow(distance2null_sp)
plt.colorbar()

# <codecell>

from scipy.spatial.distance import cdist

overall_texton_hist = np.bincount(textonmap[cropped_mask].flat)

overall_texton_hist_normalized = overall_texton_hist.astype(np.float) / overall_texton_hist.sum()

D_texton_null = np.squeeze(cdist(texton_hists, [overall_texton_hist_normalized], chi2))

# <codecell>

cropped_image = dm.load_pipeline_result('cropImg', 'tif')

# <codecell>

vis = paint_superpixels_on_image(cluster, cropped_segmentation, cropped_image)

# <codecell>

dm.save_pipeline_result(vis,'tmp','tif')

# <codecell>

plt.imshow(vis)

# <codecell>

# cluster = [1652,1638,1737,1716,1815,1789,1865,1890,1803]
cluster, thresh = grow_cluster_likelihood_ratio(2089, texton_hists[2089])

# <codecell>

re_thresh_min = 0.01
re_thresh_max = 0.8

# def grow_cluster_relative_entropy(seed, debug=False, 
#                                   frontier_contrast_diff_thresh = 0.1,
#                                   max_cluster_size = 100):
#     '''
#     find the connected cluster of superpixels that have similar texture, starting from a superpixel as seed
#     '''
    
    
#     prev_frontier_contrast = np.inf
#     for re_thresh in np.arange(re_thresh_min, re_thresh_max, .01):
    
#         curr_cluster = set([seed])
#         frontier = [seed]

#         while len(frontier) > 0:
#             u = frontier.pop(-1)
#             for v in neighbors[u]:
#                 if v in bg_superpixels or v in curr_cluster: 
#                     continue

#                 if chi2(p[v], p[seed]) < re_thresh:
#                     curr_cluster.add(v)
#                     frontier.append(v)
        
#         surround = set.union(*[neighbors[i] for i in curr_cluster]) - set.union(curr_cluster, bg_set)
#         if len(surround) == 0:
#             return curr_cluster, re_thresh

#         frontier_in_cluster = set.intersection(set.union(*[neighbors[i] for i in surround]), curr_cluster)
#         frontier_contrasts = [np.nanmax([chi2(p[i], p[j]) for j in neighbors[i] if j not in bg_set]) for i in frontier_in_cluster]
#         frontier_contrast = np.max(frontier_contrasts)
        
#         if debug:
#             print 'frontier_contrast=', frontier_contrast, 'prev_frontier_contrast=', prev_frontier_contrast, 'diff=', frontier_contrast - prev_frontier_contrast
        
#         if len(curr_cluster) > max_cluster_size or \
#         frontier_contrast - prev_frontier_contrast > frontier_contrast_diff_thresh:
#             return curr_cluster, re_thresh
        
#         prev_frontier_contrast = frontier_contrast
#         prev_cluster = curr_cluster
#         prev_re_thresh = re_thresh
                                
#     return curr_cluster, re_thresh
    

def grow_cluster_likelihood_ratio(seed, texton_model, lr_grow_thresh = 0.1):
    '''
    find the connected cluster of superpixels that are more likely to be explained by given model than by null, 
    starting from a superpixel as seed
    '''
    
    curr_cluster = set([seed])    
    frontier = [seed]
        
    while len(frontier) > 0:
        u = frontier.pop(-1)
        for v in neighbors[u]:
            if v == -1 or v in curr_cluster:
                continue
            
            d = chi2(texton_hists[v], texton_model)
            ratio_v = D_texton_null[v] - d
            print 'u=', u, 'v=',v, 'ratio_v = ', ratio_v
            print 'd_null=', D_texton_null[v], 'd_model=', d
            
            if ratio_v > lr_grow_thresh:
                curr_cluster.add(v)
                frontier.append(v)
                                
    return curr_cluster, lr_grow_thresh

# def grow_cluster_likelihood_ratio_precomputed(seed, D_texton_model, debug=False, lr_grow_thresh = 0.1):
#     '''
#     find the connected cluster of superpixels that are more likely to be explained by given model than by null, 
#     starting from a superpixel as seed
#     using pre-computed distances between model and superpixels
#     '''

#     if seed in bg_superpixels:
#         return [], -1

#     curr_cluster = set([seed])
#     frontier = [seed]
        
#     while len(frontier) > 0:
#         u = frontier.pop(-1)
#         for v in neighbors[u]:
#             if v in bg_superpixels or v in curr_cluster: 
#                 continue
            
#             ratio_v = D_texton_null[v] - D_texton_model[v]
#             if debug:  
#                 print 'u=', u, 'v=',v, 'ratio_v = ', ratio_v
#                 print D_texton_null[v],  D_texton_model[v], \
            
#             if ratio_v > lr_grow_thresh:
#                 curr_cluster.add(v)
#                 frontier.append(v)
                                
#     return curr_cluster, lr_grow_thresh

