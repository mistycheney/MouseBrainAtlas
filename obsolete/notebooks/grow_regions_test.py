# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from IPython.external import mathjax; mathjax.install_mathjax()

# <codecell>

%load_ext autoreload
%autoreload 2

# <codecell>

from utilities import *

if 'SSH_CONNECTION' in os.environ:
    DATA_DIR = '/home/yuncong/DavidData'
    REPO_DIR = '/home/yuncong/Brain'
else:
    DATA_DIR = '/home/yuncong/BrainLocal/DavidData_v4'
    REPO_DIR = '/home/yuncong/Brain'

dm = DataManager(DATA_DIR, REPO_DIR)

class args:
    stack_name = 'RS141'
    resolution = 'x5'
    slice_ind = 1
    gabor_params_id = 'blueNisslWide'
    segm_params_id = 'blueNissl'
    vq_params_id = 'blueNissl'

dm.set_image(args.stack_name, args.resolution, args.slice_ind)
dm.set_gabor_params(gabor_params_id=args.gabor_params_id)
dm.set_segmentation_params(segm_params_id=args.segm_params_id)
dm.set_vq_params(vq_params_id=args.vq_params_id)

from joblib import Parallel, delayed

n_texton = int(dm.vq_params['n_texton'])

texton_hists = dm.load_pipeline_result('texHist', 'npy')

cropped_segmentation = dm.load_pipeline_result('cropSegmentation', 'npy')
n_superpixels = len(np.unique(cropped_segmentation)) - 1
cropped_mask = dm.load_pipeline_result('cropMask', 'npy')

textonmap = dm.load_pipeline_result('texMap', 'npy')
neighbors = dm.load_pipeline_result('neighbors', 'npy')

cropped_image = dm.load_pipeline_result('cropImg', 'tif')

# <codecell>

sp_props = dm.load_pipeline_result('cropSpProps', 'npy')

# <codecell>

from scipy.spatial.distance import cdist

overall_texton_hist = np.bincount(textonmap[cropped_mask].flat)

overall_texton_hist_normalized = overall_texton_hist.astype(np.float) / overall_texton_hist.sum()

D_sp_null = np.squeeze(cdist(texton_hists, [overall_texton_hist_normalized], chi2))

# distance2null_map = D_sp_null[cropped_segmentation].copy()
# distance2null_map[~cropped_mask] = 0
# plt.matshow(distance2null_map)
# plt.colorbar()

# <codecell>

def compute_cluster_score(cluster, surround, texton_hists, D_sp_null):
    
    avg = texton_hists[list(cluster)].mean(axis=0)
    avg_sp_distances = np.squeeze(cdist([model], texton_hists[list(cluster)], chi2))
    avg_sp_total_distance = np.sum(D_model_sp)
    
    null_sp_total_distance = np.sum(D_null_sp[list(cluster)])
    
    avg_surround_distances = np.squeeze(cdist([avg], texton_hists[list(surround)], chi2))
    avg_surround_distance = avg_surround_distances.mean()
    
    print avg_sp_total_distance, null_sp_total_distance, avg_surround_distance
    

# <codecell>

# def compute_cluster_score(cluster, texton_hists, D_sp_null):
#     model = texton_hists[list(cluster)].mean(axis=0)
#     D_sp_model = np.squeeze(cdist([model], texton_hists[list(cluster)], chi2))
#     model_sum = np.sum(D_sp_model)
#     null_sum = np.sum(D_sp_null[list(cluster)])
#     # can be made weighted by superpixel size
    
#     score = null_sum - model_sum
#     return score, null_sum, model_sum

# <codecell>

# def grow_cluster(seed, neighbors, texton_hists, D_sp_null, model_fit_reduce_limit=.5):
    
#     curr_cluster = set([seed])
#     frontier = [seed]
    
#     curr_cluster_score, _, curr_model_score = compute_cluster_score(curr_cluster, texton_hists, D_sp_null)
    
#     while len(frontier) > 0:
#         u = frontier.pop(-1)
#         for v in neighbors[u]:
#             if v == -1 or v in curr_cluster: 
#                 continue

#             score_new, _, model_sum_new = compute_cluster_score(curr_cluster | set([v]), texton_hists, D_sp_null)
            
#             if score_new > curr_cluster_score and model_sum_new - curr_model_score < model_fit_reduce_limit :
#                 curr_cluster.add(v)
#                 frontier.append(v)
#                 curr_cluster_score, _, curr_model_score = compute_cluster_score(curr_cluster, texton_hists, D_sp_null)
            
#             if len(curr_cluster) > 50:
#                 return curr_cluster
            
#     return curr_cluster

# <codecell>

# seed = 2046
# seed = 1950
# seed = 387
# seed = 378
# seed = 284
# seed = 1737
# seed = 1705
# seed = 1951
# seed = 905
seed = 399
    
curr_cluster = set([seed])
frontier = [seed]
surrounds = set([])
curr_frontier = set([])

model_fit_reduce_limit = .5

scores = []
model_distances = []
null_distances = []

interior = set([])

# curr_cluster_score, curr_null_distance, curr_model_distance = compute_cluster_score(curr_cluster, texton_hists, D_sp_null)


from numpy import random

# while len(frontier) > 0:
        


#     print new_frontier
    
    
c = 0

to_test = set([seed])

while len(to_test) > 0:

#     for u in curr_cluster:
#         if neighbors[u] <= curr_cluster:
#             interior.add(u)

#     frontier = curr_cluster - interior
    
    print 'curr_cluster', curr_cluster

#     print 'frontier', frontier

    to_test = set.union(*[neighbors[v] for v in curr_cluster]) - curr_cluster

    print 'to_test', to_test
    
    u = list(to_test)[random.randint(len(to_test))]
    
    if
        curr_cluster.add(u)
        
        compute_cluster_score(cluster, to_test, texton_hists, D_sp_null)
    
    c += 1
    
    if c > 60:
        break
        

#     while len(frontier) > 0
        
#         if v == -1 or v in curr_cluster: 
#             continue
            
#         score_new, _, model_sum_new = compute_cluster_score(curr_cluster | set([v]), texton_hists, D_sp_null)

# #         if score_new > curr_cluster_score and model_sum_new - curr_model_score < model_fit_reduce_limit :
#         if len(curr_cluster) < 20:
#             curr_cluster.add(v)
#             frontier.append(v)
#             curr_cluster_score, curr_null_distance, curr_model_distance = compute_cluster_score(curr_cluster, texton_hists, D_sp_null)
            
#             scores.append(curr_cluster_score)
#             null_distances.append(curr_null_distance)
#             model_distances.append(curr_model_distance)
                    
#             print curr_cluster

# <codecell>

a = -1*np.ones_like(cropped_segmentation)

for c in curr_cluster:
    a[cropped_segmentation == c] = 0

# for c in frontier:
#     a[cropped_segmentation == c] = 1

for c in to_test:
    a[cropped_segmentation == c] = 2
    
plt.imshow(label2rgb(a))

# <codecell>

import matplotlib.pyplot as plt

plt.plot(scores, color='g', label='score (distance diff)')
plt.plot(model_distances, color='r', label='model distance')
plt.plot(null_distances, color='b', label='null distance')
plt.legend()
plt.show()

# <codecell>

print cluster
print compute_cluster_score(cluster)

# <codecell>

print compute_cluster_score(set([2081, 2089, 2093, 2036, 2037, 2075]))
print compute_cluster_score(set([2081, 2089, 2093, 2036, 2037, 2075, 2080]))

# <codecell>

print compute_cluster_score(cluster)
print compute_cluster_score(cluster - set([1847, 1882, 1929, 1896]))

# <codecell>

cluster, re_thresh = grow_cluster_relative_entropy(2046)
print cluster

# <codecell>

vis = paint_superpixels_on_image(cluster, cropped_segmentation, cropped_image)
plt.imshow(vis)

# <codecell>

dm.save_pipeline_result(vis,'tmp','tif')

# <codecell>

for c in cluster:
    plt.bar(np.arange(n_texton), texton_hists[c], width=0.8, color='b', alpha=0.5)
    plt.title('%d'%c)
    plt.show()

# <codecell>

plt.bar(np.arange(n_texton), overall_texton_hist_normalized, width=0.8, color='b', alpha=0.5)
plt.show()
plt.bar(np.arange(n_texton), model, width=0.8, color='r', alpha=0.5)

# <codecell>

distance_diff_map = D_diff[cropped_segmentation].copy()
distance_diff_map[~cropped_mask] = 0
plt.matshow(distance_diff_map)
plt.colorbar()

# <codecell>

hist, bin_edges = np.histogram(D_sp_model, bins=np.arange(0,2,0.01))
plt.bar(bin_edges[:-1], hist, width=bin_edges[1]-bin_edges[0])

hist, bin_edges = np.histogram(D_sp_model[list(cluster)], bins=np.arange(0,2,0.01))
plt.bar(bin_edges[:-1], hist, width=bin_edges[1]-bin_edges[0], color='g')

plt.xlabel('D_diff')

