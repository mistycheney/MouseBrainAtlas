# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

from preamble import *

# <codecell>

texton_hists = dm.load_pipeline_result('texHist', 'npy')

segmentation = dm.load_pipeline_result('segmentation', 'npy')
n_superpixels = len(unique(segmentation)) - 1

textonmap = dm.load_pipeline_result('texMap', 'npy')
n_texton = len(np.unique(textonmap)) - 1

neighbors = dm.load_pipeline_result('neighbors', 'npy')

sp_properties = dm.load_pipeline_result('spProps', 'npy')

segmentation_vis = dm.load_pipeline_result('segmentationWithText', 'jpg')

# <codecell>

from scipy.spatial.distance import cdist, pdist, squareform

center_dists = pdist(sp_properties[:, :2])
center_dist_matrix = squareform(center_dists)

k = 200
k_neighbors = np.argsort(center_dist_matrix, axis=1)[:, 1:k+1]

neighbor_dists = np.empty((n_superpixels, k))
for i in range(n_superpixels):
#     neighbor_dists[i] = np.squeeze(cdist(texton_hists[i][np.newaxis,:], texton_hists[k_neighbors[i]], chi2))
    neighbor_dists[i] = np.squeeze(cdist(texton_hists[i][np.newaxis,:], texton_hists[k_neighbors[i]], js))
    
sp_sp_dists = np.nan * np.ones((n_superpixels, n_superpixels))
for i in range(n_superpixels):
    sp_sp_dists[i, k_neighbors[i]] = neighbor_dists[i]

# <codecell>

def compute_cluster_score(cluster, texton_hists=texton_hists, neighbors=neighbors):
    
    cluster_list = list(cluster)
    hists_cluster = texton_hists[cluster_list]

    cluster_avg = hists_cluster.mean(axis=0)
    
    surrounds = set([i for i in set.union(*[neighbors[c] for c in cluster]) if i not in cluster and i != -1])
    
    surrounds_list = list(surrounds)
    
    hists_surround = texton_hists[surrounds_list]

#     avg_dists = np.atleast_1d(np.squeeze(cdist(np.atleast_2d(cluster_avg), hists_cluster, chi2)))
    
    avg_dists = np.atleast_1d(np.squeeze(cdist(np.atleast_2d(cluster_avg), hists_cluster, js)))
            
    surround_dists = np.empty((len(cluster_list), ))
    closest_neighboring_surround_sps = np.empty((len(cluster_list), ), dtype=np.int)
    for ind, i in enumerate(cluster_list):
        neighboring_surround_sps = list(neighbors[i] & surrounds)
        if len(neighboring_surround_sps) > 0:
            ds = sp_sp_dists[i, neighboring_surround_sps]
            surround_dists[ind] = np.min(ds)
            closest_neighboring_surround_sps[ind] = neighboring_surround_sps[np.argmin(ds)]
        else:
            surround_dists[ind] = np.nan
            closest_neighboring_surround_sps[ind] = -1
        
#     surround_dists = np.array([sp_sp_dists[i, cluster_list] for i in surrounds_list])
    
    avg_dist = np.mean(avg_dists) / np.sqrt(len(cluster_list))
#     avg_dist = np.max(avg_dists)
#     surround_dist = np.min(np.atleast_2d(surround_dists), axis=0).mean()
#     surround_dist = np.min(np.atleast_2d(surround_dists), axis=0).min()
    surround_dist = np.nanmin(surround_dists)
#     surround_dist = np.nanmean(surround_dists)

    score = (1 + 10 * np.log(len(cluster_list))) * (surround_dist - avg_dist)
#     if len(cluster_list) == 1:
#         score = 0
#     score = (1 + np.log(len(cluster_list) + 1)) * (surround_dist - .01 * avg_dist)
#     score = (1 + .01 * len(cluster_list)) * (surround_dist - .01 * avg_dist)
#     score = (1 + .03 * np.log(10 * len(cluster_list) + 1)) * (surround_dist - .15 * avg_dist)
#     score = (1 + .01 * np.log(10 * len(cluster_list) + 1)) * (surround_dist - .8 * avg_dist)

    
#     print 'cluster', cluster_list
# #     print 'surrounds_list', surrounds_list
# #     argmins = np.array(surrounds_list)[np.argmin(np.atleast_2d(surround_dists), axis=0)]
#     mins = np.min(np.atleast_2d(surround_dists), axis=0)
#     adv = mins - avg_dists
#     print 'surrounds'
#     for t in zip(cluster_list, closest_neighboring_surround_sps, mins, avg_dists, adv):
#         print t
    
#     print 'sig:', score, ', surround:', surround_dist, ', model:', avg_dist
#     print 
    
    return score, surround_dist, avg_dist

# <codecell>

def visualize_cluster(cluster, segmentation=segmentation, segmentation_vis=segmentation_vis):

    a = -1*np.ones_like(segmentation)
    
    for c in cluster:
        a[segmentation == c] = 0
        
    vis = label2rgb(a, image=segmentation_vis)

    vis = img_as_ubyte(vis[...,::-1])

    for i, sp in enumerate(cluster):
        vis = cv2.putText(vis, str(i), 
                          tuple(np.floor(sp_properties[sp, [1,0]] - np.array([10,-10])).astype(np.int)), 
                          cv2.FONT_HERSHEY_DUPLEX,
                          1., ((0,255,255)), 1)

    return vis.copy()

def visualize_multiple_clusters(clusters, segmentation=segmentation, segmentation_vis=segmentation_vis):

    n = len(clusters)
    m = -1*np.ones((n_superpixels,), dtype=np.int)
    
    for ci, c in enumerate(clusters):
        m[list(c)] = ci
        
    a = m[segmentation]
    a[~dm.mask] = -1
    
#     a = -1*np.ones_like(segmentation)
#     for ci, c in enumerate(clusters):
#         for i in c:
#             a[segmentation == i] = ci

    vis = label2rgb(a, image=segmentation_vis)

    vis = img_as_ubyte(vis[...,::-1])

    for ci, c in enumerate(clusters):
        for i, sp in enumerate(c):
            vis = cv2.putText(vis, str(i), 
                              tuple(np.floor(sp_properties[sp, [1,0]] - np.array([10,-10])).astype(np.int)), 
                              cv2.FONT_HERSHEY_DUPLEX,
                              1., ((0,255,255)), 1)
    
    return vis.copy()

# <codecell>

# def compare_two_sps(a,b):

#     print sp_sp_dists[a,b]
    
#     js(texton_hists[a], texton_hists[b])

#     plt.bar(np.arange(n_texton), texton_hists[a], width=.3, color='g')
#     plt.bar(np.arange(n_texton)+.3, texton_hists[b], width=.3, color='b')
#     plt.show()

# <codecell>

import networkx
from networkx.algorithms import node_connected_component

neighbors_dict = dict(zip(np.arange(n_superpixels), [list(i) for i in neighbors]))
neighbor_graph = networkx.from_dict_of_lists(neighbors_dict)

# <codecell>

def grow_cluster(seed):

#     seed = 3767
    # null_seed = 3066

    all_distances = np.atleast_1d(np.squeeze(cdist(texton_hists[seed][np.newaxis, :], texton_hists, js)))
    # null_distances = np.atleast_1d(np.squeeze(cdist(texton_hists[null_seed][np.newaxis, :], texton_hists, js)))

    ss = []
    cc = []

    # for i in np.arange(.01,null_distances[seed],.01):

    for i in np.arange(.01, 0.3, .01):

    #     similar_nodes = np.where(all_distances < .1)[0]
        similar_nodes = np.where(all_distances < i)[0]
    #     similar_nodes = np.where(all_distances < null_distances - i)[0]
        similar_graph = neighbor_graph.subgraph(similar_nodes)

        curr_cluster = node_connected_component(similar_graph, seed)

        if len(curr_cluster) > int(n_superpixels * 0.03):
            break
            
        cc.append(curr_cluster)

        s, _, _ = compute_cluster_score(curr_cluster)
        ss.append(s)

    curr_cluster = cc[np.argmax(ss)]
    score = np.max(ss)
    
    return curr_cluster, score

# <codecell>

# a, s = grow_cluster(1584)
# print s
# vis = visualize_cluster(a)
# display(vis)

# <codecell>

# plt.plot(range(len(ss)), ss);

# <codecell>

# seed = 3395

# all_distances = np.atleast_1d(np.squeeze(cdist(texton_hists[seed][np.newaxis, :], texton_hists, js)))
# vis = all_distances[segmentation] < 0.02
# vis[~dm.mask] = 0

# plt.matshow(vis)
# plt.colorbar()
# plt.show()

# <codecell>

# try:
#     clusters = dm.load_pipeline_result('clusters', 'pkl')
    
# except Exception as e:

import time
b = time.time()

clusters = Parallel(n_jobs=16)(delayed(grow_cluster)(s) for s in range(n_superpixels))

print time.time() - b

# dm.save_pipeline_result(clusters, 'clusters', 'pkl')

# <codecell>

dm.save_pipeline_result(clusters, 'clusters', 'pkl')

# <codecell>

# clusters = dm.load_pipeline_result('clusters', 'pkl')

# <codecell>

cluster_sps, cluster_score_sps = zip(*clusters)
cluster_size_sps = np.array([len(c) for c in cluster_sps])
cluster_score_sps = np.array(cluster_score_sps)

# <codecell>

# sigmap = cluster_score_sps[segmentation] * 5 + 1
# sigmap[~dm.mask] = 0

# plt.matshow(sigmap)
# plt.colorbar()
# plt.show()

# <codecell>

highlighted_sps = np.where((cluster_size_sps < 200) & (cluster_size_sps > 2))[0]
n_highlights = len(highlighted_sps)
print n_highlights

# <codecell>

from scipy.spatial.distance import pdist, squareform

overlap_matrix = np.zeros((n_highlights, n_highlights))
for i in range(n_highlights):
    for j in range(n_highlights):
        if i == j:
            overlap_matrix[i, j] = 1
        else:
            c1 = set(cluster_sps[highlighted_sps[i]])
            c2 = set(cluster_sps[highlighted_sps[j]])
#             overlap_matrix[i, j] = float(len(c1 & c2))/min(len(c1),len(c2))
            overlap_matrix[i, j] = float(len(c1 & c2))/len(c1 | c2)
          
distance_matrix = 1 - overlap_matrix
            
plt.matshow(overlap_matrix, cmap=cm.coolwarm)
plt.colorbar()
plt.show()

# <codecell>

from scipy.cluster.hierarchy import average, fcluster, leaders, complete, single, dendrogram

lk = average(squareform(distance_matrix))

# <codecell>

# T = fcluster(lk, 1.15, criterion='inconsistent')
T = fcluster(lk, 0.01, criterion='distance')

n_groups = len(set(T))
print n_groups
groups = [None] * n_groups

for group_id in range(n_groups):
    groups[group_id] = highlighted_sps[where(T == group_id)[0]]
    
groups = [g for g in groups if len(g) > 1]

# <codecell>

def compute_cluster_score2(cluster, texton_hists=texton_hists, neighbors=neighbors):
    
    cluster_list = list(cluster)
    hists_cluster = texton_hists[cluster_list]

    cluster_avg = hists_cluster.mean(axis=0)
    
    surrounds = set([i for i in set.union(*[neighbors[c] for c in cluster]) if i not in cluster and i != -1])
    
    surrounds_list = list(surrounds)
    
#     interior_holes = set([s for s in surrounds_list if neighbors[s] <= set(cluster)])
    
    hists_surround = texton_hists[surrounds_list]

#     avg_dists = np.atleast_1d(np.squeeze(cdist(np.atleast_2d(cluster_avg), hists_cluster, chi2)))
    
    avg_dists = np.atleast_1d(np.squeeze(cdist(np.atleast_2d(cluster_avg), hists_cluster, js)))
            
    surround_dists = np.empty((len(cluster_list), ))
    closest_neighboring_surround_sps = np.empty((len(cluster_list), ), dtype=np.int)
    for ind, i in enumerate(cluster_list):
#         neighboring_surround_sps = list((neighbors[i] & surrounds) - interior_holes)      
        neighboring_surround_sps = list(neighbors[i] & surrounds)
        if (len(neighboring_surround_sps) > 0):
            ds = sp_sp_dists[i, neighboring_surround_sps]
            surround_dists[ind] = np.min(ds)
            closest_neighboring_surround_sps[ind] = neighboring_surround_sps[np.argmin(ds)]
        else:
            surround_dists[ind] = np.nan
            closest_neighboring_surround_sps[ind] = -1
        
#     surround_dists = np.array([sp_sp_dists[i, cluster_list] for i in surrounds_list])
    
    avg_dist = np.mean(avg_dists) / np.sqrt(len(cluster_list))
#     avg_dist = np.max(avg_dists)
#     surround_dist = np.min(np.atleast_2d(surround_dists), axis=0).mean()
#     surround_dist = np.min(np.atleast_2d(surround_dists), axis=0).min()
    surround_dist = np.nanmean(surround_dists)
#     surround_dist = np.nanmean(surround_dists)

    score = surround_dist
#     if len(cluster_list) == 1:
#         score = 0
#     score = (1 + np.log(len(cluster_list) + 1)) * (surround_dist - .01 * avg_dist)
#     score = (1 + .01 * len(cluster_list)) * (surround_dist - .01 * avg_dist)
#     score = (1 + .03 * np.log(10 * len(cluster_list) + 1)) * (surround_dist - .15 * avg_dist)
#     score = (1 + .01 * np.log(10 * len(cluster_list) + 1)) * (surround_dist - .8 * avg_dist)
    
#     print 'cluster', cluster_list
# #     print 'surrounds_list', surrounds_list
# #     argmins = np.array(surrounds_list)[np.argmin(np.atleast_2d(surround_dists), axis=0)]
#     mins = np.min(np.atleast_2d(surround_dists), axis=0)
#     adv = mins - avg_dists
#     print 'surrounds'
#     for t in zip(cluster_list, closest_neighboring_surround_sps, mins, avg_dists, adv):
#         print t
    
#     print 'sig:', score, ', surround:', surround_dist, ', model:', avg_dist
#     print 
    
    return score, surround_dist, avg_dist

# <codecell>

# filtered_groups = [set.union(*[set(cluster_sps[i]) for i in g]) for g in groups]

filtered_groups = [cluster_sps[g[np.argmax(cluster_score_sps[g])]] for g in groups]

filtered_groups = [g for g in filtered_groups if len(g) > 2]

# filtered_groups = [g for g in groups if len(g) > 2]

filtered_group_scores = [compute_cluster_score2(g)[0] for g in filtered_groups]

arg_score_sorted = np.argsort(filtered_group_scores)[::-1]

res = zip(filtered_groups, filtered_group_scores)
filtered_groups_res = [res[i] for i in arg_score_sorted]

# <codecell>

dm.save_pipeline_result(filtered_groups_res, 'groups', 'pkl')

# <codecell>

# for g, s in filtered_groups_res:
#     print g,s

# <codecell>

# vis = visualize_multiple_clusters([filtered_groups[i] for i in arg_score_sorted[:100]])
# display(vis)

# <codecell>


