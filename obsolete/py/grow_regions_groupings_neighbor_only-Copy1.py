# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%reload_ext autoreload
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

    score = surround_dist

#     score = (1 + 10 * np.log(len(cluster_list))) * (surround_dist - avg_dist)
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

def grow_cluster(seed, output=False):

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

        if len(curr_cluster) > int(n_superpixels * 0.05):
            break
            
        cc.append(curr_cluster)

        s, _, _ = compute_cluster_score(curr_cluster)
        ss.append(s)
        
        if output:
            print i, curr_cluster, s

    curr_cluster = cc[np.argmax(ss)]
    score = np.max(ss)
    
    return curr_cluster, score

# <codecell>

cluster, s = grow_cluster(3640)
print 'cluster', cluster
vis = visualize_cluster(cluster)
display(vis)

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
# cluster_bounding_boxes = Parallel(n_jobs=16)(delayed(compute_bounding_box)(c) for c in cluster_sps)

# <codecell>

# sigmap = cluster_score_sps[segmentation] * 5 + 1
# sigmap[~dm.mask] = 0

# plt.matshow(sigmap)
# plt.colorbar()
# plt.show()

# <codecell>

plt.hist(cluster_size_sps);

# <codecell>

highlighted_sps = np.where((cluster_size_sps < 200) & (cluster_size_sps > 4))[0]
n_highlights = len(highlighted_sps)
print n_highlights
highlighted_clusters = [cluster_sps[s] for s in highlighted_sps]
highlighted_scores = [cluster_score_sps[s] for s in highlighted_sps]

# <codecell>

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import average, fcluster, leaders, complete, single, dendrogram

def group_clusters(clusters, dist_thresh = 0.1):

    n_clusters = len(clusters)
    
    overlap_matrix = np.zeros((n_clusters, n_clusters))
    
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i == j:
                overlap_matrix[i, j] = 1
            else:
                c1 = set(clusters[i])
                c2 = set(clusters[j])
                overlap_matrix[i, j] = float(len(c1 & c2))/min(len(c1),len(c2))

    distance_matrix = 1 - overlap_matrix
    
    lk = average(squareform(distance_matrix))

    # T = fcluster(lk, 1.15, criterion='inconsistent')
    T = fcluster(lk, dist_thresh, criterion='distance')

    n_groups = len(set(T))
    print n_groups, 'groups'
    
    groups = [None] * n_groups

    for group_id in range(n_groups):
        groups[group_id] = where(T == group_id)[0]
    
        
    return groups

# <codecell>

highlighted_groups = group_clusters(highlighted_clusters)

# <codecell>

def view_overlapping_seeds(seed):
    h = where(highlighted_sps==seed)[0]
    q = where(distance_matrix[h] <= 0.1)[1]
    vis = visualize_cluster(highlighted_sps[q])
    display(vis)

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

    surround_dist = np.nanmin(surround_dists)

    score = surround_dist
#     if len(cluster_list) == 1:
#         score = 0
#     score = (1 + np.log(len(cluster_list) + 1)) * (surround_dist - .01 * avg_dist)
#     score = (1 + .01 * len(cluster_list)) * (surround_dist - .01 * avg_dist)
#     score = (1 + .03 * np.log(10 * len(cluster_list) + 1)) * (surround_dist - .15 * avg_dist)
#     score = (1 + .01 * np.log(10 * len(cluster_list) + 1)) * (surround_dist - .8 * avg_dist)
    
    print 'cluster', cluster_list
#     print 'surrounds_list', surrounds_list
#     argmins = np.array(surrounds_list)[np.argmin(np.atleast_2d(surround_dists), axis=0)]
    mins = np.min(np.atleast_2d(surround_dists), axis=0)
    adv = mins - avg_dists
    print 'surrounds'
    for t in zip(cluster_list, closest_neighboring_surround_sps, mins, avg_dists, adv):
        print t
    
    print 'sig:', score, ', surround:', surround_dist, ', model:', avg_dist
    print 
    
    return score, surround_dist, avg_dist

# <codecell>

grow_cluster(3518)

# <codecell>

sp_groups = [ [highlighted_sps[i] for i in group] for group in highlighted_groups if len(group) > 1]
union_clusters = [cluster_sps[g[np.argmax(cluster_score_sps[g])]] for g in sp_groups]

# <codecell>

union_cluster_groups = group_clusters(union_clusters, dist_thresh=0.5)
union_cluster_groups = [u for u in union_cluster_groups if len(u) > 0]

# <codecell>

union_cluster_groups[2]

# <codecell>

# union_cluster_union_clusters = [set.union(*[set(union_clusters[i]) for i in g]) for g in union_cluster_groups]
# union_cluster_union_clusters = [set.intersection(*[set(union_clusters[i]) for i in g]) for g in union_cluster_groups]
union_cluster_union_clusters = [union_clusters[g[np.argmax([compute_cluster_score(union_clusters[i])[0] for i in g])]] 
                                for g in union_cluster_groups]

# <codecell>

where([3395 in g for g in sp_groups])

# <codecell>

union_cluster_union_clusters[87]

# <codecell>

where([2085 in g for g in union_cluster_union_clusters])

# <codecell>

filtered_group_scores = np.array([compute_cluster_score(g)[0] for g in union_cluster_union_clusters])
# filtered_group_scores = [cluster_score_sps[list(g)].max() for g in filtered_groups]

arg_score_sorted = np.argsort(filtered_group_scores)[::-1]

union_cluster_union_clusters_sorted = [union_cluster_union_clusters[i] for i in arg_score_sorted]

print len(union_cluster_union_clusters_sorted)

# <codecell>

seeds = where([2085 in g for g in union_clusters])[0]
for seed in seeds:
    print seed, groups[seed], union_clusters[seed]

# <codecell>

orig_index = where(highlighted_sps==2085)[0][0]
close_highlighted_indices = where(distance_matrix[orig_index] < 1)[0]
print highlighted_sps[close_highlighted_indices]
print distance_matrix[orig_index, close_highlighted_indices]

# <codecell>

dm.save_pipeline_result(filtered_groups_res, 'groups', 'pkl')

# <codecell>

for i in where([3395 in g for g in highlighted_clusters])[0]:
    print i, highlighted_sps[i], highlighted_clusters[i]

# <codecell>

for i, (g, s) in enumerate(zip(union_cluster_union_clusters_sorted, filtered_group_scores[arg_score_sorted])):
    print i, g, s

# <codecell>

vis = visualize_multiple_clusters(union_cluster_union_clusters_sorted[:30])
display(vis)

# <codecell>

c = union_cluster_union_clusters_sorted[24]
compute_cluster_score2(c)
vis = visualize_cluster(c)
display(vis)

