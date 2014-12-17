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

x = np.arange(100)
y = (1 + .1*np.log(10*x + 1))
# y = 1 + .01 * x
plt.plot(x,y)
y[6] - y[4]

# <codecell>

# def compute_cluster_score(cluster, texton_hists=texton_hists, neighbors=neighbors):
    
#     cluster_list = list(cluster)
#     hists_cluster = texton_hists[cluster_list]

#     cluster_avg = hists_cluster.mean(axis=0)
    
#     surrounds = set([i for i in set.union(*[neighbors[c] for c in cluster]) if i not in cluster and i != -1])
    
#     surrounds_list = list(surrounds)
    
#     hists_surround = texton_hists[surrounds_list]

#     avg_dists = np.atleast_1d(np.squeeze(cdist(np.atleast_2d(cluster_avg), hists_cluster, chi2)))
            
#     surround_dists = np.array([sp_sp_dists[i, cluster_list] for i in surrounds_list])
        
#     uncomputed_sr, uncomputed_cl = np.where(np.isnan(surround_dists))
    
#     for sr, cl in zip(uncomputed_sr, uncomputed_cl):
        
#         i = surrounds_list[sr]
#         j = cluster_list[cl]
#         d = chi2(texton_hists[i], texton_hists[j])
            
#         surround_dists[sr, cl] = d
        
#         sp_sp_dists[i, j] = d
#         sp_sp_dists[j, i] = d
    
#     assert not np.isnan(surround_dists).any(), 'some distances are not computed'
    
#     avg_dist = np.mean(avg_dists)
# #     avg_dist = np.max(avg_dists)
# #     surround_dist = np.min(np.atleast_2d(surround_dists), axis=0).mean()
#     surround_dist = np.min(np.atleast_2d(surround_dists), axis=0).min()

#     score = surround_dist - .01 * avg_dist
    
#     print 'cluster', cluster_list
# #     print 'surrounds_list', surrounds_list
#     argmins = np.array(surrounds_list)[np.argmin(np.atleast_2d(surround_dists), axis=0)]
#     mins = np.min(np.atleast_2d(surround_dists), axis=0)
#     adv = mins - avg_dists
#     print 'surrounds'
#     for t in zip(cluster_list, argmins, mins, avg_dists, adv):
#         print t
    
#     print 'sig:', score, ', surround:', surround_dist, ', model:', avg_dist
#     print 
    
#     return score, surround_dist, avg_dist

# <codecell>

# def compute_cluster_score(cluster, texton_hists=texton_hists, neighbors=neighbors):
    
#     cluster_list = list(cluster)
#     hists_cluster = texton_hists[cluster_list]
    
#     cluster_avg = hists_cluster.mean(axis=0)
    
#     surrounds = set([i for i in set.union(*[neighbors[c] for c in cluster]) if i not in cluster and i != -1])
    
#     surrounds_list = list(surrounds)
#     hists_surround = texton_hists[surrounds_list]
    
#     r = cdist(np.r_[hists_surround, np.atleast_2d(cluster_avg)], hists_cluster, chi2)
#     avg_dist = np.mean(r[-1, :])
#     surround_dist = np.min(np.atleast_2d(r[:-1, :]), axis=0).mean()
    
# #     r = np.squeeze(cdist(np.atleast_2d(cluster_avg), np.r_[hists_cluster, hists_surround], chi2))
# #     avg_dist = np.mean(r[:len(cluster_list)])
# #     surround_dist = np.mean(r[len(cluster_list):])
    
#     score = surround_dist - avg_dist
    
#     print 'cluster', cluster_list
#     print 'surrounds', surrounds_list
#     print 'argmin', np.array(surrounds_list)[np.argmin(np.atleast_2d(surround_dists), axis=0)]
#     print 'min', np.min(np.atleast_2d(surround_dists), axis=0)
#     print 'model', avg_dists
#     print 'adv', np.min(np.atleast_2d(surround_dists), axis=0) - avg_dists
#     print 'sig:', score, ', surround:', surround_dist, ', model:', avg_dist
#     print 
    
#     return score, surround_dist, avg_dist

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

    score = surround_dist - 2. * avg_dist
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

import heapq

def grow_cluster(seed, neighbors=neighbors, texton_hists=texton_hists):
    
    scores = []
    null_dists = []
    model_dists = []
    added_sp = []
        
    visited = set([])
    curr_cluster = set([])
                    
    to_visit = [(0, seed)]
        
    c = 0
        
    while len(to_visit) > 0:
        
        v = heapq.heappop(to_visit)[1]

        if v in curr_cluster:
            continue
        
#         print c
        c += 1
        
#         print 'add', v
        
        curr_cluster_score, curr_null_dist, curr_model_dist = compute_cluster_score(curr_cluster | set([v]), texton_hists, neighbors)
        
        curr_cluster.add(v)

        scores.append(curr_cluster_score)
        null_dists.append(curr_null_dist)
        model_dists.append(curr_model_dist)
        added_sp.append(v)
        
        curr_avg = texton_hists[list(curr_cluster)].mean(axis=0)

        q = set([s for d,s in to_visit]) | (neighbors[v] - set([-1])) | (visited - curr_cluster)
        to_visit_sps = list(q - curr_cluster)
        assert len(to_visit_sps) == len(set(to_visit_sps)) and -1 not in to_visit_sps
                
            
        curr_cluster_list = list(curr_cluster)

#         to_visit_dists = np.empty((len(to_visit_sps), ))
#         to_visit_farthest_sp = np.empty((len(to_visit_sps), ), dtype=np.int)
#         for ind, i in enumerate(to_visit_sps):
#             neighboring_interior_sps = list(neighbors[i] & curr_cluster)
#             if len(neighboring_interior_sps) > 0:
#                 ds = sp_sp_dists[i, neighboring_interior_sps]
# #                 to_visit_dists[ind] = np.max(ds)
# #                 to_visit_farthest_sp[ind] = neighboring_interior_sps[np.argmax(ds)]
#                 to_visit_dists[ind] = np.min(ds)
#                 to_visit_farthest_sp[ind] = neighboring_interior_sps[np.argmin(ds)]
#             else:
#                 to_visit_dists[ind] = np.nan
#                 to_visit_farthest_sp[ind] = -1

        ds = np.atleast_2d(np.squeeze(cdist(texton_hists[curr_cluster_list], texton_hists[to_visit_sps], js)))
        to_visit_dists = np.max(ds, axis=0) # for each to_visit sp, the most dissimilar in-cluster sp        
        to_visit_farthest_sp = np.array(curr_cluster_list, dtype=np.int)[np.argmax(ds, axis=0)]
#         to_visit_dists = np.mean(ds, axis=0)
#         to_visit_dists = np.median(ds, axis=0)
        
#         to_visit_dists = np.atleast_1d(np.squeeze(cdist(curr_avg[np.newaxis, :], texton_hists[to_visit_sps], js)))

        to_visit = zip((to_visit_dists*10000).astype(np.int), to_visit_sps)
        heapq.heapify(to_visit)
        
#         print 'to_visit_sps', to_visit_sps
#         print 'first three to_visit', [(sp, to_visit_farthest_sp[to_visit_sps==sp][0], sc/10000.) 
#                                        for sc, sp in heapq.nsmallest(20, to_visit)]
#         print 'first three to_visit', [(sp, sc/10000.) 
#                                        for sc, sp in heapq.nsmallest(20, to_visit)]
#         print 
        
        visited.add(v)
        
#         print curr_cluster

#         vis = visualize_cluster(curr_cluster)
#         plt.imshow(vis)
#         plt.show()
        
        if len(visited) > int(n_superpixels * 0.03):
            break
    
    sp_max = np.argmax(scores[1:]) + 1  # ignore the first score (the case with only one sp)
    curr_cluster = added_sp[:sp_max+1]
    curr_cluster_score = scores[sp_max]
        
    return curr_cluster, curr_cluster_score, scores, null_dists, model_dists, added_sp

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

    a = -1*np.ones_like(segmentation)
    
    for ci, c in enumerate(clusters):
        for i in c:
            a[segmentation == i] = ci

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

a = 3856
b = 3807

js(texton_hists[a], texton_hists[b])

plt.bar(np.arange(n_texton), texton_hists[a], width=.3, color='g')
plt.bar(np.arange(n_texton)+.3, texton_hists[b], width=.3, color='b')
plt.show()

# <codecell>

seed = 3650

curr_cluster, curr_cluster_score, scores, avg_surround_distances, sp_avg_distances, added_sp = grow_cluster(seed)

print 'curr_cluster', curr_cluster
print len(curr_cluster), 'superpixels'
print 'cluster score', curr_cluster_score

# <codecell>

vis = visualize_cluster(curr_cluster)
# plt.imshow(vis)
display(vis)

# <codecell>

vis = visualize_cluster(added_sp)
# plt.imshow(vis)
display(vis)

# <codecell>

all_distances = np.atleast_1d(np.squeeze(cdist(texton_hists[seed][np.newaxis, :], texton_hists, js)))
vis = all_distances[segmentation] * 10 + 1
vis[~dm.mask] = 0

plt.matshow(vis)
plt.colorbar()
plt.show()

# <codecell>

import matplotlib.pyplot as plt

e = -1

n = len(scores[:e])

# fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
# axes[0].plot(range(0,n), scores[:e], color='b', label='sig score')
# # plt.plot(range(1,n), np.diff(scores), color='y', label='change in l.l.r.')
# axes[1].plot(range(0,n), avg_surround_distances[:e], color='g', label='null distance')
# axes[2].plot(range(0,n), sp_avg_distances[:e], color='r', label='model distance')

plt.plot(range(0,n), scores[:e], color='b', label='sig score')
plt.plot(range(0,n), avg_surround_distances[:e], color='g', label='surround distance')
plt.plot(range(0,n), sp_avg_distances[:e], color='r', label='model distance')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.xlabel('index of most recently added superpixel')

plt.show()

# <codecell>

# try:
#     clusters = dm.load_pipeline_result('clusters', 'pkl')
    
# except Exception as e:

import time
b = time.time()

clusters = Parallel(n_jobs=16)(delayed(grow_cluster)(s) for s in range(n_superpixels))
# clusters = Parallel(n_jobs=16)(delayed(grow_cluster)(s, neighbors, texton_hists) for s in range(100))

print time.time() - b

# dm.save_pipeline_result(clusters, 'clusters', 'pkl')

# <codecell>

dm.save_pipeline_result(clusters, 'clusters', 'pkl')

# <codecell>

clusters = dm.load_pipeline_result('clusters', 'pkl')

# <codecell>

cluster_sps, curr_cluster_score_sps, scores_sps, nulls_sps, models_sps, added_sps = zip(*clusters)
cluster_size_sps = np.array([len(c) for c in cluster_sps])
cluster_score_sps = np.array(curr_cluster_score_sps)

# <codecell>

sigmap = cluster_score_sps[segmentation] * 5 + 1
sigmap[~dm.mask] = 0

plt.matshow(sigmap)
plt.colorbar()
plt.show()

# <codecell>

highlighted_sps = np.where((cluster_size_sps < 200) & (cluster_size_sps > 2))[0]
n_highlights = len(highlighted_sps)

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
            overlap_matrix[i, j] = float(len(c1 & c2))/min(len(c1),len(c2))
          
distance_matrix = 1 - overlap_matrix
            
plt.matshow(overlap_matrix, cmap=cm.coolwarm)
plt.colorbar()
plt.show()

# <codecell>

from scipy.cluster.hierarchy import average, fcluster, leaders, complete, single, dendrogram

lk = average(squareform(distance_matrix))
# T = fcluster(lk, 1.15, criterion='inconsistent')
T = fcluster(lk, .3, criterion='distance')

n_groups = len(set(T))
print n_groups
groups = [None] * n_groups

for group_id in range(n_groups):
    groups[group_id] = highlighted_sps[where(T == group_id)[0]]

# <codecell>

filtered_groups = [set.union(*[set(cluster_sps[i]) for i in g]) for g in groups if len(g) > 2]
filtered_group_scores = [compute_cluster_score(g, texton_hists, neighbors)[0] for g in filtered_groups]

arg_score_sorted = np.argsort(filtered_group_scores)[::-1]

res = zip(filtered_groups, filtered_group_scores)
filtered_groups_res = [res[i] for i in arg_score_sorted]

dm.save_pipeline_result(filtered_groups_res, 'groups', 'pkl')

# <codecell>

for g, s in filtered_groups_res:
    print g,s

# <codecell>

vis = visualize_multiple_clusters([filtered_groups[i] for i in arg_score_sorted[:20]])
display(vis)

# <codecell>

# import networkx as nx
# from networkx.algorithms.components import connected_components, strongly_connected_components
# from networkx.algorithms import find_cliques, cliques_containing_node

# G = nx.Graph(overlap_matrix)
# # components = [c for c in connected_components(G) if len(c) > 10]
# # components = [c for c in list(find_cliques(G)) if len(c) > 3 and len(c) < 50]
# # components = [[highlighted_sps[i] for i in c] for c in list(find_cliques(G)) if len(c) > 3 and len(c) < 50]

# print len(components)

# <codecell>

# from sklearn.cluster import AgglomerativeClustering
# clustering = AgglomerativeClustering(linkage='average', n_clusters=10)
# clustering.fit(X_red)
# plot_clustering(X_red, X, clustering.labels_, "%s linkage" % linkage)

# <codecell>

# components_ranked = sorted(components, key=lambda c: compute_cluster_score(c, texton_hists, neighbors)[0], reverse=True)

# print cliques_containing_node(G, 1751)

