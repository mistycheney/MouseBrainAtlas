# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

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
n_superpixels = len(unique(cropped_segmentation)) - 1
cropped_mask = dm.load_pipeline_result('cropMask', 'npy')

textonmap = dm.load_pipeline_result('texMap', 'npy')
neighbors = dm.load_pipeline_result('neighbors', 'npy')

cropped_image = dm.load_pipeline_result('cropImg', 'tif')
sp_properties = dm.load_pipeline_result('cropSpProps', 'npy')

cropped_segmentation_vis = dm.load_pipeline_result('cropSegmentation', 'tif')

# <codecell>

from scipy.spatial.distance import cdist

overall_texton_hist = np.bincount(textonmap[cropped_mask].flat, minlength=n_texton)
overall_texton_hist_normalized = overall_texton_hist.astype(np.float) / overall_texton_hist.sum()
D_sp_null = np.squeeze(cdist(texton_hists, [overall_texton_hist_normalized], chi2))

# <codecell>

# def compute_cluster_score(cluster, texton_hists, D_sp_null):
#     model = texton_hists[list(cluster)].mean(axis=0)
#     D_sp_model = np.squeeze(cdist([model], texton_hists[list(cluster)], chi2))
    
#     model_sum = np.sum(D_sp_model)
#     null_sum = np.sum(D_sp_null[list(cluster)])

#     # can be made weighted by superpixel size
    
#     score = null_sum - model_sum
#     return score, null_sum, model_sum


def compute_cluster_score(cluster, texton_hists, neighbors):
    cluster_avg = texton_hists[list(cluster)].mean(axis=0)
    sp_avg_dists = np.squeeze(cdist([cluster_avg], texton_hists[list(cluster)], chi2))
    
    sp_avg_dist = np.sum(sp_avg_dists)
#     null_sum = np.sum(D_sp_null[list(cluster)])
    
    surrounds = set([i for i in set.union(*[neighbors[c] for c in cluster]) if i not in cluster])
    avg_surround_dists = np.squeeze(cdist([cluster_avg], texton_hists[list(surrounds)], chi2))
    avg_surround_dist = len(cluster) * np.mean(avg_surround_dists)

    # can be made weighted by superpixel size
    
    score = avg_surround_dist - sp_avg_dist
    return score, avg_surround_dist, sp_avg_dist

# <codecell>

import heapq

# def grow_cluster(seed, neighbors, texton_hists, D_sp_null, model_fit_reduce_limit=.5, score_drop_tolerance=0.):
# def grow_cluster(seed, neighbors, texton_hists, D_sp_null, score_drop_tolerance=0.):

# def grow_cluster(seed, neighbors, texton_hists, score_drop_tolerance=0.):

def grow_cluster(seed, neighbors, texton_hists, score_drop_tolerance=0, model_fit_reduce_limit=.2):
    
    scores = []
    null_dists = []
    model_dists = []
    added_sp = []
        
    visited = set([])
    curr_cluster = set([])
                    
    to_visit = [(0, seed)]
        
    while len(to_visit) > 0:
        
        v = heapq.heappop(to_visit)[1]
#         print 'testing %d' % v
                
        score_new, null_dist_new, model_dist_new = compute_cluster_score(curr_cluster | set([v]), texton_hists, neighbors)
          
        if (v == seed) or ((v != seed) and (score_new > curr_cluster_score - score_drop_tolerance) 
                           and (model_dist_new < curr_model_dist + model_fit_reduce_limit)):
#         if v == seed or score_new > curr_cluster_score - score_drop_tolerance:

            curr_cluster.add(v)
#             print 'added %d' % v

            curr_cluster_score = score_new
            curr_null_dist = null_dist_new
            curr_model_dist = model_dist_new

            scores.append(curr_cluster_score)
            null_dists.append(curr_null_dist)
            model_dists.append(curr_model_dist)
            added_sp.append(v)

#             ns = np.array(list(neighbors[v]))
            ns = np.array(list(neighbors[v] | (visited - curr_cluster)))
    
            curr_avg = texton_hists[list(curr_cluster)].mean(axis=0)
            ds = np.atleast_1d(np.squeeze(cdist(curr_avg[np.newaxis, :], texton_hists[ns])))
            
#             ds = np.atleast_1d(np.squeeze(cdist(texton_hists[seed][np.newaxis, :], texton_hists[ns])))
            items = zip((ds*10000).astype(np.int), ns)

            for dist, sp in items:
                if sp != -1 and (dist, sp) not in to_visit and sp not in visited:
                    heapq.heappush(to_visit, (dist, sp))

        visited.add(v)
    
#         if len(visited) > int(n_superpixels * 0.03):
#         if len(curr_cluster) > int(n_superpixels * 0.03):
#         if len(curr_cluster) > 100:
#             break
        
    return curr_cluster, scores, null_dists, model_dists, added_sp

# <codecell>

# def visualize_cluster(cluster, segmentation, segmentation_vis, sp_properties):
def visualize_cluster(cluster, segmentation, segmentation_vis):

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


def visualize_multiple_clusters(clusters, segmentation, segmentation_vis):

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

seed = 905

# curr_cluster, scores, null_distances, model_distances, added_sp = grow_cluster(seed, neighbors, texton_hists, D_sp_null,
#                                                                      model_fit_reduce_limit = .0,
#                                                                      score_drop_tolerance = 0.0)

# curr_cluster, scores, null_distances, model_distances , added_sp = grow_cluster(seed, neighbors, texton_hists, D_sp_null,
#                                                                                score_drop_tolerance = 0.0)

curr_cluster, scores, avg_surround_distances, sp_avg_distances, added_sp = grow_cluster(seed, neighbors, texton_hists,
                                                                                        score_drop_tolerance = 0,
                                                                                        model_fit_reduce_limit = .5)

print len(scores)
print 'average model significance', scores[-1]/len(scores)

# <codecell>

all_distances = np.atleast_1d(np.squeeze(cdist(texton_hists[seed][np.newaxis, :], texton_hists)))
vis = all_distances[cropped_segmentation] * 10 + 1
vis[~cropped_mask] = 0

plt.matshow(vis)
plt.colorbar()
plt.show()

# <codecell>

vis = visualize_cluster(added_sp, cropped_segmentation, cropped_segmentation_vis)

# plt.imshow(vis)

cv2.imwrite('tmp.jpg', vis) 

from IPython.display import FileLink
FileLink('tmp.jpg')

# <codecell>

import matplotlib.pyplot as plt

n = len(scores)

plt.plot(range(0,n), scores, color='g', label='log likelihood ratio (large = cluster significant)')

plt.plot(range(1,n), np.diff(scores), color='y', label='change in l.l.r.')

 
plt.plot(range(0,n), sp_avg_distances, color='r', label='model neg. log likelihood (small = good model fit)')
plt.plot(range(0,n), avg_surround_distances, color='b', label='null neg. log likelihood (small = similar to null)')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.xlabel('index of most recently added superpixel')

plt.show()

# <codecell>

import time

b = time.time()

clusters = Parallel(n_jobs=16)(delayed(grow_cluster)(s, neighbors, texton_hists) for s in range(n_superpixels))

print time.time() - b

dm.save_pipeline_result(clusters, 'clusters', 'pkl')

# <codecell>

clusters = dm.load_pipeline_result('clusters', 'pkl')

# <codecell>

cluster_sps, scores_sps, nulls_sps, models_sps, added_sps = zip(*clusters)

cluster_size_sps = np.array([len(c) for c in cluster_sps])

# <codecell>

sig_sps = np.array([scores[-1]/len(scores) for scores in scores_sps])

# <codecell>

plt.hist(sig_sps)
plt.show()

# <codecell>

thresh = sorted(sig_sps)[int(n_superpixels*.5)]

# <codecell>

# sig_sps[where(cluster_size_sps > 60)] = 0
sig_sps[sig_sps < thresh] = 0
sig_sps[cluster_size_sps < 2] = 0

# <codecell>

sigmap = sig_sps[cropped_segmentation]
sigmap[~cropped_mask] = 0

# <codecell>

plt.matshow(sigmap)
plt.colorbar()
plt.show()

# <codecell>

highlighted_sps = np.where((cluster_size_sps < 60) & (cluster_size_sps > 2))[0]
n_highlights = len(highlighted_sps)

# <codecell>

votes = np.zeros((n_highlights,), dtype=np.int)
for i in range(n_highlights):
    voting_member = highlighted_sps[overlap_matrix[i]]
    votes[i] = len(voting_member)
    
# for i in where([905 in cluster_sps[c] for c in highlighted_sps])[0]:
#     c = highlighted_sps[i]
#     print c, cluster_sps[c]

# <codecell>

from scipy.spatial.distance import pdist, squareform

# <codecell>

overlap_matrix = np.zeros((n_highlights, n_highlights), dtype=np.bool)
for i in range(n_highlights):
    for j in range(n_highlights):
        if i != j:
            c1 = cluster_sps[highlighted_sps[i]]
            c2 = cluster_sps[highlighted_sps[j]]
            overlap_matrix[i, j] = (len(c1 & c2) > .6 * len(c1 | c2))

# <codecell>

plt.matshow(overlap_matrix)
plt.show()

# <codecell>

from scipy.cluster.hierarchy import average, fcluster, leaders, complete, single, dendrogram

lk = complete(1-squareform(overlap_matrix).astype(np.int))
T = fcluster(lk, .5)

n_cliques = len(set(T))
# cliques = [None for _ in range(n_cliques)]
cliques = [None] * n_cliques

for clique_id in range(n_cliques):
    cliques[clique_id] = highlighted_sps[where(T == clique_id)[0]]
#     cliques[clique_id] = where(T == clique_id)[0]

# <codecell>

[c for c in cliques if len(c) > 4]

# <codecell>

import networkx as nx
from networkx.algorithms.components import connected_components, strongly_connected_components
from networkx.algorithms import find_cliques, cliques_containing_node

# <codecell>

G = nx.Graph(overlap_matrix)
# components = [c for c in connected_components(G) if len(c) > 10]
# components = [c for c in list(find_cliques(G)) if len(c) > 3 and len(c) < 50]
# components = [[highlighted_sps[i] for i in c] for c in list(find_cliques(G)) if len(c) > 3 and len(c) < 50]

print len(components)

# <codecell>

from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(linkage='average', n_clusters=10)
clustering.fit(X_red)
plot_clustering(X_red, X, clustering.labels_, "%s linkage" % linkage)

# <codecell>

components_ranked = sorted(components, key=lambda c: compute_cluster_score(c, texton_hists, neighbors)[0], reverse=True)

# <codecell>

print cliques_containing_node(G, 1751)

# <codecell>

# hash_sps = np.array([np.sum(list(c)) for c in cluster_sps])

# q = dict([])

# for s in range(n_superpixels):
    
#     h = hash_sps[s]
    
#     if h not in q:
#         q[h] = [s]
#     else:
#         q[h].append(s)
        
# consensus_clusters = [qq for qq in q.itervalues() if len(qq) > 4]

# <codecell>

vis = visualize_cluster(cliques_containing_node(G, 1751)[0], cropped_segmentation, cropped_segmentation_vis)

# <codecell>

vis = visualize_cluster(components_ranked[0], cropped_segmentation, cropped_segmentation_vis)
plt.imshow(vis)
plt.show()

# <codecell>

for i, c in enumerate(components_ranked[:5]):
    vis = visualize_cluster(c, cropped_segmentation, cropped_segmentation_vis)
    cv2.imwrite('tmp%d.jpg'%i, vis)

# plt.imshow(vis)
# plt.show()

# <codecell>

vis = visualize_multiple_clusters(components_ranked[:10], cropped_segmentation, cropped_segmentation_vis)

# <codecell>

vis = visualize_multiple_clusters(components, cropped_segmentation, cropped_segmentation_vis)

# <codecell>

cv2.imwrite('sig_clusters.png', vis)

from IPython.display import FileLink
FileLink('sig_clusters.png')

