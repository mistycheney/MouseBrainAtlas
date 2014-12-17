# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from utilities import *
from scipy.spatial.distance import cdist

# <codecell>

def compute_cluster_score(cluster, texton_hists, D_sp_null):
    model = texton_hists[list(cluster)].mean(axis=0)
    D_sp_model = np.squeeze(cdist([model], texton_hists[list(cluster)], chi2))
    model_sum = np.sum(D_sp_model)
    null_sum = np.sum(D_sp_null[list(cluster)])

    # can be made weighted by superpixel size
    
    score = null_sum - model_sum
    return score, null_sum, model_sum

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

def grow_cluster(seed, neighbors, texton_hists, D_sp_null, model_fit_reduce_limit=.5, score_drop_tolerance=0.):
        
    curr_cluster = set([seed])
    frontier = [seed]
    
    curr_cluster_score, curr_null_score, curr_model_score = compute_cluster_score(curr_cluster, texton_hists, D_sp_null)
    
    while len(frontier) > 0:
        u = frontier.pop(-1)
            
        ns = np.array(list(neighbors[u]))
        
        ds = np.squeeze(cdist(texton_hists[u][np.newaxis, :], texton_hists[ns]))
        
        for v in ns[ds.argsort()]:
            if v == -1 or v in curr_cluster: 
                continue

            score_new, null_sum_new, model_sum_new = compute_cluster_score(curr_cluster | set([v]), texton_hists, D_sp_null)
            
#             if score_new > curr_cluster_score and model_sum_new - curr_model_score < model_fit_reduce_limit:
            if score_new > curr_cluster_score - score_drop_tolerance and model_sum_new - curr_model_score < model_fit_reduce_limit:
                curr_cluster.add(v)
                frontier.append(v)
                curr_cluster_score, curr_null_score, curr_model_score = compute_cluster_score(curr_cluster, texton_hists, D_sp_null)
                
            if len(curr_cluster) > 100:
                return curr_cluster
            
    return curr_cluster

