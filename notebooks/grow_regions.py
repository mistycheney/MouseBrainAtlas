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

def grow_cluster(seed, neighbors, texton_hists, D_sp_null):
    
    curr_cluster = set([seed])
    frontier = [seed]

    while len(frontier) > 0:
        u = frontier.pop(-1)
        for v in neighbors[u]:
            if v == -1 or v in curr_cluster: 
                continue

            score_new, null_sum_new, model_sum_new = compute_cluster_score(curr_cluster | set([v]),  texton_hists, D_sp_null)
            score_old, null_sum_old, model_sum_old = compute_cluster_score(curr_cluster,  texton_hists, D_sp_null)

            if score_new > score_old and model_sum_new - model_sum_old < 0.5 :
                curr_cluster.add(v)
#                 print curr_cluster
                frontier.append(v)
                
    return curr_cluster

