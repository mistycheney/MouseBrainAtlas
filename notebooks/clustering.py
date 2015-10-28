from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import average, fcluster, leaders, complete, single, dendrogram, ward

from collections import defaultdict, Counter
from itertools import combinations, chain, product
from operator import itemgetter

from joblib import Parallel, delayed

import numpy as np

import networkx

def compute_overlap_minjaccard(c1, c2):
    if isinstance(c1, list):
        c1 = set(c1)
    if isinstance(c2, list):
        c2 = set(c2)
    return float(len(c1 & c2)) / min(len(c1),len(c2))

def compute_overlap_jaccard(c1, c2):
    if isinstance(c1, list):
        c1 = set(c1)
    if isinstance(c2, list):
        c2 = set(c2)
    return float(len(c1 & c2)) / len(c1 | c2)

def compute_overlap_size(c1, c2):
    if isinstance(c1, list):
        c1 = set(c1)
    if isinstance(c2, list):
        c2 = set(c2)
    return len(c1 & c2)

def compute_nonoverlap_area(c1, c2, sp_areas):
    if isinstance(c1, list):
        c1 = set(c1)
    if isinstance(c2, list):
        c2 = set(c2)
    if len(c1&c2) == 0:
        return 100.
    else:
        nonoverlap_sps = (c1|c2)-(c1&c2)
        return sp_areas[list(nonoverlap_sps)].sum()/5e5 # area of the whole frame is 100

def compute_overlap_partial(indices, sets, metric='jaccard', sp_areas=None):
    n_sets = len(sets)
    
    overlap_matrix = np.zeros((len(indices), n_sets))
        
    for ii, i in enumerate(indices):
        for j in range(n_sets):
            c1 = set(sets[i])
            c2 = set(sets[j])
            if len(c1) == 0 or len(c2) == 0:
                overlap_matrix[ii, j] = 0
            else:
                if metric == 'min-jaccard':
                    overlap_matrix[ii, j] = compute_overlap_minjaccard(c1, c2)
                elif metric == 'jaccard':
                    overlap_matrix[ii, j] = compute_overlap_jaccard(c1, c2)
                elif metric == 'overlap-size':
                    overlap_matrix[ii, j] = compute_overlap_size(c1, c2)
                elif metric == 'nonoverlap-area':
                    overlap_matrix[ii, j] = compute_nonoverlap_area(c1, c2, sp_areas)
                else:
                    raise Exception('metric %s is unknown'%metric)
            
    return overlap_matrix

def compute_pairwise_distances(sets, metric, sp_areas=None):

    if metric == 'nonoverlap-area':
        
        partial_distance_matrix = Parallel(n_jobs=16, max_nbytes=1e6)(delayed(compute_overlap_partial)(s, sets, metric=metric,
                                                                                                      sp_areas=sp_areas) 
                                                              for s in np.array_split(range(len(sets)), 16))
        distance_matrix = np.vstack(partial_distance_matrix)
        np.fill_diagonal(distance_matrix, 0)
        return distance_matrix
        
    elif hasattr(metric, '__call__'):
        
        partial_distance_matrix = Parallel(n_jobs=16, max_nbytes=1e6)(delayed(cdist)(s, sets, metric=metric) 
                                                                      for s in np.array_split(sets, 16))
        distance_matrix = np.vstack(partial_distance_matrix)
        np.fill_diagonal(distance_matrix, 0)
        return distance_matrix
        
    elif metric == 'overlap-size':
        partial_overlap_mat = Parallel(n_jobs=16, max_nbytes=1e6)(delayed(compute_overlap_partial)(s, sets, metric='overlap-size') 
                                            for s in np.array_split(range(len(sets)), 16))
        overlap_matrix = np.vstack(partial_overlap_mat)
        return overlap_matrix
    
    else:
    
        partial_overlap_mat = Parallel(n_jobs=16, max_nbytes=1e6)(delayed(compute_overlap_partial)(s, sets, metric=metric) 
                                            for s in np.array_split(range(len(sets)), 16))
        overlap_matrix = np.vstack(partial_overlap_mat)
        distance_matrix = 1 - overlap_matrix
    
        np.fill_diagonal(distance_matrix, 0)
    
        return distance_matrix


def group_tuples(items=None, val_ind=None, dist_thresh = 0.1, distance_matrix=None, 
                 metric='jaccard', linkage='complete', sp_areas=None):
    '''
    items: a dict or list of tuples
    val_ind: the index of the item of interest within each tuple
    '''
    
    if distance_matrix is not None:
        if items is not None:
            if isinstance(items, dict):
                keys = items.keys()
                values = items.values()
            elif isinstance(items, list):
                keys = range(len(items))
                if isinstance(items[0], tuple):
                    values = map(itemgetter(val_ind), items)
                else:
                    values = items
    else:
        if isinstance(items, dict):
            keys = items.keys()
            values = items.values()
        elif isinstance(items, list):
            keys = range(len(items))
            if isinstance(items[0], tuple):
                values = map(itemgetter(val_ind), items)
            else:
                values = items
        else:
            raise Exception('clusters is not the right type')

        assert items is not None, 'items must be provided'
        distance_matrix = compute_pairwise_distances(values, metric, sp_areas=sp_areas)
    
    if items is None:
        assert distance_matrix is not None, 'distance_matrix must be provided.'    
        
    if linkage=='complete':
        lk = complete(squareform(distance_matrix))
    elif linkage=='average':
        lk = average(squareform(distance_matrix))
    elif linkage=='single':
        lk = single(squareform(distance_matrix))

    # T = fcluster(lk, 1.15, criterion='inconsistent')
    T = fcluster(lk, dist_thresh, criterion='distance')
    
    n_groups = len(set(T))
    groups = [None] * n_groups

    for group_id in range(n_groups):
        groups[group_id] = np.where(T == group_id+1)[0]

    index_groups = [[keys[i] for i in g] for g in groups if len(g) > 0]
    item_groups = [[items[i] for i in g] for g in groups if len(g) > 0]
    
    return index_groups, item_groups, distance_matrix

def smart_union(x):
    cc = Counter(chain(*x))
    gs = set([s for s, c in cc.iteritems() if c > (cc.most_common(1)[0][1]*.3)])                           
    return gs
