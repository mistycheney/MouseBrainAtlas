#! /usr/bin/env python

import os
import argparse
import sys
import time

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Find growed cluster consensus',
    epilog="")

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("slice_ind", type=int, help="slice index")
parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
args = parser.parse_args()


from joblib import Parallel, delayed

sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], 'notebooks'))
from utilities2015 import *

dm = DataManager(data_dir=os.environ['GORDON_DATA_DIR'], 
                 repo_dir=os.environ['GORDON_REPO_DIR'], 
                 result_dir=os.environ['GORDON_RESULT_DIR'], 
                 labeling_dir=os.environ['GORDON_LABELING_DIR'],
                 gabor_params_id=args.gabor_params_id, 
                 segm_params_id=args.segm_params_id, 
                 vq_params_id=args.vq_params_id,
                 stack=args.stack_name, 
                 section=args.slice_ind)

#======================================================


def compute_cluster_coherence_score(cluster, verbose=False):
    
    if len(cluster) > 1:
        cluster_avg = texton_hists[cluster].mean(axis=0)
        ds = np.squeeze(chi2s([cluster_avg], texton_hists[list(cluster)]))
        var = np.sum(ds**2)/len(ds)
    else:
        var = 0
    
    return var

def compute_cluster_significance_score(cluster, verbose=False, method='min'):
    
    cluster_avg = texton_hists[cluster].mean(axis=0)
    
    surround_list = [i for i in set.union(*[neighbors[c] for c in cluster]) if i not in cluster and i != -1]
    
    surrounds = set(surround_list)
  
    if verbose:
        print 'surround_list', surround_list
        
    ds = np.squeeze(chi2s([cluster_avg], texton_hists[surround_list])) 

    if method == 'min':
        surround_dist = ds.min()
    elif method == 'mean':
        surround_dist = ds.mean()
    elif method == 'percentage':
        surround_dist = np.count_nonzero(ds > .2) / float(len(ds)) # hard
    elif method == 'percentage-soft':
        sigma = .05
        surround_dist = np.sum(1./(1+np.exp((.2-ds)/sigma)))/len(ds); #soft
    
    if verbose:
        print 'min', surround_list[ds.argmin()]

    score = surround_dist
    
    return score

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

def compute_overlap_partial(indices, sets, metric='jaccard'):
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
                else:
                    raise Exception('metric %s is unknown'%metric)
            
    return overlap_matrix

def compute_pairwise_distances(sets, metric):

    partial_overlap_mat = Parallel(n_jobs=16, max_nbytes=1e6)(delayed(compute_overlap_partial)(s, sets, metric=metric) 
                                        for s in np.array_split(range(len(sets)), 16))
    overlap_matrix = np.vstack(partial_overlap_mat)
    distance_matrix = 1 - overlap_matrix
    
    np.fill_diagonal(distance_matrix, 0)
    
    return distance_matrix


def group_tuples(items=None, val_ind=None, dist_thresh = 0.1, distance_matrix=None, metric='jaccard', linkage='complete'):
    '''
    items: a dict or list of tuples
    val_ind: the index of the item of interest within each tuple
    '''
    
    if distance_matrix is not None:
        if items is not None:
            values = map(itemgetter(val_ind), items)
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
        distance_matrix = compute_pairwise_distances(values, metric)
    
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

segmentation = dm.load_pipeline_result('segmentation')
n_superpixels = segmentation.max() + 1

textonmap = dm.load_pipeline_result('texMap')
n_texton = textonmap.max() + 1
texton_hists = dm.load_pipeline_result('texHist')

neighbors = dm.load_pipeline_result('neighbors')

edge_coords = dict(dm.load_pipeline_result('edgeCoords'))
edge_neighbors = dm.load_pipeline_result('edgeNeighbors')

dedge_vectors = dm.load_pipeline_result('edgeVectors')
dedge_neighbors = dm.load_pipeline_result('dedgeNeighbors')

try:
    raise
    good_clusters = dm.load_pipeline_result('goodClusters')
    good_dedges = dm.load_pipeline_result('goodDedges')

except:

    all_seed_cluster_score_dedge_tuples = dm.load_pipeline_result('allSeedClusterScoreDedgeTuples')

    all_seed, all_clusters, all_cluster_scores, all_cluster_dedges = zip(*all_seed_cluster_score_dedge_tuples)

    all_cluster_coherences = np.array([compute_cluster_coherence_score(cl) for cl in all_clusters])

    remaining_cluster_indices = [i for i, (cl, coh, sig) in enumerate(zip(all_clusters, all_cluster_coherences, all_cluster_scores)) 
                    if coh < .005 and sig > .03]

    all_seed = [all_seed[i] for i in remaining_cluster_indices]
    all_clusters = [all_clusters[i] for i in remaining_cluster_indices]
    all_cluster_scores = [all_cluster_scores[i] for i in remaining_cluster_indices]
    all_cluster_coherences = [all_cluster_coherences[i] for i in remaining_cluster_indices]
    all_cluster_dedges = [all_cluster_dedges[i] for i in remaining_cluster_indices]
    all_seed_cluster_score_dedge_tuples = [all_seed_cluster_score_dedge_tuples[i] for i in remaining_cluster_indices]

    from scipy.spatial.distance import cdist, pdist, squareform
    from scipy.cluster.hierarchy import average, fcluster, leaders, complete, single, dendrogram, ward

    from collections import defaultdict, Counter
    from itertools import combinations, chain, product

    import networkx


    sys.stderr.write('group clusters ...\n')
    t = time.time()

    all_seed_cluster_sig_coh_dedge_tuples = zip(all_seed, all_clusters, all_cluster_scores, all_cluster_coherences,
                                             all_cluster_dedges)

    tuple_indices_grouped, tuples_grouped, _ = group_tuples(all_seed_cluster_sig_coh_dedge_tuples, 
                                                             val_ind = 1,
                                                             metric='jaccard',
                                                             dist_thresh=.2)
    sys.stderr.write('done in %f seconds\n' % (time.time() - t))

    all_seeds_grouped, all_clusters_grouped, \
    all_scores_grouped, all_cohs_grouped, all_dedges_grouped = [list(map(list, lst)) for lst in zip(*[zip(*g) for g in tuples_grouped])]


    all_scores_grouped = [[compute_cluster_significance_score(g, method='percentage') for g in cg] 
                          for cg in all_clusters_grouped]

    group_rep_indices = map(np.argmax, all_scores_grouped)

    rep_tuples = [(sc_grp[rep_ind], coh_grp[rep_ind], cl_grp[rep_ind], dedge_grp[rep_ind], gi) 
                    for gi, (rep_ind, cl_grp, sc_grp, coh_grp, dedge_grp) in enumerate(zip(group_rep_indices, all_clusters_grouped, 
                                                                                  all_scores_grouped, 
                                                                                  all_cohs_grouped,
                                                                                  all_dedges_grouped))]

    rep_tuples_ranked = sorted(rep_tuples, reverse=True)

    rep_score_ranked = map(itemgetter(0), rep_tuples_ranked)
    rep_coh_ranked = map(itemgetter(1), rep_tuples_ranked)
    rep_clusters_ranked = map(itemgetter(2), rep_tuples_ranked)
    rep_dedges_ranked = map(itemgetter(3), rep_tuples_ranked)
    group_indices_ranked = map(itemgetter(4), rep_tuples_ranked)

    rep_entropy_ranked = np.nan_to_num([-np.sum(texton_hists[cl].mean(axis=0)*np.log(texton_hists[cl].mean(axis=0)) )
                          for cl in rep_clusters_ranked])

    sp_centroids = dm.load_pipeline_result('spCentroids')[:, ::-1]
    rep_centroids_ranked = np.array([sp_centroids[cl].mean(axis=0) for cl in rep_clusters_ranked])

    valid_indices = [i for i, (cl, ent, cent) in enumerate(zip(rep_clusters_ranked, rep_entropy_ranked, rep_centroids_ranked))
                if len(cl) > 3 and (ent > 2. or \
                  (cent[0] - dm.xmin > 800 and \
                   dm.xmax - cent[0] > 800 and \
                   cent[1] - dm.ymin > 800 and \
                   dm.ymax - cent[1] > 800)
                 )]
    
    good_cluster_tuples = [(sig, cl, dedges, i) for i, (sig, coh, cl, dedges, grp_ind) in enumerate(rep_tuples_ranked) 
                   if i in valid_indices]
    
    good_cluster_indices_grouped, good_cluster_tuples_grouped, _ = group_tuples(good_cluster_tuples, 
                                            val_ind = 1,
                                            metric='jaccard',
                                            dist_thresh=.6)

    good_cluster_rep_tuples = [tpls[np.argmax([sig for sig,cl,dedges,i in tpls])] 
                               for tpls in good_cluster_tuples_grouped]

    good_cluster_rep_tuples_ranked = sorted(good_cluster_rep_tuples, reverse=True)

    good_clusters = map(itemgetter(1), good_cluster_rep_tuples_ranked)
    good_dedges = map(itemgetter(2), good_cluster_rep_tuples_ranked)

    dm.save_pipeline_result(good_clusters, 'goodClusters')
    dm.save_pipeline_result(good_dedges, 'goodDedges')

viz = dm.visualize_edge_sets(good_dedges[:60], show_set_index=True)
dm.save_pipeline_result(viz, 'landmarksViz')

boundary_models = []

for i, es in enumerate(good_dedges[:60]):

    es = list(es)

    interior_texture = texton_hists[list(good_clusters[i])].mean(axis=0)

    surrounds = [e[0] for e in es]
    exterior_textures = np.array([texton_hists[s] if s!=-1 else np.nan * np.ones((texton_hists.shape[1],)) 
                                  for s in surrounds])
    # how to deal with -1 in surrounds? Assign to an all np.nan vector

    points = np.array([edge_coords[frozenset(e)].mean(axis=0) for e in es])
    center = points.mean(axis=0)

    boundary_models.append((es, interior_texture, exterior_textures, points, center))    

dm.save_pipeline_result(boundary_models, 'boundaryModels')