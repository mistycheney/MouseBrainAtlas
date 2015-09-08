import sys
import os
import time

sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], 'pipeline_scripts'))

if os.environ['DATASET_VERSION'] == '2014':
	from utilities2014 import *
elif os.environ['DATASET_VERSION'] == '2015':
	from utilities import *

from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import average, fcluster, single, complete

from joblib import Parallel, delayed

from skimage.color import gray2rgb
from skimage.measure import find_contours
from skimage.util import img_as_float

import matplotlib.pyplot as plt

sys.path.append('/home/yuncong/project/opencv-2.4.9/release/lib/python2.7/site-packages')
import cv2

from networkx import from_dict_of_lists, Graph, adjacency_matrix, dfs_postorder_nodes
from networkx.algorithms import node_connected_component

# os.environ['GORDON_DATA_DIR'] = '/home/yuncong/project/DavidData2014tif/'
# os.environ['GORDON_REPO_DIR'] = '/home/yuncong/Brain'
# os.environ['GORDON_RESULT_DIR'] = '/home/yuncong/project/DavidData2014results/'
# os.environ['GORDON_LABELING_DIR'] = '/home/yuncong/project/DavidData2014labelings/'


def find_boundary_sps(clusters, neighbors, neighbor_graph, mode='both'):
    '''
    Identify superpixels that are at the boundary of regions: surround set and frontier set
    
    Parameters
    ----------
    clusters : list of integer lists
    neighbors : neighbor_list
    neighbor_graph : 
    '''
        
    n_superpixels = len(clusters)
    
    surrounds_sps = []
    frontiers_sps = []
    
    for cluster_ind, cluster in enumerate(clusters):
        
        surrounds = set([i for i in set.union(*[neighbors[c] for c in cluster]) if i not in cluster and i != -1])
#         surrounds = set([i for i in surrounds if any([(n not in cluster) and (n not in surrounds) for n in neighbors[i]])])
        surrounds = set([i for i in surrounds if any([n not in cluster for n in neighbors[i]])])

        if len(surrounds) == 0:
            surrounds_sps.append([])
            frontiers_sps.append([])
        else:
            if mode == 'surround' or mode == 'both':
                surrounds_subgraph = neighbor_graph.subgraph(surrounds)
                surrounds_traversal = list(dfs_postorder_nodes(surrounds_subgraph))
                surrounds_sps.append(surrounds_traversal)

            if mode == 'frontier' or mode == 'both':
                frontiers = set.union(*[neighbors[c] for c in surrounds]) & set(cluster)
                frontiers_subgraph = neighbor_graph.subgraph(frontiers)
                frontiers_traversal = list(dfs_postorder_nodes(frontiers_subgraph))
                frontiers_sps.append(frontiers_traversal)
    
    if mode == 'frontier':
        return frontiers_sps
    elif mode == 'surround':
        return surrounds_sps
    elif mode == 'both':
        return surrounds_sps, frontiers_sps


def compute_cluster_score(cluster, texton_hists, neighbors):
    
    cluster_list = list(cluster)
    cluster_avg = texton_hists[cluster_list].mean(axis=0)
    
    surrounds = set([i for i in set.union(*[neighbors[c] for c in cluster]) if i not in cluster and i != -1])
    if len(surrounds) == 0: # single sp on background
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    surrounds_list = list(surrounds)
    surround_dist = np.squeeze(cdist([cluster_avg], texton_hists[surrounds_list], chi2)).min()

    surds = find_boundary_sps([cluster], neighbors=neighbors, neighbor_graph=neighbor_graph, mode='surround')
    
    compactness = len(surds[0])**2/float(len(cluster))
    compactness = .001 * np.maximum(compactness-40,0)**2
    
    size_prior = .1 * (1-np.exp(-.8*len(cluster)))
    
    score = surround_dist - compactness + size_prior
    
    interior_dist = np.nan
    interior_pval = np.nan
    surround_pval = np.nan
    
    return score, surround_dist, interior_dist, compactness, surround_pval, interior_pval, size_prior


neighbors_global = None

def grow_cluster3(seed, texton_hists, neighbors=None, output=False, all_history=False):
    
    if neighbors is None:
        neighbors = neighbors_global
    
    visited = set([])
    curr_cluster = set([])
        
    candidate_scores = [0]
    candidate_sps = [seed]

    score_tuples = []
    added_sps = []
    
    iter_ind = 0
        
    while len(candidate_sps) > 0:

        best_ind = np.argmax(candidate_scores)
        
        heuristic = candidate_scores[best_ind]
        sp = candidate_sps[best_ind]
        
        del candidate_scores[best_ind]
        del candidate_sps[best_ind]
        
        if sp in curr_cluster:
            continue
                
        iter_ind += 1
        curr_cluster.add(sp)
        added_sps.append(sp)
        
        tt = compute_cluster_score(curr_cluster, texton_hists=texton_hists, neighbors=neighbors)
        tot, exterior, interior, compactness, surround_pval, interior_pval, size_prior = tt
        if np.isnan(tot):
            return [seed], -np.inf
        score_tuples.append(np.r_[heuristic, tt])
        
        if output:
            print 'iter', iter_ind, 'add', sp

        visited.add(sp)
        
        candidate_sps = (set(candidate_sps) | (neighbors[sp] - set([-1])) | (visited - curr_cluster)) - curr_cluster
        candidate_sps = list(candidate_sps)
        
#         f_avg = texton_freqs[list(curr_cluster)].sum(axis=0)
#         candidate_scores = [chi2pval(f_avg, texton_freqs[i])[0] for i in candidate_sps]

        h_avg = texton_hists[list(curr_cluster)].mean(axis=0)
        candidate_scores = [-chi2(h_avg, texton_hists[i]) for i in candidate_sps]

#         candidate_scores = [compute_cluster_score(curr_cluster | set([s])) for s in candidate_sps]
                
        if len(visited) > int(n_superpixels * 0.03):
            break

    score_tuples = np.array(score_tuples)
    
    min_size = 2
    scores = score_tuples[:,1]
    cutoff = np.argmax(scores[min_size:]) + min_size
    
    if output:
        print 'cutoff', cutoff

    final_cluster = added_sps[:cutoff]
    final_score = scores[cutoff]
    
    if all_history:
        return list(final_cluster), final_score, added_sps, score_tuples
    else:
        return list(final_cluster), final_score


def spSet_to_edgeSet(cluster, n_superpixels, neighbors=None, fill_holes=False):

    
    if neighbors is None:
        neighbors = neighbors_global

    
    cluster = set(cluster)
    
    surrounds = set([i for i in set.union(*[neighbors[c] for c in cluster]) if i not in cluster and i != -1])
    surrounds = set([i for i in surrounds if any([n not in cluster for n in neighbors[i]])])

    if fill_holes:    
        outside = set(range(n_superpixels)) - cluster - surrounds
        hole_candidates = set(range(n_superpixels)) - cluster
    
        goon = True
        while goon:
            goon = False
            for hc in hole_candidates - outside:
                if any([s in outside for s in neighbors[hc]]):
        #             print hc
                    outside.add(hc)
                    goon = True

        holes = set(range(n_superpixels)) - outside - cluster
    #     print holes

        cluster = set(cluster) | holes

    surrounds = set([i for i in set.union(*[neighbors[c] for c in cluster]) if i not in cluster and i != -1])
    surrounds = set([i for i in surrounds if any([n not in cluster for n in neighbors[i]])])

    # only happen in island node
    if len(surrounds) == 0:
        return []
    else:
        frontiers = set.union(*[neighbors[c] for c in surrounds]) & set(cluster)    

    region_edges = []
    for s in surrounds:
        for f in neighbors[s] & set(frontiers):
            region_edges.append((s, f))

    for i in cluster:
        if -1 in neighbors[i]:
            region_edges.append((-1, i))

    return sorted(region_edges)


import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Generate region proposals and detect closedRegion landmarks')

    parser.add_argument("stack_name", type=str, help="stack name")
    parser.add_argument("slice_ind", type=int, help="slice index")
    parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
    parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
    parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
    args = parser.parse_args()

    section_id = int(args.slice_ind)
    
    dm = DataManager(generate_hierarchy=False, stack=args.stack_name, resol='x5', section=section_id,
                    gabor_params_id=args.gabor_params_id,
                    segm_params_id=args.segm_params_id,
                    vq_params_id=args.vq_params_id)
    dm._load_image()

    texton_hists = dm.load_pipeline_result('texHist', 'npy')
    segmentation = dm.load_pipeline_result('segmentation', 'npy')
    n_superpixels = len(np.unique(segmentation)) - 1
    textonmap = dm.load_pipeline_result('texMap', 'npy')
    n_texton = len(np.unique(textonmap)) - 1

    neighbors = dm.load_pipeline_result('neighbors', 'pkl')
    neighbors_global = neighbors

    sp_properties = dm.load_pipeline_result('spProps', 'npy')
    # each item is (center_y, center_x, area, mean_intensity, ymin, xmin, ymax, xmax)
    segmentation_vis = dm.load_pipeline_result('segmentationWithoutText', 'jpg')

#     try:
#         sp_sp_dists = dm.load_pipeline_result('texHistPairwiseDist', 'npy')
# #         raise
#     except:
#         def f(a):
#             sp_dists = cdist(a, texton_hists, metric=chi2)
#     #         sp_dists = cdist(a, texton_hists, metric=js)
#             return sp_dists

#         sp_dists = Parallel(n_jobs=16)(delayed(f)(s) for s in np.array_split(texton_hists, 16))
#         sp_sp_dists = np.vstack(sp_dists)

#         dm.save_pipeline_result(sp_sp_dists, 'texHistPairwiseDist', 'npy')

#     center_dists = pdist(sp_properties[:, :2])
#     center_dist_matrix = squareform(center_dists)

    neighbors_dict = dict(zip(np.arange(n_superpixels), [list(i) for i in neighbors]))
    neighbor_graph = from_dict_of_lists(neighbors_dict)


    try:
        expansion_clusters_tuples = dm.load_pipeline_result('clusters', 'pkl')
#         raise
    except Exception as e:

#         b = time.time()

        # Observation: if `neighbors` is passed as argument, execution takes 4.5 times than if `neighbors` is global
        # Reason: `neighbors` is a list of sets, not mem-mappable. To see the error, run the lines below:
        # from joblib import load, dump
        # _ = dump(neighbors, '/tmp/tmp')
        # large_memmap = load('/tmp/tmp', mmap_mode='r+')
        # Solution: make `neighbors` global
        # p.s. if CPU utilizations of many processes are low, it means IO is taking much of the time
        # p.s. specify max_nbytes argument to Parallel enables memmap for shared variables, but using it results in 
        # error `zero-dimensional array concatenates...`
        expansion_clusters_tuples = Parallel(n_jobs=16)(delayed(grow_cluster3)(s, texton_hists=texton_hists)
                                                                    for s in range(n_superpixels))

#         print 'grow cluster', time.time() - b

        dm.save_pipeline_result(expansion_clusters_tuples, 'clusters', 'pkl')
    
    expansion_clusters, expansion_cluster_scores = zip(*expansion_clusters_tuples)
    
    if not dm.check_pipeline_result('clusterEdges', 'pkl'):
        edgeSets = Parallel(n_jobs=16)(delayed(spSet_to_edgeSet)(c, n_superpixels=n_superpixels,
                                                                 fill_holes=False) for c in expansion_clusters)
        dm.save_pipeline_result(edgeSets, 'clusterEdges', 'pkl')
    
    if not dm.check_pipeline_result('clusterSurrounds', 'pkl'):
        surrounds_sps, frontiers_sps = find_boundary_sps(expansion_clusters, neighbors=neighbors, 
                                                 neighbor_graph=neighbor_graph, mode='both')

        dm.save_pipeline_result(surrounds_sps, 'clusterSurrounds', 'pkl')
        dm.save_pipeline_result(frontiers_sps, 'clusterFrontiers', 'pkl')
        
#     expansion_cluster_scores = np.array(expansion_cluster_scores)


#     try:
#         D = dm.load_pipeline_result('clusterPairwiseDist', 'npy')
# #         raise
#     except:

#         b = time.time()

#         D = set_pairwise_distances(expansion_clusters, metric=2)
#         dm.save_pipeline_result(D, 'clusterPairwiseDist', 'npy')

#         print 'compute pairwise', time.time() - b


#     try:
#         expansion_cluster_groups = dm.load_pipeline_result('clusterGroups', 'pkl')
# #         raise
#     except:

#         b = time.time()

#         expansion_cluster_groups = group_clusters(expansion_clusters, dist_thresh=.8, distance_matrix=D)
#         dm.save_pipeline_result(expansion_cluster_groups, 'clusterGroups', 'pkl')

#         print 'group clusters', time.time() - b

    
#     try:
#         representative_clusters = dm.load_pipeline_result('representativeClusters', 'pkl')
#         edgeSets = dm.load_pipeline_result('closedRegionsTop30Edgesets', 'pkl')
#     except:
        
#         print len(expansion_cluster_groups), 'expansion cluster groups'
#         expansion_cluster_group_sizes = np.array(map(len, expansion_cluster_groups))


#         big_group_indices = np.where(expansion_cluster_group_sizes > 5)[0]
#         n_big_groups = len(big_group_indices)
#         print n_big_groups, 'big cluster groups'
#         big_groups = [expansion_cluster_groups[i] for i in big_group_indices]

#         from collections import Counter

#         representative_clusters = []
#         representative_cluster_scores = []
#         representative_cluster_indices = []

#         big_groups_valid = []

#         for g in big_groups:
#             for i in np.argsort(expansion_cluster_scores[g])[::-1]:
#                 c = expansion_clusters[g[i]]
#                 sc = expansion_cluster_scores[g[i]]
#                 if len(c) > n_superpixels * .004:
#                     representative_clusters.append(c)
#                     representative_cluster_indices.append(g[i])
#                     representative_cluster_scores.append(sc)
#                     big_groups_valid.append(g)
#                     break

#         print len(representative_clusters), 'representative clusters'

#         representative_cluster_scores_sorted,\
#         representative_clusters_sorted_by_score,\
#         representative_cluster_indices_sorted_by_score,\
#         big_groups_sorted_by_score = map(list, zip(*sorted(zip(representative_cluster_scores,\
#                                                                representative_clusters,\
#                                                                representative_cluster_indices,\
#                                                                big_groups_valid), reverse=True)))

#         representative_clusters = zip(representative_cluster_scores_sorted, representative_clusters_sorted_by_score, 
#                        representative_cluster_indices_sorted_by_score, 
#                        big_groups_sorted_by_score)

#         dm.save_pipeline_result(representative_clusters, 'representativeClusters', 'pkl')
    
#         edgeSets = Parallel(n_jobs=16)(delayed(spSet_to_edgeSet)(c, n_superpixels=n_superpixels,
#                                                  fill_holes=True) for c in representative_clusters_sorted_by_score[:30])

#         dm.save_pipeline_result(edgeSets, 'closedRegionsTop30Edgesets', 'pkl')

#     vis = dm.visualize_edge_sets(edgeSets[:10], width=5)
#     dm.save_pipeline_result( vis, 'contoursTop10' , 'jpg')

#     vis = dm.visualize_edge_sets(edgeSets[:20], width=5)
#     dm.save_pipeline_result( vis, 'contoursTop20' , 'jpg')

#     vis = dm.visualize_edge_sets(edgeSets[:30], width=5)
#     dm.save_pipeline_result( vis, 'contoursTop30' , 'jpg')
    
#     vis = dm.visualize_edge_sets(edgeSets[:30], text=True)
#     dm.save_pipeline_result( vis, 'contoursTop30WithText' , 'jpg')
