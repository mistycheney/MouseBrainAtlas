#! /usr/bin/env python

import os
import argparse
import sys
import time

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Grow regions',
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

from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import average, fcluster, single, complete

from skimage.color import gray2rgb
from skimage.measure import find_contours
from skimage.util import img_as_float

from networkx import from_dict_of_lists, Graph, adjacency_matrix, dfs_postorder_nodes
from networkx.algorithms import node_connected_component, dfs_successors, dfs_postorder_nodes

import networkx
from itertools import chain

texton_hists = dm.load_pipeline_result('texHist')
segmentation = dm.load_pipeline_result('segmentation')
n_superpixels = segmentation.max() + 1
textonmap = dm.load_pipeline_result('texMap')
n_texton = textonmap.max() + 1
neighbors = dm.load_pipeline_result('neighbors')

edge_coords = dict(dm.load_pipeline_result('edgeCoords'))
neighbor_graph = from_dict_of_lists(dict(enumerate(neighbors)))

neighbors_long = dict([(s, set([n for n in nbrs if len(edge_coords[frozenset([s,n])]) > 10])) 
                       for s, nbrs in enumerate(neighbors)])

neighbor_long_graph = from_dict_of_lists(neighbors_long)

sp_centroids = dm.load_pipeline_result('spCentroids')[:, ::-1]

def compute_nearest_surround(cluster):
    cluster_list = list(cluster)
    cluster_avg = texton_hists[cluster_list].mean(axis=0)    
    surrounds = set([i for i in set.union(*[neighbors[c] for c in cluster]) if i not in cluster and i != -1])
    surrounds_list = list(surrounds)
    ds = np.squeeze(chi2s([cluster_avg], texton_hists[surrounds_list]))     
    surround_dist = ds.min()
    return surrounds_list[ds.argmin()]
    

def compute_cluster_score(cluster, texton_hists, verbose=False, thresh=.2):
    
    cluster_list = list(cluster)
    cluster_avg = texton_hists[cluster_list].mean(axis=0)
    
    surrounds = set([i for i in set.union(*[neighbors[c] for c in cluster]) if i not in cluster and i != -1])
    if len(surrounds) == 0: # single sp on background
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    surrounds_list = list(surrounds)
    ds = np.atleast_1d(np.squeeze(cdist([cluster_avg], texton_hists[surrounds_list], chi2)))
   
    surround_dist = np.count_nonzero(ds > thresh) / float(len(ds)) # hard

    if verbose:
        print 'min', surrounds_list[ds.argmin()]


    score = surround_dist
    
    if len(cluster) > 1:
        ds = np.squeeze(chi2s([cluster_avg], texton_hists[list(cluster)]))
        var = ds.mean()
    else:
        var = 0
    
    interior_dist = np.nan
    compactness = np.nan
    interior_pval = np.nan
    surround_pval = np.nan
    size_prior = np.nan
    
    return score, surround_dist, var, compactness, surround_pval, interior_pval, size_prior


from skimage.feature import peak_local_max
from scipy.spatial import ConvexHull
from matplotlib.path import Path

from skimage.feature import peak_local_max
from scipy.spatial import ConvexHull
from matplotlib.path import Path

def grow_cluster4(seed, verbose=False, all_history=False, coherence_limit=0.005, num_sp_percentage_limit=0.05,
                 min_size=4, min_distance=5, thresh=.2):
    try:
    
        visited = set([])
        curr_cluster = set([])

        candidate_scores = [0]
        candidate_sps = [seed]

        score_tuples = []
        added_sps = []
        n_sps = []
        
        cluster_list = []
        addorder_list = []
        
        iter_ind = 0

        hull_begin = False
        
        nearest_surrounds = []
        toadd_list = []

        while len(candidate_sps) > 0:
            
            if verbose:
                print '\niter', iter_ind

            best_ind = np.argmax(candidate_scores)

            just_added_score = candidate_scores[best_ind]
            sp = candidate_sps[best_ind]

            del candidate_scores[best_ind]
            del candidate_sps[best_ind]

            if sp in curr_cluster:
                continue

            curr_cluster.add(sp)
            added_sps.append(sp)

            extra_sps = []
            
#             sg = neighbor_graph.subgraph(list(set(range(n_superpixels))-curr_cluster))
            sg = neighbor_long_graph.subgraph(list(set(range(n_superpixels))-curr_cluster))
            for c in networkx.connected_components(sg):
                if len(c) < 10: # holes
                    extra_sps.append(c)
            extra_sps = list(chain(*extra_sps))
            curr_cluster |= set(extra_sps)
            added_sps += extra_sps
                
#             if not hull_begin:
#                 pts = sp_centroids[list(curr_cluster)]
#                 # if all points are colinear, ConvexHull will fail
#                 try:
#                     hull = ConvexHull(pts, incremental=True)
#                     hull_begin = True
#                 except:
#                     pass

#             if hull_begin:
                
#                 hull.add_points([sp_centroids[sp]])

#                 vertices = hull.points[hull.vertices]
#                 mpath = Path(vertices)
#                 xmin, ymin = vertices.min(axis=0)
#                 xmax, ymax = vertices.max(axis=0)

#                 in_bbox_sps = np.where((sp_centroids[:,0] >= xmin) & (sp_centroids[:,1] >= ymin) & \
#                                 (sp_centroids[:,0] <= xmax) & (sp_centroids[:,1] <= ymax))[0]

#                 extra_sps_to_consider = np.asarray(list(set(in_bbox_sps)))
#                 if verbose:
#                     print '\nextra_sps_to_consider', extra_sps_to_consider
#                 if len(extra_sps_to_consider) > 0:
#                     in_hull = mpath.contains_points(sp_centroids[extra_sps_to_consider])
#                     if np.any(in_hull):
#                         extra_sps = extra_sps_to_consider[in_hull]
#                         hull.add_points(sp_centroids[extra_sps])
#                         curr_cluster |= set(list(extra_sps))
#                         added_sps += extra_sps.tolist()


            tt = compute_cluster_score(curr_cluster, texton_hists=texton_hists, verbose=verbose,
                                      thresh=thresh)
    
            nearest_surround = compute_nearest_surround(curr_cluster)
            nearest_surrounds.append(nearest_surround)
            
            tot, exterior, interior, compactness, surround_pval, interior_pval, size_prior = tt
            
            if len(curr_cluster) > 5 and (interior > coherence_limit):
                break
            
            if np.isnan(tot):
                return [seed], -np.inf
            score_tuples.append(np.r_[just_added_score, tt])
                    
            n_sps.append(len(curr_cluster))
    
            # just_added_score, curr_total_score, exterior_score, interior_score, compactness_score, surround_pval,
            # interior_pval, size_prior

            if verbose:
                print 'add', sp
                print 'extra', extra_sps
                print 'added_sps', added_sps
                print 'curr_cluster', curr_cluster
                print 'n_sps', n_sps
                print 'tt', tot
                if len(curr_cluster) != len(added_sps):
                    print len(curr_cluster), len(added_sps)
                    raise                

#             visited.add(sp)
            
            cluster_list.append(curr_cluster.copy())
            addorder_list.append(added_sps[:])
            
            
#             if hull_begin:
#             visited |= set(list(extra_sps))
            
#             if hull_begin and len(extra_sps) > 0:
            candidate_sps = (set(candidate_sps) | \
                             (set.union(*[neighbors_long[i] for i in list(extra_sps)+[sp]]) - {-1})) - curr_cluster
#             else:
            
#             candidate_sps = (set(candidate_sps) | (neighbors[sp] - set([-1])) | (visited - curr_cluster)) - curr_cluster                
#             candidate_sps = (set(candidate_sps) | (neighbors[sp] - {-1})) - curr_cluster                

#             if hull_begin and len(extra_sps) > 0:
#                 candidate_sps = (set(candidate_sps) | ((set.union(*[neighbors_long[i] for i in list(extra_sps)]) | neighbors_long[sp]) - set([-1])) | (visited - curr_cluster)) - curr_cluster
#             else:
#                 candidate_sps = (set(candidate_sps) | (neighbors_long[sp] - set([-1])) | (visited - curr_cluster)) - curr_cluster

            candidate_sps = list(candidate_sps)
            
            h_avg = texton_hists[list(curr_cluster)].mean(axis=0)
        
#             candidate_scores = -chi2s([h_avg], texton_hists[candidate_sps])

            candidate_scores = -.5*chi2s([h_avg], texton_hists[candidate_sps])-\
                            .5*chi2s([texton_hists[seed]], texton_hists[candidate_sps])
            
            candidate_scores = candidate_scores.tolist()

            if verbose:
#                 print 'candidate', candidate_sps
                print 'candidate\n'
    
                for i,j in sorted(zip(candidate_scores, candidate_sps), reverse=True):
                    print i, j
                print 'best', candidate_sps[np.argmax(candidate_scores)]
                
            toadd_list.append(candidate_sps[np.argmax(candidate_scores)])

            if len(curr_cluster) > int(n_superpixels * num_sp_percentage_limit):
                break
                
            iter_ind += 1

        score_tuples = np.array(score_tuples)
        scores = score_tuples[:,1]
        
        peaks_sorted, peakedness_sorted = find_score_peaks(scores, min_size=min_size, min_distance=min_distance)
        
        if all_history:
            return addorder_list, score_tuples, peaks_sorted, peakedness_sorted, nearest_surrounds, toadd_list
        else:
            return [addorder_list[i] for i in peaks_sorted], score_tuples[peaks_sorted, 1]
            
    except:
        print seed
        raise


def find_score_peaks(scores, min_size = 4, min_distance=10, threshold_rel=.3, peakedness_lim=.001,
                    peakedness_radius=5):
    
    if len(scores) > min_size + 1:
    
        peaks = peak_local_max(scores[min_size:]-scores[min_size:].min(), min_distance=min_distance, 
                                      threshold_rel=threshold_rel, exclude_border=False)
        
        if len(peaks) > 0:
            peaks = min_size + peaks.T
            peaks = peaks[0]

            peakedness = np.array([scores[p]-np.mean(np.r_[scores[p-peakedness_radius:p], 
                                                           scores[p+1:p+1+peakedness_radius]]) for p in peaks])

            peaks = peaks[peakedness > peakedness_lim]
            peakedness = peakedness[peakedness > peakedness_lim]

            peaks_sorted = peaks[scores[peaks].argsort()[::-1]]
        else:
            peaks = np.array([np.argmax(scores[min_size:]) + min_size])
            peakedness = np.atleast_1d([scores[p]-np.mean(np.r_[scores[max(0, p-peakedness_radius):p], 
                                                           scores[p+1:min(p+1+peakedness_radius, len(scores))]]) 
                                       for p in peaks])
            peaks = peaks[peakedness > peakedness_lim]
            peakedness = peakedness[peakedness > peakedness_lim]
            peaks_sorted = peaks[scores[peaks].argsort()[::-1]]
    else:
        peaks = np.array([np.argmax(scores)])
        peakedness = np.atleast_1d([scores[p]-np.mean(np.r_[scores[max(0, p-peakedness_radius):p], 
                                                           scores[p+1:min(p+1+peakedness_radius, len(scores))]]) 
                                       for p in peaks])
        peaks_sorted = peaks[scores[peaks].argsort()[::-1]]
    
    peakedness_sorted = np.atleast_2d(peakedness[scores[peaks].argsort()[::-1]])[0]
    
    return peaks_sorted, peakedness_sorted   


sys.stderr.write('growing regions ...\n')
t = time.time()
expansion_clusters_tuples = Parallel(n_jobs=16)(delayed(grow_cluster4)(s, min_size=1,
                                                                          min_distance=10,
                                                                         coherence_limit=0.05,
                                                                         thresh=.3) for s in range(n_superpixels))
sys.stderr.write('done in %f seconds\n' % (time.time() - t))

expansion_clusters, expansion_cluster_scores = zip(*expansion_clusters_tuples)

all_see_cluster_tuples = [(s,c) for s in range(n_superpixels)  for c in expansion_clusters[s] ]
all_cluster_scores = [c for s in range(n_superpixels)  for c in expansion_cluster_scores[s]]

all_seeds, all_clusters = zip(*all_see_cluster_tuples)

dedge_neighbors = dm.load_pipeline_result('dedgeNeighbors')
dedge_neighbor_graph = from_dict_of_lists(dedge_neighbors)


def order_nodes(sps, neighbor_graph):

    subg = neighbor_graph.subgraph(sps)
    d_suc = dfs_successors(subg)
    
    x = [(a,b) for a,b in d_suc.iteritems() if len(b) == 2]
    
    if len(x) == 0:
        trav = list(dfs_postorder_nodes(subg))
    else:
        root, two_leaves = x[0]

        left_branch = []
        right_branch = []

        c = two_leaves[0]
        left_branch.append(c)
        while c in d_suc:
            c = d_suc[c][0]
            left_branch.append(c)

        c = two_leaves[1]
        right_branch.append(c)
        while c in d_suc:
            c = d_suc[c][0]
            right_branch.append(c)

        trav = left_branch[::-1] + [root] + right_branch
        
    return trav

def find_boundary_dedges_ordered(cluster):

    surrounds = set([i for i in set.union(*[neighbors[c] for c in cluster]) if i not in cluster])
    surrounds = set([i for i in surrounds if any([n not in cluster for n in neighbors[i]])])
    
    non_border_dedges = [(s, int_sp) for s in surrounds for int_sp in set.intersection(set(cluster), neighbors[s]) 
                         if int_sp != -1 and s != -1]
    border_dedges = [(-1,f) for f in cluster if -1 in neighbors[f]] if -1 in surrounds else []
            
    dedges_cluster = non_border_dedges + border_dedges
    dedges_cluster_long = [dedge for dedge in dedges_cluster if len(edge_coords[frozenset(dedge)]) > 10]
    dedges_cluster_long_sorted = order_nodes(dedges_cluster_long, dedge_neighbor_graph)    
    
    return dedges_cluster_long_sorted

sys.stderr.write('find boundary edges ...\n')
t = time.time()
all_cluster_dedges = Parallel(n_jobs=16)(delayed(find_boundary_dedges_ordered)(c) for c in all_clusters)
sys.stderr.write('done in %f seconds\n' % (time.time() - t))

dm.save_pipeline_result(zip(all_seeds, all_clusters, all_cluster_scores, all_cluster_dedges), 'allSeedClusterScoreDedgeTuples')