import sys
sys.path.append('/home/yuncong/project/opencv-2.4.9/release/lib/python2.7/site-packages')

sys.path.insert(0, '/home/yuncong/project/cython-munkres-wrapper/build/lib.linux-x86_64-2.7')
# import munkres
# reload(munkres)
from munkres import munkres

# https://github.com/jfrelinger/cython-munkres-wrapper,
# 10 times faster than the python implementation http://software.clapper.org/munkres/

# from munkres import Munkres # http://software.clapper.org/munkres/

sys.path.append('/home/yuncong/Brain/pipeline_scripts')
import utilities2014
reload(utilities2014)
from utilities2014 import *

import os
import time

# os.environ['GORDON_DATA_DIR'] = '/home/yuncong/project/DavidData2014tif/'
# os.environ['GORDON_REPO_DIR'] = '/home/yuncong/Brain'
# os.environ['GORDON_RESULT_DIR'] = '/home/yuncong/project/DavidData2014results/'
# os.environ['GORDON_LABELING_DIR'] = '/home/yuncong/project/DavidData2014labelings/'

from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import average, fcluster, leaders, complete, single, dendrogram, ward

from joblib import Parallel, delayed

from skimage.color import gray2rgb
from skimage.util import img_as_float, pad
from skimage.morphology import disk
from skimage.filter.rank import gradient

from collections import defaultdict, Counter
from itertools import combinations, chain, product

import networkx
from networkx import from_dict_of_lists, dfs_postorder_nodes

def compute_overlap(c1, c2):
    return float(len(c1 & c2)) / min(len(c1),len(c2))

def compute_overlap2(c1, c2):
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
                    overlap_matrix[ii, j] = compute_overlap(c1, c2)
                elif metric == 'jaccard':
                    overlap_matrix[ii, j] = compute_overlap2(c1, c2)
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


def group_clusters(clusters=None, dist_thresh = 0.1, distance_matrix=None, metric='jaccard', linkage='complete'):
    
    if distance_matrix is not None:
        keys = range(len(distance_matrix))
        if clusters is not None:
            values = clusters
        else:
            values = range(len(distance_matrix))
    else:
        if isinstance(clusters, dict):
            keys = clusters.keys()
            values = clusters.values()
        elif isinstance(clusters, list):
            if isinstance(clusters[0], tuple):
                keys = [i for i,j in clusters]
                values = [j for i,j in clusters]
            else:
                keys = range(len(clusters))
                values = clusters
        else:
            raise Exception('clusters is not the right type')
    
    if clusters is None:
        assert distance_matrix is not None, 'distance_matrix must be provided.'
    
    if distance_matrix is None:
        assert clusters is not None, 'clusters must be provided'
        distance_matrix = compute_pairwise_distances(values, metric)
        
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
    res = [[values[i] for i in g] for g in groups if len(g) > 0]
        
    return index_groups, res, distance_matrix

def smart_union(x):
    cc = Counter(chain(*x))
    gs = set([s for s, c in cc.iteritems() if c > (cc.most_common(1)[0][1]*.3)])                           
    return gs

import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Detect significant edge groups')

    parser.add_argument("stack_name", type=str, help="stack name")
    parser.add_argument("slice_ind", type=int, help="slice index")
    parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
    parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
    parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
    args = parser.parse_args()

    section_id = int(args.slice_ind)
    
    dm = DataManager(generate_hierarchy=False, stack=args.stack_name, resol='x5', section=section_id)
    dm._load_image()

    # Load image and relevant data
    im_height, im_width = imread(dm._get_image_filepath()).shape[:2]

    texton_hists = dm.load_pipeline_result('texHist', 'npy')
    segmentation = dm.load_pipeline_result('segmentation', 'npy')
    n_superpixels = len(np.unique(segmentation)) - 1
    textonmap = dm.load_pipeline_result('texMap', 'npy')
    n_texton = len(np.unique(textonmap)) - 1
    neighbors = dm.load_pipeline_result('neighbors', 'pkl')
    sp_properties = dm.load_pipeline_result('spProps', 'npy')
    segmentation_vis = dm.load_pipeline_result('segmentationWithText', 'jpg')

    # Load region proposals
    expansion_clusters_tuples = dm.load_pipeline_result('clusters', 'pkl')
    expansion_clusters, expansion_cluster_scores = zip(*expansion_clusters_tuples)
    expansion_cluster_scores = np.array(expansion_cluster_scores)

    neighbors_dict = dict(zip(np.arange(n_superpixels), [list(i) for i in neighbors]))
    neighbor_graph = from_dict_of_lists(neighbors_dict)

    surrounds_sps = dm.load_pipeline_result('clusterSurrounds', 'pkl')
    frontiers_sps = dm.load_pipeline_result('clusterFrontiers', 'pkl')
    
    
    # votes for directed edgelets
    dedge_vote_dict = defaultdict(float)

    # Compute the supporter sets of every edgelet, based on region proposals
    # supporter_all[(100,101)] is the set of superpixels that supports directed edgelet (100,101)
    dedge_supporters = defaultdict(list)

    for s in range(n_superpixels):

        c = list(expansion_clusters[s])
        interior_texture = texton_hists[c].mean(axis=0)
        b_sps = surrounds_sps[s]
        b_contrasts = cdist(texton_hists[b_sps], interior_texture[np.newaxis, :], chi2)

        for b_sp, b_contrast in zip(b_sps, b_contrasts):
            int_sps = neighbors[b_sp] & set(expansion_clusters[s])
            for int_sp in int_sps:
                # weight of each edgelet is the contrast normalized by region size
    #             weight = float(b_contrast) / max(len(c), 5)
    #             weight = 1. / max(len(c), 5)
                weight = 1.
                dedge_vote_dict[(b_sp, int_sp)] += weight
    #             dedge_vote_dict[(int_sp, b_sp)] += weight
                dedge_supporters[(b_sp, int_sp)].append(s) # (border_sp, interior_sp) or (out, in)

    dedge_vote_dict.default_factory = None
    dedge_supporters.default_factory = None
    
    
    edge_coords = dict(dm.load_pipeline_result('edgeCoords', 'pkl'))
    edge_neighbors = dm.load_pipeline_result('edgeNeighbors', 'pkl')
    
    
    
    all_edges = edge_coords.keys()
    all_dedges = set(chain(*[[(i,j),(j,i)] for i,j in all_edges]))

    try:
        edge_contained_by = dm.load_pipeline_result('edgeContainedBy', 'pkl')
        print "edgeContainedBy.pkl already exists, skip"

    except:
        cluster_edges = dm.load_pipeline_result('clusterEdges', 'pkl')

        def f(c, e):
            q = set(chain(*[[(i,j),(j,i)] for i,j in combinations(c, 2) if frozenset([i,j]) in all_edges]))
            return q | set(e)

        contain_edges = Parallel(n_jobs=16)(delayed(f)(c,e) for c, e in zip(expansion_clusters, cluster_edges))

        edge_contained_by = defaultdict(set)
        for sp, es in enumerate(contain_edges):
            for e in es:
                edge_contained_by[e].add(sp)

        edge_contained_by.default_factory = None

        dm.save_pipeline_result(edge_contained_by, 'edgeContainedBy', 'pkl')
    
    
    nz_dedges = dedge_vote_dict.keys()
    
    # dedge_contrast = dict([((i,j), chi2(texton_hists[i], texton_hists[j])) for i,j in all_dedges])
    dedge_contrast = dict([((i,j), chi2(texton_hists[i], texton_hists[dedge_supporters[(i,j)]].mean(axis=0))) 
                           for i,j in nz_dedges])
    
    nz_dedges2 = [e for e,sps in edge_contained_by.iteritems() if len(sps) > 3 and e in nz_dedges]
    nz_dedges2 = [e for e in nz_dedges2 if dedge_contrast[e] > .5]
    dedge_stopperness = dict([(e, dedge_vote_dict[e]/len(edge_contained_by[e])) for e in nz_dedges2])
    nz_dedges2 = [e for e in nz_dedges2 if dedge_stopperness[e] == 1.]
#     print len(nz_dedges2), 'valid edges'
    
    
    dedge_expandedSupporters = dict([(e, smart_union([expansion_clusters[s] for s in dedge_supporters[e]])) 
                                 for e in nz_dedges2])

    dedges_grouped, dedge_supporters_grouped, _ = group_clusters(clusters=dict((e, dedge_expandedSupporters[e]) for e in nz_dedges2),
                                                                 dist_thresh=.01, linkage='complete', metric='jaccard')

#     print len(dedges_grouped), 'edge groups'
    
    dedge_group_supporters = map(smart_union, dedge_supporters_grouped)
    dedge_group_supporter_distmat = compute_pairwise_distances(dedge_group_supporters, metric='jaccard')
    
    np.fill_diagonal(dedge_group_supporter_distmat, 0)

    dedge_vectors = dm.load_pipeline_result('edgeVectors', 'pkl')
    dedge_neighbors = dm.load_pipeline_result('dedgeNeighbors', 'pkl')
        
    import networkx as nx
    G = nx.from_dict_of_lists(dedge_neighbors)
    dedges_grouped = map(set, dedges_grouped)
    ngroup = len(dedges_grouped)

    conns = [[set() if any([sorted(e1)==sorted(e2) for e1, e2 in product(eg1, eg2)]) 
              else set([(i,j) for i,j in G.edges(eg1|eg2) if (i in eg1 and j in eg2) or (j in eg1 and i in eg2)]) 
             for eg1 in dedges_grouped] for eg2 in dedges_grouped]
    conns_flat = [a for b in conns for a in b ]
    
    dedge_group_supporterTex_distmat = np.reshape([chi2(texton_hists[list(sps1)].mean(axis=0), texton_hists[list(sps2)].mean(axis=0))  
 for sps1, sps2 in product(dedge_group_supporters, dedge_group_supporters)], (ngroup, ngroup))
    
    dedge_group_edgeConn_distmat = np.reshape(map(lambda x: len(x) < 1, conns_flat), (ngroup, ngroup))
    np.fill_diagonal(dedge_group_edgeConn_distmat, 0)

   
    dedge_group_distmat = 1 - (1-dedge_group_edgeConn_distmat) * (1-dedge_group_supporter_distmat>0.1) * (dedge_group_supporterTex_distmat < .25)
    
    _, edge_groups, _ = group_clusters(clusters=dedges_grouped, 
                                   distance_matrix=dedge_group_distmat, 
                                   dist_thresh=.5, linkage='single')

#     print len(edge_groups), 'edge groups after considering connectivity'

    edge_groups = map(lambda x: set(chain(*x)), edge_groups)
    
    edge_groups_sorted = sorted(edge_groups, key=lambda x: sum(dedge_contrast[e] for e in x), reverse=True)
    edge_group_supporters_sorted = [smart_union(map(lambda e: dedge_expandedSupporters[e], es)) 
                                    for es in edge_groups_sorted]
    
    
    viz = dm.visualize_edge_sets(edge_groups_sorted[:40], text_size=3, img=segmentation_vis)
    dm.save_pipeline_result(viz, 'topLandmarks', 'jpg')
    
    dm.save_pipeline_result(edge_groups_sorted, 'goodEdgeSets', 'pkl')
    dm.save_pipeline_result(edge_group_supporters_sorted, 'goodEdgeSetsSupporters', 'pkl')
    
    
    boundary_models = []

    for i, es in enumerate(edge_groups_sorted[:40]):
        
        es = list(es)
        
        interior_texture = texton_hists[list(edge_group_supporters_sorted[i])].mean(axis=0)

        surrounds = [e[0] for e in es]
        exterior_textures = np.array([texton_hists[s] if s!=-1 else np.nan * np.ones((texton_hists.shape[1],)) 
                                      for s in surrounds])
        # how to deal with -1 in surrounds? Assign to an all np.nan vector

        points = np.array([edge_coords[frozenset(e)].mean(axis=0) for e in es])
        center = points.mean(axis=0)

        boundary_models.append((es, interior_texture, exterior_textures, points, center))

    dm.save_pipeline_result(boundary_models, 'boundaryModels', 'pkl')
    