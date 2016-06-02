#! /usr/bin/env python

import os
import argparse
import sys
import time

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Match landmarks',
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

dms = dict([(sec_ind, DataManager(data_dir=os.environ['GORDON_DATA_DIR'], 
                 repo_dir=os.environ['GORDON_REPO_DIR'], 
                 result_dir=os.environ['GORDON_RESULT_DIR'], 
                 labeling_dir=os.environ['GORDON_LABELING_DIR'],
                 gabor_params_id=args.gabor_params_id, 
                 segm_params_id=args.segm_params_id, 
                 vq_params_id=args.vq_params_id,
                 stack=args.stack_name, 
                 section=sec_ind))
        for sec_ind in range(args.slice_ind, args.slice_ind+4)])
       
#======================================================

sys.path.insert(0, '/home/yuncong/project/cython-munkres-wrapper/build/lib.linux-x86_64-2.7')
from munkres import munkres

from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import average, fcluster, single, complete

from skimage.color import gray2rgb
from skimage.measure import find_contours
from skimage.util import img_as_float

from networkx import from_dict_of_lists, Graph, adjacency_matrix, dfs_postorder_nodes
from networkx.algorithms import node_connected_component

from shape_matching import *
from bipartite_matching import *


def boundary_distance(b1, b2, sc1=None, sc2=None, loc_thresh=1500, verbose=False):
    '''
    Compute the distance between two boundaries.
    Each tuple consists of (edgeSet, interior_texture, exterior_textures, points, center)
    
    Parameters
    ----------
    b1 : tuple
    b2 : tuple
    sc1 : #points-by-32 array
        pre-computed shape context descriptor
    sc2 : #points-by-32 array
        pre-computed shape context descriptor
    '''
    
    _, interior_texture1, exterior_textures1, points1, center1, \
                        majorv1, minorv1, majorlen1, minorlen1, ell_center1 = b1
        
    _, interior_texture2, exterior_textures2, points2, center2, \
                        majorv2, minorv2, majorlen2, minorlen2, ell_center2 = b2
 
    if sc1 is not None:
        assert len(sc1) == points1.shape[0], 'number mismatch %d %d'%(len(sc1), points1.shape[0])
    
    if sc2 is not None:
        assert len(sc2) == points2.shape[0], 'number mismatch %d %d'%(len(sc2), points2.shape[0])

    # compute location difference
    d_loc = np.linalg.norm(center1 - center2)
    D_loc = np.maximum(0, d_loc - 500)
    
#     print 'd_loc', d_loc

    if d_loc > loc_thresh:
        return np.inf, np.inf, np.inf, np.inf, np.inf
    
    n1 = len(points1)
    n2 = len(points2)
    if max(n1,n2) > min(n1,n2) * 3:
        return np.inf, np.inf, np.inf, np.inf, np.inf
    
    # compute interior texture difference
    D_int = chi2(interior_texture1, interior_texture2)
#     D_ext = hausdorff_histograms(exterior_textures1, exterior_textures2, metric=chi2)

    # compute shape difference, exterior texture difference
    D_shape, matches = shape_context_score(points1, points2, descriptor1=sc1, descriptor2=sc2)
#         D_ext = np.mean([chi2(exterior_textures1[i], exterior_textures2[j]) for i, j in matches])
    
    bg_match = 0

    if len(matches) == 0:
        D_ext = np.inf
    else:
        ddd = []
        for i, j in matches:
            # -1 vs -1
            if np.isnan(exterior_textures1[i]).all() and np.isnan(exterior_textures2[j]).all():
                s = 0
                bg_match += 1
                ddd.append(s)
            # non -1 vs non -1
            elif not np.isnan(exterior_textures1[i]).all() and not np.isnan(exterior_textures2[j]).all():
                s = chi2(exterior_textures1[i], exterior_textures2[j])
                if verbose:
                    print 'exterior', i,j,s
                ddd.append(s)
            # -1 vs non -1
            else:
                ddd.append(2.)

        if len(ddd) == 0:
            D_ext = np.inf
        elif len(ddd) == bg_match:
            D_ext = 2.
        else:
            D_ext = np.mean(ddd)
    
    D_shape = D_shape * .01

    # weighted average of four terms
    d = D_int + D_ext + D_shape + 0 * D_loc
    
    return d, D_int, D_ext, D_shape, D_loc



def shape_context_score(pts1, pts2, descriptor1=None, descriptor2=None, verbose=False):

    if descriptor1 is None:
        descriptor1 = compute_shape_context_descriptors(pts1, dist_limit=.8)
    
    if descriptor2 is None:
        descriptor2 = compute_shape_context_descriptors(pts2, dist_limit=.8)
        
    descriptor_dists = cdist(descriptor1, descriptor2, metric='euclidean')
        
#     b = time.time()

    T, best_match, best_sample, best_score = ransac_compute_rigid_transform(descriptor_dists, pts1, pts2, 
                                                                            ransac_iters=50, confidence_thresh=0.03, 
                                                                            sample_size=3, matching_iter=10,
                                                                           n_neighbors=3)
#     print 'ransac_compute_rigid_transform', time.time() - b

    
    if T is None and len(best_match)==0:
        return np.inf, []
    
    if verbose:
        print 'best_match', best_match
        print 'best_sample', best_sample
        print 'best_score', best_score

    return best_score, best_match


from skimage.util import pad

def generate_matching_visualizations(sec1, sec2, matchings=None):
    '''
    Generate visualization for matching between sec1 and sec2
    '''
    
    dm1 = dms[sec1]
    dm2 = dms[sec2]
    
    boundaries1 = dm1.load_pipeline_result('boundaryModels')
    boundaries2 = dm2.load_pipeline_result('boundaryModels')
    
    if matchings is None:
        matchings = dm1.load_pipeline_result('matchings%dWith%d'%(sec1, sec2))

    matched_boundaries1 = [boundaries1[i][0] for ind, (d,i,j) in enumerate(matchings)]
    vis_matched_boundaries_next = dm1.visualize_edge_sets(matched_boundaries1, show_set_index=True)

    matched_boundaries2 = [boundaries2[j][0] for ind, (d,i,j) in enumerate(matchings)]
    vis_matched_boundaries_prev = dm2.visualize_edge_sets(matched_boundaries2, show_set_index=True)

    # Place two images vertically 
    h1, w1 = vis_matched_boundaries_next.shape[:2]
    h2, w2 = vis_matched_boundaries_prev.shape[:2]
    
    if w1 < w2:
        left_margin = int((w2 - w1)/2)
        right_margin = w2 - w1 - left_margin
        vis_matched_boundaries_next = pad(vis_matched_boundaries_next, 
                                          ((0,0),(left_margin,right_margin),(0,0)), 
                                          'constant', constant_values=255)
    else:
        left_margin = int((w1 - w2)/2)
        right_margin = w1 - w2 - left_margin
        vis_matched_boundaries_prev = pad(vis_matched_boundaries_prev, 
                                          ((0,0),(left_margin,right_margin),(0,0)), 
                                          'constant', constant_values=255)
        
    vis = np.r_[vis_matched_boundaries_next, vis_matched_boundaries_prev]
    
    return vis

def compute_boundary_distances(sec1, sec2, verbose=False):
    
    dm1 = dms[sec1]
    dm2 = dms[sec2]
    
    boundaries1 = dm1.load_pipeline_result('boundaryModels')
    boundaries2 = dm2.load_pipeline_result('boundaryModels')
    
    sc1 = dm1.load_pipeline_result('shapeContext')
    sc2 = dm2.load_pipeline_result('shapeContext')

    n_boundaries1 = len(boundaries1)
    n_boundaries2 = len(boundaries2)
    
#     Ds = Parallel(n_jobs=16)(delayed(boundary_distance)(boundaries1[i], boundaries2[j], sc1=sc1[i], sc2=sc2[j]) 
#                              for i, j in product(range(n_boundaries1), range(n_boundaries2)))

    center_dist_thresh = 1500
    
    centers1 = [b[4] for b in boundaries1]
    centers2 = [b[4] for b in boundaries2]
    center_distances = cdist(centers1, centers2, metric='euclidean')
    b1s, b2s = np.where(center_distances < center_dist_thresh)
    
#     b = time.time()

    Ds = Parallel(n_jobs=16)(delayed(boundary_distance)(boundaries1[i], boundaries2[j], 
                                                        sc1=sc1[i], sc2=sc2[j], verbose=verbose) 
                             for i, j in zip(b1s, b2s))
#     print  'boundary_distance', time.time() - b
    
    D_boundaries = np.inf * np.ones((n_boundaries1, n_boundaries2))
    D_int = np.inf * np.ones((n_boundaries1, n_boundaries2))
    D_ext = np.inf * np.ones((n_boundaries1, n_boundaries2))
    D_shape = np.inf * np.ones((n_boundaries1, n_boundaries2))
    
    D_boundaries[b1s, b2s] = [d for d, d_int, d_ext, d_shape, d_loc in Ds]
    D_int[b1s, b2s] = [d_int for d, d_int, d_ext, d_shape, d_loc in Ds]
    D_ext[b1s, b2s] = [d_ext for d, d_int, d_ext, d_shape, d_loc in Ds]
    D_shape[b1s, b2s] = [d_shape for d, d_int, d_ext, d_shape, d_loc in Ds]
    
    return D_boundaries, D_int, D_ext, D_shape
    
    
def match_landmarks(sec1, sec2, D=None, must_match=[], cannot_match=[]):
    
    dm1 = dms[sec1]
    dm2 = dms[sec2]
    boundaries1 = dm1.load_pipeline_result('boundaryModels')
    boundaries2 = dm2.load_pipeline_result('boundaryModels')
    
    if D is None:
        D = dm1.load_pipeline_result('DBoundaries%dWith%d'%(sec1, sec2))
        
    matchings = knn_matching(D, boundaries1, boundaries2)
    matchings = sorted(matchings)
    
    return matchings


sec1 = args.slice_ind

try:
    sec2 = sec1 + 1
    D_boundaries, D_int, D_ext, D_shape = compute_boundary_distances(sec1, sec2, verbose=False)
    matchings = match_landmarks(sec1, sec2, D=D_boundaries)
    viz = generate_matching_visualizations(sec1, sec2, matchings=matchings)
    dms[sec1].save_pipeline_result(D_boundaries, 'DBoundariesNext1')
    dms[sec1].save_pipeline_result(matchings, 'matchingsNext1')
    dms[sec1].save_pipeline_result(viz, 'matchingsVizNext1')
# dms[sec2].save_pipeline_result(D_boundaries.T, 'DBoundariesPrev1')
# dms[sec2].save_pipeline_result(matchings, 'matchingsPrev1')
# dms[sec2].save_pipeline_result(viz, 'matchingsVizPrev1')
except Exception as e:
    print e
    pass

try:
    sec2 = sec1 + 2
    D_boundaries, D_int, D_ext, D_shape = compute_boundary_distances(sec1, sec2, verbose=False)
    matchings = match_landmarks(sec1, sec2, D=D_boundaries)
    viz = generate_matching_visualizations(sec1, sec2, matchings=matchings)
    dms[sec1].save_pipeline_result(D_boundaries, 'DBoundariesNext2')
    dms[sec1].save_pipeline_result(matchings, 'matchingsNext2')
    dms[sec1].save_pipeline_result(viz, 'matchingsVizNext2')
    # dms[sec2].save_pipeline_result(D_boundaries.T, 'DBoundariesPrev2')
    # dms[sec2].save_pipeline_result(matchings, 'matchingsPrev2')
    # dms[sec2].save_pipeline_result(viz, 'matchingsVizPrev2')
except Exception as e:
    print e
    pass

try:
    sec2 = sec1 + 3
    D_boundaries, D_int, D_ext, D_shape = compute_boundary_distances(sec1, sec2, verbose=False)
    matchings = match_landmarks(sec1, sec2, D=D_boundaries)
    viz = generate_matching_visualizations(sec1, sec2, matchings=matchings)
    dms[sec1].save_pipeline_result(D_boundaries, 'DBoundariesNext3')
    dms[sec1].save_pipeline_result(matchings, 'matchingsNext3')
    dms[sec1].save_pipeline_result(viz, 'matchingsVizNext3')
# dms[sec2].save_pipeline_result(D_boundaries.T, 'DBoundariesPrev3')
# dms[sec2].save_pipeline_result(matchings, 'matchingsPrev3')
# dms[sec2].save_pipeline_result(viz, 'matchingsVizPrev3')
except Exception as e:
    print e
    pass