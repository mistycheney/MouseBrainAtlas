import sys
sys.path.append('/home/yuncong/project/opencv-2.4.9/release/lib/python2.7/site-packages')

sys.path.insert(0, '/home/yuncong/project/cython-munkres-wrapper/build/lib.linux-x86_64-2.7')
from munkres import munkres

sys.path.append('/home/yuncong/Brain/pipeline_scripts')
import utilities2014
reload(utilities2014)
from utilities2014 import *

import os
import time
import random

# os.environ['GORDON_DATA_DIR'] = '/home/yuncong/project/DavidData2014tif/'
# os.environ['GORDON_REPO_DIR'] = '/home/yuncong/Brain'
# os.environ['GORDON_RESULT_DIR'] = '/home/yuncong/project/DavidData2014results/'
# os.environ['GORDON_LABELING_DIR'] = '/home/yuncong/project/DavidData2014labelings/'

from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import average, fcluster, leaders, complete, single, dendrogram

from joblib import Parallel, delayed

from skimage.color import gray2rgb
from skimage.util import img_as_float, pad
from skimage.morphology import disk
from skimage.filter.rank import gradient

from sklearn.preprocessing import normalize

from collections import defaultdict, Counter
from itertools import combinations, chain, product

import networkx
from networkx import from_dict_of_lists, dfs_postorder_nodes

import matplotlib.pyplot as plt


def boundary_distance(b1, b2, sc1=None, sc2=None, loc_thresh=1500):
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
    
    _, interior_texture1, exterior_textures1, points1, center1 = b1
    _, interior_texture2, exterior_textures2, points2, center2 = b2
    if sc1 is not None:
        assert len(sc1) == points1.shape[0], 'number mismatch %d %d'%(len(sc1), points1.shape[0])
    
    if sc2 is not None:
        assert len(sc2) == points2.shape[0], 'number mismatch %d %d'%(len(sc2), points2.shape[0])

    # compute location difference
    d_loc = np.linalg.norm(center1 - center2)
    D_loc = np.maximum(0, d_loc - 500)

    if d_loc > loc_thresh:
        return np.inf, np.inf, np.inf, np.inf, np.inf
    
    # if the sizes of two point sets vary too much, discard
    n1 = len(points1)
    n2 = len(points2)
    if max(n1,n2) > min(n1,n2) * 3:
        return np.inf, np.inf, np.inf, np.inf, np.inf
    
    # compute interior texture difference
    D_int = chi2(interior_texture1, interior_texture2)

    # compute shape difference, exterior texture difference
#     b = time.time()
    D_shape, matches = shape_context_score(points1, points2, descriptor1=sc1, descriptor2=sc2)
#     print 'shape_context_score', time.time() - b
    
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
                if OUTPUT:
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
        
    D_shape = D_shape * .004

    # weighted average of four terms
    d = D_int + D_ext + D_shape + 0 * D_loc
    
    return d, D_int, D_ext, D_shape, D_loc



def centering(ps):
    return ps - ps.mean(axis=0)

def hausdorff(ps1, ps2, center=False):
    if center:
        ps1 = centering(ps1)
        ps2 = centering(ps2)
    D = cdist(ps1, ps2)
    return max(np.max(D.min(axis=0)), np.max(D.min(axis=1)))

def hausdorff_histograms(h1s, h2s, metric):
    Ds = cdist(h1s, h2s, metric)
    return max(np.nanmax(np.nanmin(Ds, axis=1)), np.nanmax(np.nanmin(Ds, axis=0)))


def rigid_transform_from_pairs(X,Y):
    Xcentroid = X.mean(axis=0)
    Ycentroid = Y.mean(axis=0)
    
    Xcentered = X - Xcentroid
    Ycentered = Y - Ycentroid
    
    U, S, V = np.linalg.svd(np.dot(Xcentered.T, Ycentered))
    M = [[1, 0], [0, np.linalg.det(np.dot(V, U.T))]]
    R = np.dot(V, np.dot(M, U.T))
    angle = np.arctan2(R[1,0], R[0,0])
    t = Ycentroid.T - np.dot(R, Xcentroid.T)
    
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = t
   
    return T, angle

def rigid_transform_to(pts1, T):
    pts1_trans = np.dot(T, np.column_stack([pts1, np.ones((pts1.shape[0],))]).T).T
    pts1_trans = pts1_trans[:,:2]/pts1_trans[:,-1][:,np.newaxis]
    return pts1_trans


from scipy.misc import comb

def ransac_compute_rigid_transform(Dm, pts1, pts2, confidence_thresh=.01, ransac_iters=20, sample_size=5,
                                  matching_iter=10, n_neighbors=10):

#     q = time.time()
    
    high_confidence_thresh = np.sort(Dm.flat)[int(confidence_thresh * np.size(Dm))]
#     print 'high_confidence_thresh', high_confidence_thresh
    
    N1 = len(pts1)
    N2 = len(pts2)
    
    rs, cs = np.where(Dm < high_confidence_thresh)
    high_confidence_pairs = np.c_[rs,cs]
    
    if OUTPUT:
        print 'high_confidence_pairs', high_confidence_pairs
    
#     from itertools import combinations
#     possible_samples = list(combinations(high_confidence_pairs, sample_size))
#     random.shuffle(possible_samples)
    
#     n_possible_samples = len([t for t in combinations(high_confidence_pairs, sample_size) 
#                         if allunique([tt[0] for tt in t]) and allunique([tt[1] for tt in t])])
#     print 'n_possible_samples', len(possible_samples)
#     random.shuffle(possible_samples)

#     print 'comb', time.time() - q

#     return
    
    p1s = np.sort(list(set(rs)))
    p2s = np.sort(list(set(cs)))
    n1 = len(p1s)
    n2 = len(p2s)
    
    offsets = []
    scores = []
    matches_list = []
    samples_list = []
    
    sample_counter = 0
    n_possible_samples = comb(len(high_confidence_pairs), sample_size, exact=False)
#     n_possible_samples = len(possible_samples)
    for ri in range(min(ransac_iters, n_possible_samples)):
        
        samples = []
        
        while True:
#         for tt in range(10):
#             s = possible_samples[sample_counter]
            s = random.sample(high_confidence_pairs, sample_size)
            sample_counter += 1
            w1, w2 = zip(*s)
            if len(set(w1)) == len(w1) and len(set(w2)) == len(w2):
                samples = s
                break

#         samples = np.array(possible_samples[ri])

        if OUTPUT:
            print '\nsamples', ri, samples
        
        X = pts1[[s[0] for s in samples]]
        Y = pts2[[s[1] for s in samples]]
                
        # generate transform hypothesis
        T, angle = rigid_transform_from_pairs(X, Y)
        if np.abs(angle) > np.pi/2:
            if OUTPUT:
                print 'angle too wide', np.rad2deg(angle)
            continue
        
        # apply transform hypothesis
        pts1_trans = rigid_transform_to(pts1, T)
        
        # iterative closest point association
        matches = None
        matches_prev = None
        
        for mi in range(matching_iter):
  
            # given transform, find matching

#             t1 = time.time()
        
#             b = time.time()
    
            Dh = cdist(pts1_trans, pts2, metric='euclidean')
            Dargmin1 = Dh.argsort(axis=1)
            Dargmin0 = Dh.argsort(axis=0)
#             print 'cdist', time.time() - b
        
#             b = time.time()
            
            D2 = Dh.copy()
            D2[np.arange(N1)[:,np.newaxis], Dargmin1[:,n_neighbors:]] = 999
            D2[Dargmin0[n_neighbors:,:], np.arange(N2)[np.newaxis,:]] = 999
            D_hc_pairs = D2[p1s[:,np.newaxis], p2s]
                
#             D_hc_pairs = 9999 * np.ones((n1, n2))
#             for i,j in high_confidence_pairs:
#                 if j in Dargmin1[i,:10] and i in Dargmin0[:10,j]:
#                     ii = p1s.index(i)
#                     jj = p2s.index(j)
#                     D_hc_pairs[ii, jj] = Dh[i,j]

#             print 'D_hc_pairs', time.time() - b

            if matches is not None:
                matches_prev = matches
            
#             b = time.time()
            matches_hc_pairs = np.array(zip(*np.nonzero(munkres(D_hc_pairs))))
#             print 'munkres', time.time() - b, mi
            
#             b = time.time()

#                 print [(p1s[ii], p2s[jj]) for (ii,jj) in matches_hc_pairs]
            matches = np.array([(p1s[ii], p2s[jj]) for (ii,jj) in matches_hc_pairs
                                if D_hc_pairs[ii, jj] != 999])
            # some 9999 edges will be included, the "if" above removes them
#             print 'matches', time.time() - b
        
            if OUTPUT:
#                 print 'considered pairs', w
#                 print 'matches', [(i,j) for i,j in matches
                q1, q2 = np.where(D_hc_pairs < 99)
                w = zip(*[p1s[q1], p2s[q2]])
                print 'matches', len(matches), '/', 'considered pairs', len(w), '/', 'all hc pairs', len(high_confidence_pairs)

#             t2 = time.time()
            
            if len(matches) < 3:
                s = np.inf
                break
            else:
                xs1 = pts1_trans[matches[:,0], 0]
                x_coverage1 = float(xs1.max() - xs1.min()) / (pts1_trans[:,0].max() - pts1_trans[:,0].min())
                ys1 = pts1_trans[matches[:,0], 1]
                y_coverage1 = float(ys1.max() - ys1.min()) / (pts1_trans[:,1].max() - pts1_trans[:,1].min())
                
                xs2 = pts2[matches[:,1], 0]
                x_coverage2 = float(xs2.max() - xs2.min())/ (pts2[:,0].max() - pts2[:,0].min())
                ys2 = pts2[matches[:,1], 1]
                y_coverage2 = float(ys2.max() - ys2.min())/ (pts2[:,1].max() - pts2[:,1].min())
                
                coverage = .5 * x_coverage1 * y_coverage1 + .5 * x_coverage2 * y_coverage2
                
                s = Dh[matches[:,0], matches[:,1]].mean() / coverage    
#             s = .5 * Dm[Dh.argmin(axis=0), np.arange(len(pts2))].mean() + .5 * Dm[np.arange(len(pts1)), Dh.argmin(axis=1)].mean()            
#             s = np.mean([np.mean(Dh.min(axis=0)), np.mean(Dh.min(axis=1))])
    
            X = pts1[matches[:,0]]
            Y = pts2[matches[:,1]]

            T, angle = rigid_transform_from_pairs(X, Y)
            if np.abs(angle) > np.pi/2:
                break

            pts1_trans = rigid_transform_to(pts1, T)
            
            if matches_prev is not None and all([(i,j) in matches_prev for i,j in matches]):
                break
                
#             print 'coverage and remaining', mi, time.time() - t2
        
#             print mi, time.time() - t1
                
                
        samples_list.append(samples)
        offsets.append(T)
        matches_list.append(matches)
        scores.append(s)
    
#         print matches
#         print s
#         plot_two_pointsets(pts1_trans[:,::-1]*np.array([1,-1]), pts2[:,::-1]*np.array([1,-1]), 
#                    center1=False, center2=False,
#                    text=True, matchings=matches)
            
    if len(scores) > 0:
        best_i = np.argmin(scores)

        best_score = scores[best_i]
        best_T = offsets[best_i]
        best_sample = samples_list[best_i]
        best_matches = matches_list[best_i]    
    
        return best_T, best_matches, best_sample, best_score
    else:
        return None, [], None, np.inf


def shape_context_score(pts1, pts2, descriptor1=None, descriptor2=None):

    if descriptor1 is None:
        descriptor1 = compute_shape_context_descriptors(pts1, dist_limit=.8)
    
    if descriptor2 is None:
        descriptor2 = compute_shape_context_descriptors(pts2, dist_limit=.8)
        
    descriptor_dists = cdist(descriptor1, descriptor2, metric='euclidean')
        
#     b = time.time()
    T, best_match, best_sample, best_score = ransac_compute_rigid_transform(descriptor_dists, pts1, pts2, 
                                                                            ransac_iters=50, confidence_thresh=0.03, 
                                                                           sample_size=3, matching_iter=10)
#     print 'ransac_compute_rigid_transform', time.time() - b

    
    if T is None and len(best_match)==0:
        return np.inf, []
    
    if OUTPUT:
        print 'best_match', best_match
        print 'best_sample', best_sample
        print 'best_score', best_score

    return best_score, best_match
    

def euclidean_dists_angles(points):
    """Returns symmetric pairwise ``dists`` and ``angles`` arrays."""
    
    n = len(points)
    dists = squareform(pdist(points, 'euclidean'))

    xd = -np.subtract.outer(points[:,0], points[:,0])
    yd = -np.subtract.outer(points[:,1], points[:,1])
    angles = np.arctan2(yd,xd)

    closest_neighbor = dists.argsort(axis=1)[:,1]
    tangent_vectors = points[closest_neighbor] - points
    tangent_angles = np.arctan2(tangent_vectors[:,1], tangent_vectors[:,0])

    angles = angles - tangent_angles[:, np.newaxis]
    angles = angles%(2*np.pi)
    angles[angles > np.pi] = angles[angles > np.pi] - 2*np.pi

    return dists, angles

def compute_r_theta_bins(n_radial_bins=5, n_polar_bins=12, dist_limit=1.):
    """
    Generate radius-theta bins for the shape context histogram.
    
    Args:
        n_radial_bins (int): number of radial bins
        n_polar_bins (int): number of polar bins
        dist_limit (float): between 0 and 1

    Returns:
        (float array, float array): (r_array, theta_array)

    """
        
    base = 10.    
    r_array = dist_limit * (np.logspace(0, 1, n_radial_bins + 1, base=10) - 1)[1:] / (base-1)
    theta_array = np.linspace(-np.pi, np.pi, n_polar_bins + 1)[1:]
    
    return r_array, theta_array

def compute_shape_context_descriptors(pts, n_radial_bins=5, n_polar_bins=12, 
                                      dist_limit=1., density=True):

    dists, angles = euclidean_dists_angles(pts)
    n_points = dists.shape[0]
    
    r_array, theta_array = compute_r_theta_bins(n_radial_bins, n_polar_bins, dist_limit)
    
    descriptors_mat = np.zeros((n_points, n_radial_bins, n_polar_bins), dtype=np.int)

    dists = dists / dists.max()

    for i in range(n_points):
        for j in range(i + 1, n_points):
            
            if dists[i, j] >= r_array[-1]:
                continue

            r_idx = np.searchsorted(r_array, dists[i, j])
            theta_idx = np.searchsorted(theta_array, angles[i, j])

            descriptors_mat[i, r_idx, theta_idx] += 1

            theta_idx = np.searchsorted(theta_array, angles[j, i])
            descriptors_mat[j, r_idx, theta_idx] += 1

    descriptors = descriptors_mat.reshape(descriptors_mat.shape[0], -1)

    if density:
        descriptors = normalize(descriptors.astype(np.float), axis=1)
    
    return descriptors


def generate_matching_visualizations(stack_name, sec1, sec2):
    '''
    Generate visualization for matching between sec1 and sec2
    '''

    dm1 = DataManager(generate_hierarchy=False, stack=stack_name, resol='x5', section=sec1)
    dm2 = DataManager(generate_hierarchy=False, stack=stack_name, resol='x5', section=sec2)
    
    boundaries1 = dm1.load_pipeline_result('boundaryModels', 'pkl')
    boundaries2 = dm2.load_pipeline_result('boundaryModels', 'pkl')    
    
    matchings = dm1.load_pipeline_result('matchings%dWith%d'%(sec1, sec2), 'pkl')

    matched_boundaries1 = [boundaries1[i][0] for ind, (d,i,j) in enumerate(matchings)]
    vis_matched_boundaries_next = dm1.visualize_edge_sets(matched_boundaries1, text=True)
#     dm1.save_pipeline_result(vis_matched_boundaries_next, 'matchedBoundaries%dWith%d'%(sec1, sec2), 'jpg')

    matched_boundaries2 = [boundaries2[j][0] for ind, (d,i,j) in enumerate(matchings)]
    vis_matched_boundaries_prev = dm2.visualize_edge_sets(matched_boundaries2, text=True)
#     dm2.save_pipeline_result(vis_matched_boundaries_prev, 'matchedBoundaries%dWith%d'%(sec2, sec1), 'jpg')

    
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
    
    dm1.save_pipeline_result(vis, 'matchedBoundaries%dWith%dSideBySide'%(sec1, sec2), 'jpg')
    dm2.save_pipeline_result(vis, 'matchedBoundaries%dWith%dSideBySide'%(sec2, sec1), 'jpg')
    
    
#     for ind, (s, i, j) in enumerate(matchings):
#         # image with a particular boundary highlighted
        
#         vis1 = dm1.load_pipeline_result('boundary%02d'%i, 'jpg')
#         vis2 = dm2.load_pipeline_result('boundary%02d'%j, 'jpg')
        
#         h1, w1 = vis1.shape[:2]
#         h2, w2 = vis2.shape[:2]
        
#         # Place vertically
#         if w1 < w2:
#             left_margin = int((w2 - w1)/2)
#             right_margin = w2 - w1 - left_margin
#             vis1 = pad(vis1, ((0,0),(left_margin,right_margin),(0,0)), 'constant', constant_values=255)
#         else:
#             left_margin = int((w1 - w2)/2)
#             right_margin = w1 - w2 - left_margin
#             vis2 = pad(vis2, ((0,0),(left_margin,right_margin),(0,0)), 'constant', constant_values=255)

#         vis_pair_prev = np.r_[vis1, vis2]
#         vis_pair_next = np.r_[vis2, vis1]
        
#         dm1.save_pipeline_result(vis_pair_prev, 'matchedBoundaryPair%dFor%dWith%d'%(ind, sec1, sec2), 'jpg')
#         dm2.save_pipeline_result(vis_pair_next, 'matchedBoundaryPair%dFor%dWith%d'%(ind, sec2, sec1), 'jpg')
    
    
def compute_shape_context_descriptors_section(stack_name, sec_ind):
    dm = DataManager(generate_hierarchy=False, stack=stack_name, resol='x5', section=sec_ind)
    boundaries = dm.load_pipeline_result('boundaryModels', 'pkl')
    descs = [compute_shape_context_descriptors(b[3], dist_limit=.8) for b in boundaries]
    dm.save_pipeline_result(descs, 'shapeContext', 'pkl')
#     return descs

def compute_boundary_distances(stack_name, sec1, sec2):
    
    dm1 = DataManager(generate_hierarchy=False, stack=stack_name, resol='x5', section=sec1)
    dm2 = DataManager(generate_hierarchy=False, stack=stack_name, resol='x5', section=sec2)
    
    boundaries1 = dm1.load_pipeline_result('boundaryModels', 'pkl')
    boundaries2 = dm2.load_pipeline_result('boundaryModels', 'pkl')
    
    sc1 = dm1.load_pipeline_result('shapeContext', 'pkl')
    sc2 = dm2.load_pipeline_result('shapeContext', 'pkl')

    n_boundaries1 = len(boundaries1)
    n_boundaries2 = len(boundaries2)
    
#     Ds = Parallel(n_jobs=16)(delayed(boundary_distance)(boundaries1[i], boundaries2[j], sc1=sc1[i], sc2=sc2[j]) 
#                              for i, j in product(range(n_boundaries1), range(n_boundaries2)))

    center_dist_thresh = 1500
    
    centers1 = [b[4] for b in boundaries1]
    centers2 = [b[4] for b in boundaries2]
    center_distances = cdist(centers1, centers2, metric='euclidean')
    b1s, b2s = np.where(center_distances < center_dist_thresh)

    Ds = Parallel(n_jobs=16)(delayed(boundary_distance)(boundaries1[i], boundaries2[j], sc1=sc1[i], sc2=sc2[j]) 
                             for i, j in zip(b1s, b2s))
    
    D_boundaries = np.inf * np.ones((n_boundaries1, n_boundaries2))
    D_int = np.inf * np.ones((n_boundaries1, n_boundaries2))
    D_ext = np.inf * np.ones((n_boundaries1, n_boundaries2))
    D_shape = np.inf * np.ones((n_boundaries1, n_boundaries2))

#     D_boundaries[b1s, b2s] = [d for d, d_int, d_ext, d_shape, d_loc in Ds]
#     D_int[b1s, b2s] = [d_int for d, d_int, d_ext, d_shape, d_loc in Ds]
#     D_ext[b1s, b2s] = [d_ext for d, d_int, d_ext, d_shape, d_loc in Ds]
#     D_shape[b1s, b2s] = [d_shape for d, d_int, d_ext, d_shape, d_loc in Ds]
    
    dm1.save_pipeline_result(D_boundaries, 'DBoundaries%dWith%d'%(sec1, sec2), 'npy')
    dm1.save_pipeline_result(D_int, 'D1s%dWith%d'%(sec1, sec2), 'npy')
    dm1.save_pipeline_result(D_ext, 'D2s%dWith%d'%(sec1, sec2), 'npy')
    dm1.save_pipeline_result(D_shape, 'D3s%dWith%d'%(sec1, sec2), 'npy')

#     dm2.save_pipeline_result(D_boundaries.T, 'DBoundaries%dWith%d'%(sec2, sec1), 'npy')
#     dm2.save_pipeline_result(D_int.T, 'D1s%dWith%d'%(sec2, sec1), 'npy')
#     dm2.save_pipeline_result(D_ext.T, 'D2s%dWith%d'%(sec2, sec1), 'npy')
#     dm2.save_pipeline_result(D_shape.T, 'D3s%dWith%d'%(sec2, sec1), 'npy')


def stable(rankings, A, B):
    partners = dict((a, (rankings[(a, 1)], 1)) for a in A)
    is_stable = False # whether the current pairing (given by `partners`) is stable
    while is_stable == False:
        is_stable = True
        for b in B:
            is_paired = False # whether b has a pair which b ranks <= to n
            for n in range(1, len(B) + 1):
                a = rankings[(b, n)]
                a_partner, a_n = partners[a]
                if a_partner == b:
                    if is_paired:
                        is_stable = False
                        partners[a] = (rankings[(a, a_n + 1)], a_n + 1)
                    else:
                        is_paired = True
    return sorted((a, b) for (a, (b, n)) in partners.items())


def stable_marriage_matchings(D_boundaries):

    A = ['A'+str(i) for i in range(n_boundaries1)]
    B = ['B'+str(i) for i in range(n_boundaries2)]

    ao = np.zeros(D_boundaries)
    for q in range(n_boundaries1):
        ao[q, np.argsort(D_boundaries[q,:])] = np.arange(n_boundaries2)

    bo = np.zeros_like(D_boundaries.T)
    for q in range(n_boundaries2):
        bo[q, np.argsort(D_boundaries[:,q])] = np.arange(n_boundaries1)

    rankings1 = zip(A, ao+1)
    rankings2 = zip(B, bo+1)
    rank = dict(rankings1+rankings2)

    Arankings = dict(((a, rank[a][b_]), B[b_]) for (a, b_) in product(A, range(n_boundaries2)))
    Brankings = dict(((b, rank[b][a_]), A[a_]) for (b, a_) in product(B, range(n_boundaries1)))
    rankings = Arankings
    rankings.update(Brankings)

    m = stable(rankings, A, B)
    m = [(int(i[1:]), int(j[1:])) for i, j in m]
    m = sorted([(D_boundaries[i,j], i,j) for i,j in m if D_boundaries[i,j]<np.inf])

#     for s, i,j in m:
#         print s, i, j
    
    return m

def greedy_matching(D_boundaries, thresh_percentage=.2, verified_matchings=[], excluded_matchings=[]):
    
    Dnz = D_boundaries[D_boundaries < np.inf]
    if len(Dnz) == 0:
        return []
    
    th = np.sort(Dnz.flat)[int(len(Dnz.flat)*thresh_percentage)]
    print 'thresh', th
    
    matchings = [(0,i,j) for i,j in verified_matchings]
    rs, cs = np.unravel_index(np.argsort(D_boundaries.flat), D_boundaries.shape)
    for r, c in zip(rs, cs):
        if (r,c) in excluded_matchings:
            continue
        if D_boundaries[r,c] > th:
            break
        if r not in [i for d,i,j in matchings] and c not in [j for d,i,j in matchings]:
            matchings.append((D_boundaries[r,c],r,c))
            
    return matchings

def match_landmarks(sec1, sec2, must_match=[], cannot_match=[]):
    
    dm = DataManager(generate_hierarchy=False, stack=stack_name, resol='x5', section=sec_ind)
#     boundaries = dm.load_pipeline_result('boundaryModels', 'pkl')
    D_boundaries = dm.load_pipeline_result('DBoundaries%dWith%d'%(sec1, sec2), 'npy')
    
    matchings = greedy_matching(D_boundaries, verified_matchings=must_match, excluded_matchings=cannot_match)
    if len(matchings) > 0:
        dm.save_pipeline_result(matchings, 'matchings%dWith%d'%(sec1, sec2), 'pkl')
        dm.save_pipeline_result([(d,j,i) for d,i,j in matchings], 'matchings%dWith%d'%(sec2, sec1), 'pkl')
        
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
    
    sec_ind = args.slice_ind
    
    OUTPUT = False

    from subprocess import check_output
    s = check_output("ssh gordon.sdsc.edu ls %s" % os.path.join(os.environ['GORDON_DATA_DIR'], args.stack_name, 'x5'), shell=True)
    slide_indices = [int(f) for f in s.split('\n') if len(f) > 0]
    n_slice = len(slide_indices)
    last_sec = max(slide_indices)
        
    if sec_ind < last_sec:
        compute_boundary_distances(args.stack_name, sec_ind, sec_ind+1)
#     if sec_ind < last_sec-1:
#         compute_boundary_distances(args.stack_name, sec_ind, sec_ind+2)
#     if sec_ind < last_sec-2:
#         compute_boundary_distances(args.stack_name, sec_ind, sec_ind+3)
        
#     must_matches = []

#     must_match_sections = defaultdict(list)
#     for m in must_matches:
#         mm = sorted(list(m))
#         must_match_sections[(mm[0][0], mm[1][0])].append((mm[0][1], mm[1][1]))
    
#     if sec_ind < last_sec:
#         match_landmarks(args.stack_name, sec_ind, sec_ind+1, must_match=must_match_sections[(sec_ind, sec_ind+1)])
#     if sec_ind < last_sec-1:
#         match_landmarks(args.stack_name, sec_ind, sec_ind+2, must_match=must_match_sections[(sec_ind, sec_ind+2)])
#     if sec_ind < last_sec-2:
#         match_landmarks(args.stack_name, sec_ind, sec_ind+3, must_match=must_match_sections[(sec_ind, sec_ind+3)])
        
#     Parallel(n_jobs=16)(delayed(generate_matching_visualizations)(args.stack_name, sec_ind, sec_ind+1) for sec_ind in range(n_slice-1))
#     Parallel(n_jobs=16)(delayed(generate_matching_visualizations)(args.stack_name, sec_ind, sec_ind+2) for sec_ind in range(n_slice-2))
#     Parallel(n_jobs=16)(delayed(generate_matching_visualizations)(args.stack_name, sec_ind, sec_ind+3) for sec_ind in range(n_slice-3))

