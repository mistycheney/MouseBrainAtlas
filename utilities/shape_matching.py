from scipy.misc import comb
import random
import numpy as np
from scipy.spatial.distance import cdist

import sys
sys.path.insert(0, '/home/yuncong/project/cython-munkres-wrapper/build/lib.linux-x86_64-2.7')
from munkres import munkres

import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist, pdist, squareform

def rigid_transform_from_pairs(X,Y):
    '''
    X, Y are n-by-2 matrices
    '''
    
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

    descriptors = descriptors_mat.reshape(descriptors_mat.shape[0], -1).astype(np.float)

    if density:
        descriptors = descriptors/np.sum(descriptors, axis=1)[:,None]
    
    return descriptors

def ransac_compute_rigid_transform(Dm, pts1, pts2, confidence_thresh=.01, ransac_iters=20, sample_size=5,
                                  matching_iter=10, n_neighbors=10, verbose=False):

#     q = time.time()
    
    high_confidence_thresh = np.sort(Dm.flat)[int(confidence_thresh * np.size(Dm))]
#     print 'high_confidence_thresh', high_confidence_thresh
    
    N1 = len(pts1)
    N2 = len(pts2)
    
    rs, cs = np.where(Dm < high_confidence_thresh)
    high_confidence_pairs = np.c_[rs,cs]
    
    if len(high_confidence_pairs) == 0:
        return None, [], None, np.inf
    
    if verbose:
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
    
    if n1 < sample_size or n2 < sample_size:
        return None, [], None, np.inf
    
    offsets = []
    scores = []
    matches_list = []
    samples_list = []
    
    sample_counter = 0
    n_possible_samples = int(comb(len(high_confidence_pairs), sample_size, exact=False))

    if verbose:
        sys.stderr.write('n_possible_samples = %d\n' % n_possible_samples)
    
#     n_possible_samples = len(possible_samples)
    for ri in range(min(ransac_iters, n_possible_samples)):
#         sys.stderr.write('ri = %d\n' % ri)

        samples = []
        
        for tt in range(100):
#             sys.stderr.write('tt = %d\n' % tt)

#             s = possible_samples[sample_counter]
            s = random.sample(high_confidence_pairs, sample_size)
            sample_counter += 1
            w1, w2 = zip(*s)
            if len(set(w1)) == len(w1) and len(set(w2)) == len(w2):
                samples = s
                break
                
        if len(samples) == 0:
            continue
            
#         samples = np.array(possible_samples[ri])

        if verbose:
            sys.stderr.write('samples = %d\n' % ri)
#             print '\nsamples', ri, samples
        
        X = pts1[[s[0] for s in samples]]
        Y = pts2[[s[1] for s in samples]]
                
        # generate transform hypothesis
        T, angle = rigid_transform_from_pairs(X, Y)
        if np.abs(angle) > np.pi/4:
            if verbose:
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
        
            expanded_matches = []
            matches1 = set([i for i,j in matches])
            matches2 = set([j for i,j in matches])
            rem1 = set(range(N1)) - matches1
            rem2 = set(range(N2)) - matches2
            add1 = set([])
            add2 = set([])
            for i in rem1:
                for j in rem2:
                    if j in Dargmin1[i,:3] and i in Dargmin0[:3,j] and i not in add1 and j not in add2:
                        add1.add(i)
                        add2.add(j)
                        expanded_matches.append((i,j))

            if len(expanded_matches) > 0 and len(matches) > 0 :
                matches = np.vstack([matches, np.array(expanded_matches)])
    
            if verbose:
#                 print 'considered pairs', w
#                 print 'matches', [(i,j) for i,j in matches
                q1, q2 = np.where(D_hc_pairs < 999)
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
                
                s = Dh[matches[:,0], matches[:,1]].mean() / coverage**2    
#             s = .5 * Dm[Dh.argmin(axis=0), np.arange(len(pts2))].mean() + .5 * Dm[np.arange(len(pts1)), Dh.argmin(axis=1)].mean()            
#             s = np.mean([np.mean(Dh.min(axis=0)), np.mean(Dh.min(axis=1))])
    
            X = pts1[matches[:,0]]
            Y = pts2[matches[:,1]]

            T, angle = rigid_transform_from_pairs(X, Y)
            if np.abs(angle) > np.pi/4:
                break

            pts1_trans = rigid_transform_to(pts1, T)
            
            if matches_prev is not None and all([(i,j) in matches_prev for i,j in matches]):
                break
                            
        samples_list.append(samples)
        offsets.append(T)
        matches_list.append(matches)
        scores.append(s)
    
        if verbose:
            print matches
            print s
            plot_two_pointsets(pts1_trans[:,::-1]*np.array([1,-1]), pts2[:,::-1]*np.array([1,-1]), 
                       center1=False, center2=False,
                       text=True, matchings=matches)
            
    if len(scores) > 0:
        best_i = np.argmin(scores)

        best_score = scores[best_i]
        best_T = offsets[best_i]
        best_sample = samples_list[best_i]
        best_matches = matches_list[best_i]    
    
        return best_T, best_matches, best_sample, best_score
    else:
        return None, [], None, np.inf
    
def plot_two_pointsets(pts1, pts2, center1=True, center2=True, text=True, 
                       matchings=None,
                       show_sc1=None, show_sc2=None, r_array=None, n_angles=None):
    '''
    show_sc1 is the point index on which to draw shape context polar histogram boundaries
    '''
    
    pts1 = pts1*np.array([1,-1])
    pts2 = pts2*np.array([1,-1])

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, aspect='equal')
#     ax.scatter(pts1[:,0], pts1[:,1], c='r', label='PointSet 1', s=40)
#     ax.scatter(pts2[:,0], pts2[:,1], c='g', label='PointSet 2', s=40)
    ax.scatter(pts1[:,0], pts1[:,1], c='r', label='PointSet 1', s=5)
    ax.scatter(pts2[:,0], pts2[:,1], c='g', label='PointSet 2', s=5)

    if center1:
        center1 = pts1.mean(axis=0)
        ax.scatter(center1[0], center1[1], c='b')

    if center2:
        center2 = pts2.mean(axis=0)
        ax.scatter(center2[0], center2[1], c='b')

    if text:
        for i, (y,x) in enumerate(pts1):
            ax.text(y, x, str(i), color='r')

        for i, (y,x) in enumerate(pts2):
            ax.text(y, x, str(i), color='g')
    
    if matchings is not None:
        for i,j in matchings:
            ln = plt.Line2D(np.r_[pts1[i,0], pts2[j,0]], np.r_[pts1[i,1], pts2[j,1]], c='k')
            fig.gca().add_artist(ln)
    
    if show_sc1 is not None:
        assert r_array is not None and n_angles is not None
        dists_max1 = pdist(pts1).max() 
        scales1 = dists_max1 * r_array
        
        for s in scales1:
            circle = plt.Circle(pts1[show_sc1], s, color='k', fill=False)
            fig.gca().add_artist(circle)

        angs = np.linspace(np.pi, -np.pi, n_angles+1)[:-1]
        na = len(angs)
        for ai, a in enumerate(angs):
            ln = plt.Line2D(np.r_[pts1[show_sc1,0], pts1[show_sc1,0]+scales1[-1]*np.cos(a)], 
                            np.r_[pts1[show_sc1,1], pts1[show_sc1,1]+scales1[-1]*np.sin(a)], c='k')
            fig.gca().add_artist(ln)
            fig.gca().text(.5*(pts1[show_sc1,0]+scales1[0]*np.cos(a))+.5*(pts1[show_sc1,0]+scales1[0]*np.cos(angs[(ai+1)%na])), 
                           .5*(pts1[show_sc1,1]+scales1[0]*np.sin(a))+.5*(pts1[show_sc1,1]+scales1[0]*np.sin(angs[(ai+1)%na])), 
                           str(ai),
                           horizontalalignment='center',
                           verticalalignment='center')
            
    if show_sc2 is not None:
        assert r_array is not None and n_angles is not None
        dists_max2 = pdist(pts2).max()
        scales2 = dists_max2 * r_array

        for s in scales2:
            circle = plt.Circle(pts2[show_sc2], s, color='k', fill=False)
            fig.gca().add_artist(circle)
            
        angs = np.linspace(np.pi, -np.pi, n_angles+1)[:-1]
        na = len(angs)
        for ai, a in enumerate(angs):
            ln = plt.Line2D(np.r_[pts2[show_sc2,0], pts2[show_sc2,0]+scales2[-1]*np.cos(a)], 
                            np.r_[pts2[show_sc2,1], pts2[show_sc2,1]+scales2[-1]*np.sin(a)], c='k')
            fig.gca().add_artist(ln)
            fig.gca().text(.5*(pts2[show_sc2,0]+scales2[0]*np.cos(a))+.5*(pts2[show_sc2,0]+scales2[0]*np.cos(angs[(ai+1)%na])), 
               .5*(pts2[show_sc2,1]+scales2[0]*np.sin(a))+.5*(pts2[show_sc2,1]+scales2[0]*np.sin(angs[(ai+1)%na])), 
               str(ai),
               horizontalalignment='center',
               verticalalignment='center')


            
#     ax.legend()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()