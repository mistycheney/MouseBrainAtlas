from collections import defaultdict
import time
import numpy as np
from scipy.spatial.distance import cdist

def diff_offset(segm, x_offset, y_offset):

    h, w = segm.shape

    if x_offset == 1 and y_offset == -1:
        d = np.dstack([segm[0:h-1, 1:w], segm[1:h, 0:w-1]])
        # if diff's location y,x is (0,0), the edge is at (1,0) and (0,1)
        ys, xs = np.mgrid[:d.shape[0], :d.shape[1]]
        nzs = ~(d[...,0]==d[...,1])
        r = np.c_[d[nzs], ys[nzs] + 1, xs[nzs], ys[nzs], xs[nzs] + 1] # [sp_label1, sp_label2, y1,x1,y2,x2]        
    else:
        moving_x_low = max(x_offset, 0)
        moving_x_high = min(x_offset + w, w)
        moving_width = moving_x_high - moving_x_low
        moving_y_low = max(y_offset, 0)
        moving_y_high = min(y_offset + h, h)
        moving_height = moving_y_high - moving_y_low

        d = np.dstack([segm[moving_y_low:moving_y_high, moving_x_low:moving_x_high], 
                       segm[:moving_height, :moving_width]])

        ys, xs = np.mgrid[:d.shape[0], :d.shape[1]]
        nzs = ~(d[...,0]==d[...,1])
        # if diff's location y,x is (0,0), the edge is at (0,0) and (y_offset, x_offset)
        r = np.c_[d[nzs], ys[nzs], xs[nzs], ys[nzs] + y_offset, xs[nzs] + x_offset] # [sp_label1, sp_label2, y1,x1,y2,x2]     

    return r


def neighbors_info(segmentation, sp_centroids=None):
    
    n_superpixels = segmentation.max() + 1
    h, w = segmentation.shape

    print 'computing neighbors ...',
    t = time.time()

    # diffs = np.vstack([diff_offset(1,0), diff_offset(0,1), diff_offset(1,1), diff_offset(1,-1)])
    diffs = np.vstack([diff_offset(segmentation, 1,0), diff_offset(segmentation, 0,1)])

    edge_coords = defaultdict(set)
    edge_junctions = defaultdict(set)
    neighbors = [set() for _ in range(n_superpixels)]

    for i, j, y1, x1, y2, x2 in diffs:
        edge_coords[frozenset([i,j])] |= {(x1,y1), (x2,y2)}

        if x1 == x2:
            edge_junctions[frozenset([i,j])] |= {frozenset([(x1,y1),(x2,y2),(x1-1,y1),(x2-1,y2)]),
                                                frozenset([(x1,y1),(x2,y2),(x1+1,y1),(x2+1,y2)])}
        elif y1 == y2:
            edge_junctions[frozenset([i,j])] |= {frozenset([(x1,y1),(x2,y2),(x1,y1-1),(x2,y2-1)]),
                                                frozenset([(x1,y1),(x2,y2),(x1,y1+1),(x2,y2+1)])}
        else:
            edge_junctions[frozenset([i,j])] |= {frozenset([(x1,y1),(x2,y2),(x1,y2),(x2,y1)])}

        if i != -1:
            neighbors[i].add(j)
        if j != -1:
            neighbors[j].add(i)

    edge_coords = dict((e, np.array(list(pts))) for e, pts in edge_coords.iteritems())


    print 'done in', time.time() - t, 'seconds'

    print 'compute edge info ...',
    t = time.time()

    dedge_vectors = {}
    edge_coords_sorted = {}
    edge_midpoints = {}

    for e, pts in edge_coords.iteritems():

        X = pts.astype(np.float)
        c = X.mean(axis=0)
        edge_midpoints[e] = X[np.squeeze(cdist([c], X)).argmin()] # closest point to the centroid
        Xc = X - c
        U,S,V = np.linalg.svd(np.dot(Xc.T, Xc))
        u1 = U[:,0]
        n1 = np.array([-u1[1], u1[0]])

        s1, s2 = e
        if s1 == -1:
            mid_to_s1 = edge_midpoints[e] - sp_centroids[s2, ::-1]
        else:
            mid_to_s1 = sp_centroids[s1, ::-1] - edge_midpoints[e]

        if np.dot(n1, mid_to_s1) > 0:
            dedge_vectors[(s1,s2)] = n1
            dedge_vectors[(s2,s1)] = -n1
        else:
            dedge_vectors[(s2,s1)] = n1
            dedge_vectors[(s1,s2)] = -n1

        projs = np.dot(Xc,u1)
        order = projs.argsort()
        if Xc[order[0],0] > Xc[order[-1],0]:
            order = order[::-1]
        edge_coords_sorted[e] = X[order].astype(np.int)

    print 'done in', time.time() - t, 'seconds'

    edge_coords = edge_coords_sorted

    print 'computing edge neighbors ...',
    t = time.time()

    edge_map = -1 * np.ones_like(segmentation, np.int)

    for ei, pts in enumerate(edge_coords.itervalues()):
        edge_map[pts[:,1], pts[:,0]] = ei

    edges = edge_coords.keys()

    xs, ys = np.mgrid[-1:2, -1:2]

    def compute_edge_neighbors_worker(pts):
        nbrs = set(edge_map[np.maximum(0, np.minimum(h-1, (pts[:,1] + ys[:,:,None]).flat)), 
                            np.maximum(0, np.minimum(w-1, (pts[:,0] + xs[:,:,None]).flat))])
        return nbrs

    edge_neighbors = {}
    for ei, (e, pts) in enumerate(edge_coords.iteritems()):
        nbr_ids = compute_edge_neighbors_worker(pts) - {-1, ei}
        edge_neighbors[e] = set([edges[i] for i in nbr_ids if len(set.intersection(edge_junctions[e], edge_junctions[edges[i]])) > 0])

    print 'done in', time.time() - t, 'seconds'

    print 'compute dedge neighbors ...',
    t = time.time()

    dedge_neighbors = defaultdict(set)
    for edge, nbr_edges in edge_neighbors.iteritems():
        s1, s2 = edge

        for nbr_edge in nbr_edges:
            t1, t2 = nbr_edge

            if s1 == t1 or s2 == t2:
                dedge_neighbors[(s1, s2)].add((t1, t2))
                dedge_neighbors[(t1, t2)].add((s1, s2))
                dedge_neighbors[(s2, s1)].add((t2, t1))
                dedge_neighbors[(t2, t1)].add((s2, s1))      
                continue
            elif s1 == t2 or s2 == t1:
                dedge_neighbors[(s2, s1)].add((t1, t2))
                dedge_neighbors[(t1, t2)].add((s2, s1))
                dedge_neighbors[(s1, s2)].add((t2, t1))
                dedge_neighbors[(t2, t1)].add((s1, s2))
                continue

            ep1 = edge_coords[edge][0]
            ep2 = edge_coords[edge][-1]
            nbr_ep1 = edge_coords[nbr_edge][0]
            nbr_ep2 = edge_coords[nbr_edge][-1]
            endpoints_dists = cdist([ep1, ep2], [nbr_ep1, nbr_ep2])
            ep_ind, nbr_ep_ind = np.unravel_index(endpoints_dists.argmin(), endpoints_dists.shape)
            if ep_ind == 0:
                ep_ind = 0
                ep_inner_ind = min(100, len(edge_coords[edge])-1)
            else:
                ep_ind = -1
                ep_inner_ind = max(-101, -len(edge_coords[edge]))

            if nbr_ep_ind == 0:
                nbr_ep_ind = 0
                nbr_ep_inner_ind = min(100, len(edge_coords[nbr_edge])-1)
            else:
                nbr_ep_ind = -1
                nbr_ep_inner_ind = max(-101, -len(edge_coords[nbr_edge]))

            ep_inner = edge_coords[edge][ep_inner_ind]
            nbr_ep_inner = edge_coords[nbr_edge][nbr_ep_inner_ind]

            junction = .5 * (edge_coords[edge][ep_ind] + edge_coords[nbr_edge][nbr_ep_ind])

            vec_to_junction = junction - .5 * (ep_inner + nbr_ep_inner)

            unit_vec_to_junction = vec_to_junction/np.linalg.norm(vec_to_junction)

            midpoint_to_midpoint = ep_inner - nbr_ep_inner
            midpoint_to_midpoint = midpoint_to_midpoint/np.linalg.norm(midpoint_to_midpoint)
            n_mp_mp = np.array([-midpoint_to_midpoint[1], midpoint_to_midpoint[0]])
            if np.dot(n_mp_mp, unit_vec_to_junction) < 0:
                n_mp_mp = -n_mp_mp

            tang_ep = junction - ep_inner
            n_ep = np.array([-tang_ep[1], tang_ep[0]])
            if np.linalg.norm(n_ep) == 0:
                n_ep = n_ep
            else:
                n_ep = n_ep/np.linalg.norm(n_ep)

            x_ep, y_ep = ep_inner + (5*n_ep).astype(np.int)
            x_ep2, y_ep2 = ep_inner - (5*n_ep).astype(np.int)

            if segmentation[y_ep, x_ep] == s2 or segmentation[y_ep2, x_ep2] == s1:
                n_ep = -n_ep

            tang_nbrep = junction - nbr_ep_inner
            n_nbrep = np.array([-tang_nbrep[1], tang_nbrep[0]])
            if np.linalg.norm(n_nbrep) == 0:
                n_nbrep = n_nbrep
            else:
                n_nbrep = n_nbrep/np.linalg.norm(n_nbrep)

            x_nbrep, y_nbrep =  nbr_ep_inner + (5*n_nbrep).astype(np.int)
            x_nbrep2, y_nbrep2 =  nbr_ep_inner - (5*n_nbrep).astype(np.int)

            if segmentation[y_nbrep, x_nbrep] == t2 or segmentation[y_nbrep2, x_nbrep2] == t1:
                n_nbrep = -n_nbrep

            if np.dot(np.cross(n_ep, n_mp_mp), np.cross(n_mp_mp, n_nbrep)) > 0:
                dedge_neighbors[(s1, s2)].add((t1, t2))
                dedge_neighbors[(t1, t2)].add((s1, s2))
                dedge_neighbors[(s2, s1)].add((t2, t1))
                dedge_neighbors[(t2, t1)].add((s2, s1))            
            else:
                dedge_neighbors[(s2, s1)].add((t1, t2))
                dedge_neighbors[(t1, t2)].add((s2, s1))
                dedge_neighbors[(s1, s2)].add((t2, t1))
                dedge_neighbors[(t2, t1)].add((s1, s2))

    dedge_neighbors.default_factory = None

    print 'done in', time.time() - t, 'seconds'

#     dm.edge_neighbors = edge_neighbors
#     dm.neighbors = neighbors
#     dm.dedge_neighbors = dedge_neighbors
#     dm.dedge_vectors = dedge_vectors

    return neighbors, edge_neighbors, dedge_neighbors