#! /usr/bin/env python

import os
import argparse
import sys
import time

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Segment image into superpixels',
    epilog="")

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("slice_ind", type=int, help="slice index")
parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
args = parser.parse_args()

from joblib import Parallel, delayed

sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], 'utilities'))
from utilities2015 import *

dm = DataManager(gabor_params_id=args.gabor_params_id, 
                 segm_params_id=args.segm_params_id, 
                 vq_params_id=args.vq_params_id,
                 stack=args.stack_name, 
                 section=args.slice_ind)

#======================================================

from skimage.segmentation import slic, mark_boundaries, relabel_sequential
from skimage.measure import regionprops
from skimage.util import img_as_ubyte, pad
import cv2

try:
    raise
    segmentation = dm.load_pipeline_result('segmentation')
    print "segmentation.npy already exists, skip"

except Exception as e:

    # grid_size = dm.grid_size

    # segmentation = np.zeros((dm.image_height, dm.image_width), np.int16)
    # rss, css = np.mgrid[0:dm.image_height:grid_size, 0:dm.image_width:grid_size]
    # for gi, (rs, cs) in enumerate(zip(rss.flat, css.flat)):
    #     segmentation[rs:rs+grid_size, cs:cs+grid_size] = gi

    # n = len(np.unique(segmentation))
    # segmentation[~dm.mask] = -1

    sys.stderr.write('superpixel segmentation ...\n')
    t1 = time.time()

    if dm.segm_params_id in ['gridsize200', 'gridsize100', 'gridsize50']:
        grid_size = dm.grid_size

        segmentation = np.zeros((dm.image_height, dm.image_width), np.int16)
        rss, css = np.mgrid[0:dm.image_height:grid_size, 0:dm.image_width:grid_size]
        for gi, (rs, cs) in enumerate(zip(rss.flat, css.flat)):
            segmentation[rs:rs+grid_size, cs:cs+grid_size] = gi

    elif dm.segm_params_id in ['blueNisslRegular', 'n1500']:

        dm._load_image(versions=['rgb'])
        segmentation = -1 * np.ones((dm.image_height, dm.image_width), np.int16)
        segmentation[dm.ymin:dm.ymax+1, dm.xmin:dm.xmax+1] = slic(dm.image_rgb[dm.ymin:dm.ymax+1, dm.xmin:dm.xmax+1], 
                                                                n_segments=int(dm.segm_params['n_superpixels']), 
                                                                max_iter=10, 
                                                                compactness=float(dm.segm_params['slic_compactness']), 
                                                                sigma=float(dm.segm_params['slic_sigma']), 
                                                                enforce_connectivity=True)

    elif dm.segm_params_id in ['tSLIC200']:

        from slic_texture import slic_texture, enforce_connectivity
        from skimage.transform import integral_image

        segmentation = np.zeros((dm.image_height, dm.image_width), np.int16)

        textonmap = dm.load_pipeline_result('texMap')
        n_texton = textonmap.max() + 1


        sys.stderr.write('compute texture histogram map\n')
        t = time.time()

        window_size = 201
        window_halfsize = (window_size-1)/2

        single_channel_maps = [textonmap[dm.ymin-window_halfsize : dm.ymax+1+window_halfsize, 
                                         dm.xmin-window_halfsize : dm.xmax+1+window_halfsize] == c
                               for c in range(n_texton)]

        # it is important to pad the integral image with zeros before first row and first column
        def compute_integral_image(m):
            return np.pad(integral_image(m), ((1,0),(1,0)), mode='constant', constant_values=0)

        int_imgs = np.dstack(Parallel(n_jobs=4)(delayed(compute_integral_image)(m) for m in single_channel_maps))

        histograms = int_imgs[window_size:, window_size:] + \
                    int_imgs[:-window_size, :-window_size] - \
                    int_imgs[window_size:, :-window_size] - \
                    int_imgs[:-window_size, window_size:]

        histograms_normalized = histograms/histograms.sum(axis=-1)[...,None].astype(np.float)

        del single_channel_maps, histograms, int_imgs

        sys.stderr.write('done in %.2f seconds\n' % (time.time() - t))

        seg = slic_texture(histograms_normalized, max_iter=10)

        sys.stderr.write('enforce connectivity\n')
        t = time.time()

        segmentation[dm.ymin:dm.ymax+1, dm.xmin:dm.xmax+1] = enforce_connectivity(seg)

        sys.stderr.write('done in %.2f seconds\n' % (time.time() - t))


    segmentation[~dm.mask] = -1
    				
    # segmentation starts from 0
    masked_segmentation_relabeled, _, _ = relabel_sequential(segmentation + 1)

    # make background label -1
    segmentation = masked_segmentation_relabeled - 1

    dm.save_pipeline_result(segmentation.astype(np.int16), 'segmentation')

    sys.stderr.write('done in %.2f seconds\n' % (time.time() - t1))

n_superpixels = len(np.unique(segmentation)) - 1

print 'computing sp properties ...',
t = time.time()

sp_all_props = regionprops(segmentation + 1, cache=True)

def obtain_props_worker(spp):
    return spp.centroid, spp.area, spp.bbox, spp.coords
    # (row, col), a, (min_row, min_col, max_row, max_col),(rows, cols)

sp_props = Parallel(n_jobs=16)(delayed(obtain_props_worker)(spp) for spp in sp_all_props)
sp_centroids, sp_areas, sp_bbox, spp_coords = map(np.asarray, zip(*sp_props))

dm.save_pipeline_result(sp_centroids, 'spCentroids')
dm.save_pipeline_result(sp_areas, 'spAreas')
dm.save_pipeline_result(sp_bbox, 'spBbox')
dm.save_pipeline_result(spp_coords, 'spCoords')

print 'done in', time.time() - t, 'seconds'


from collections import defaultdict

try:
    raise
    edge_coords = dm.load_pipeline_result('edgeCoords')
    neighbors = dm.load_pipeline_result('neighbors')
    edge_midpoints = dm.load_pipeline_result('edgeMidpoints')
    dedge_vectors = dm.load_pipeline_result('dedgeVectors')

except:

    print 'computing neighbors ...',
    t = time.time()


    def diff_offset(x_offset, y_offset):

        h, w = segmentation.shape
        
        if x_offset == 1 and y_offset == -1:
            d = np.dstack([segmentation[0:h-1, 1:w], segmentation[1:h, 0:w-1]])
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

            d = np.dstack([segmentation[moving_y_low:moving_y_high, moving_x_low:moving_x_high], 
                           segmentation[:moving_height, :moving_width]])

            ys, xs = np.mgrid[:d.shape[0], :d.shape[1]]
            nzs = ~(d[...,0]==d[...,1])
            # if diff's location y,x is (0,0), the edge is at (0,0) and (y_offset, x_offset)
            r = np.c_[d[nzs], ys[nzs], xs[nzs], ys[nzs] + y_offset, xs[nzs] + x_offset] # [sp_label1, sp_label2, y1,x1,y2,x2]     
        
        return r

    diffs = np.vstack([diff_offset(1,0), diff_offset(0,1)])

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

    dm.save_pipeline_result(neighbors, 'neighbors')

    print 'done in', time.time() - t, 'seconds'

    print 'compute edge info ...',
    t = time.time()

    dedge_vectors = {}
    edge_coords_sorted = {}
    edge_midpoints = {}
    edge_endpoints = {}

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

    dm.save_pipeline_result(edge_coords, 'edgeCoords')
    dm.save_pipeline_result(edge_midpoints, 'edgeMidpoints')
    dm.save_pipeline_result(dedge_vectors, 'dedgeVectors')
    dm.save_pipeline_result(dict([(edge, (coords[0], coords[-1])) for edge, coords in edge_coords.iteritems()]), 
                            'edgeEndpoints')

 
# if dm.check_pipeline_result('segmentationWithText'):
if False:
    sys.stderr.write('visualizations exist, skip')
else:

    e_coords = np.vstack(edge_coords.itervalues())

    print 'generating segmentation visualization ...',
    t = time.time()

    dm._load_image(versions=['rgb-jpg'])
    img_superpixelized = dm.image_rgb_jpg.copy()
    img_superpixelized[e_coords[:,1], e_coords[:,0]] = (1,0,0)

    # img_superpixelized = mark_boundaries(dm.image_rgb_jpg, segmentation, color=(1,0,0))
    img_superpixelized = img_as_ubyte(img_superpixelized)
    dm.save_pipeline_result(img_superpixelized, 'segmentationWithoutText')

    for s in range(n_superpixels):
        cv2.putText(img_superpixelized, str(s), 
                    tuple(sp_centroids[s][::-1].astype(np.int) - (10,-10)), 
                    cv2.FONT_HERSHEY_DUPLEX, .5, ((255,0,0)), 1)

    dm.save_pipeline_result(img_superpixelized, 'segmentationWithText')

    rgba = np.zeros((dm.image_height, dm.image_width, 4), np.uint8)
    rgba[e_coords[:,1], e_coords[:,0], 3] = 255
    dm.save_pipeline_result(rgba, 'segmentationTransparent', is_rgb=True)

    print 'done in', time.time() - t, 'seconds'

try:
    raise
    edge_neighbors = dm.load_pipeline_result('edgeNeighbors')

except:

    print 'computing edge neighbors ...',
    t = time.time()

    edge_map = -1 * np.ones_like(segmentation, np.int)

    for ei, pts in enumerate(edge_coords.itervalues()):
        edge_map[pts[:,1], pts[:,0]] = ei
        
    edges = edge_coords.keys()

    xs, ys = np.mgrid[-1:2, -1:2]

    def compute_edge_neighbors_worker(pts):
        nbrs = set(edge_map[np.maximum(0, np.minimum(dm.image_height-1, (pts[:,1] + ys[:,:,None]).flat)), 
                            np.maximum(0, np.minimum(dm.image_width-1, (pts[:,0] + xs[:,:,None]).flat))])
        return nbrs

    edge_neighbors = {}
    for ei, (e, pts) in enumerate(edge_coords.iteritems()):
        nbr_ids = compute_edge_neighbors_worker(pts) - {-1, ei}
        edge_neighbors[e] = set([edges[i] for i in nbr_ids if len(set.intersection(edge_junctions[e], edge_junctions[edges[i]])) > 0])
    
    print 'done in', time.time() - t, 'seconds'

    dm.save_pipeline_result(edge_neighbors, 'edgeNeighbors')


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

dm.save_pipeline_result(dedge_neighbors, 'dedgeNeighbors')
