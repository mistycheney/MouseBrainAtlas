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

sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], 'notebooks'))
from utilities2015 import *

os.environ['GORDON_DATA_DIR'] = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_processed'
os.environ['GORDON_REPO_DIR'] = '/oasis/projects/nsf/csd395/yuncong/Brain'
os.environ['GORDON_RESULT_DIR'] = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_results'

dm = DataManager(data_dir=os.environ['GORDON_DATA_DIR'], repo_dir=os.environ['GORDON_REPO_DIR'], 
    result_dir=os.environ['GORDON_RESULT_DIR'], labeling_dir=os.environ['GORDON_LABELING_DIR'],
    stack=args.stack_name, section=args.slice_ind)

#======================================================

from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops
from skimage.util import img_as_ubyte, pad
import cv2

try:
    segmentation = dm.load_pipeline_result('segmentation', 'npy')
    print "segmentation.npy already exists, skip"

except Exception as e:

    # masked_image = dm.image.copy()
    # masked_image[~dm.mask] = 0

    grid_size = 100

    segmentation = np.zeros((dm.image_height, dm.image_width), np.int)
    rss, css = np.mgrid[0:dm.image_height:grid_size, 0:dm.image_width:grid_size]
    for gi, (rs, cs) in enumerate(zip(rss.flat, css.flat)):
        segmentation[rs:rs+grid_size, cs:cs+grid_size] = gi

    # segmentation = slic(gray2rgb(masked_image), n_segments=int(dm.segm_params['n_superpixels']), 
    #                                         max_iter=10, 
    #                                         compactness=float(dm.segm_params['slic_compactness']), 
    #                                         sigma=float(dm.segm_params['slic_sigma']), 
    #                                         enforce_connectivity=True)

    n = len(np.unique(segmentation))

    segmentation[~dm.mask] = -1
    				
    from skimage.segmentation import relabel_sequential

    # segmentation starts from 0
    masked_segmentation_relabeled, _, _ = relabel_sequential(segmentation + 1)

    # make background label -1
    segmentation = masked_segmentation_relabeled - 1

    dm.save_pipeline_result(segmentation, 'segmentation', 'npy')


n_superpixels = len(np.unique(segmentation)) - 1

print 'computing sp properties ...',
t = time.time()

sp_all_props = regionprops(segmentation + 1, cache=True)

def obtain_props_worker(spp):
    return spp.centroid, spp.area, spp.bbox, spp.coords

sp_props = Parallel(n_jobs=16)(delayed(obtain_props_worker)(spp) for spp in sp_all_props)
sp_centroids, sp_areas, sp_bbox, spp_coords = map(np.asarray, zip(*sp_props))

dm.save_pipeline_result(sp_centroids, 'spCentroids', 'npy')
dm.save_pipeline_result(sp_areas, 'spAreas', 'npy')
dm.save_pipeline_result(sp_bbox, 'spBbox', 'npy')
dm.save_pipeline_result(spp_coords, 'spCoords', 'npy')

print 'done in', time.time() - t, 'seconds'


# from skimage.segmentation import mark_boundaries

print 'generating segmentation visualization ...',
t = time.time()

dm._load_image()
img_superpixelized = mark_boundaries(dm.image, segmentation)
img_superpixelized = img_as_ubyte(img_superpixelized)
dm.save_pipeline_result(img_superpixelized, 'segmentationWithoutText', 'jpg')

for s in range(n_superpixels):
    cv2.putText(img_superpixelized, str(s), 
                tuple(np.floor(sp_centroids[s][::-1]).astype(np.int) - np.array([10,-10])), 
                cv2.FONT_HERSHEY_DUPLEX, .5, ((255,0,255)), 1)

dm.save_pipeline_result(img_superpixelized, 'segmentationWithText', 'jpg')

emptycanvas_superpixelized = mark_boundaries(np.ones((dm.image_height, dm.image_width)), segmentation, 
                                             color=(0,0,0), outline_color=None)

alpha_channel = ~ emptycanvas_superpixelized.all(axis=2)
rgba = np.dstack([emptycanvas_superpixelized, alpha_channel])

dm.save_pipeline_result(rgba, 'segmentationTransparent', 'png', is_rgb=True)

print 'done in', time.time() - t, 'seconds'


from collections import defaultdict

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

diffs = np.vstack([diff_offset(1,0), diff_offset(0,1), diff_offset(1,1), diff_offset(1,-1)])

edge_coords = defaultdict(set)
neighbors = [set() for _ in range(n_superpixels)]

for i, j, y1, x1, y2, x2 in diffs:
    edge_coords[frozenset([i,j])] |= {(x1,y1), (x2,y2)}
    if i != -1:
        neighbors[i].add(j)
    if j != -1:
        neighbors[j].add(i)
        
edge_coords = dict((e, np.array(list(pts))) for e, pts in edge_coords.iteritems())

# check symmetry; note that this CANNOT check if neighbors is complete
# A = np.zeros((n_superpixels, n_superpixels))
# for i, nbrs in enumerate(neighbors):
#     q = list([j for j in nbrs if j != -1])
#     A[i, q] = 1    
# assert np.all(A == A.T), 'neighbor list is not symmetric'

dm.save_pipeline_result(neighbors, 'neighbors', 'pkl')
dm.save_pipeline_result(edge_coords, 'edgeCoords', 'pkl')

print 'done in', time.time() - t, 'seconds'


print 'computing edge neighbors ...',
t = time.time()

edge_map = -1 * np.ones_like(segmentation, np.int)

for ei, pts in enumerate(edge_coords.itervalues()):
    edge_map[pts[:,1], pts[:,0]] = ei
    
edges = edge_coords.keys()

xs, ys = np.mgrid[-5:5, -5:5]

def compute_edge_neighbors_worker(pts):
    nbrs = set(edge_map[np.maximum(0, np.minimum(dm.image_height-1, (pts[:,1] + ys[:,:,None]).flat)), 
                        np.maximum(0, np.minimum(dm.image_width-1, (pts[:,0] + xs[:,:,None]).flat))])
    return nbrs


edge_neighbors = {}
for ei, (e, pts) in enumerate(edge_coords.iteritems()):
    nbr_ids = compute_edge_neighbors_worker(pts) - {-1, ei}
    edge_neighbors[e] = set(edges[i] for i in nbr_ids)
    
print 'done in', time.time() - t, 'seconds'

dm.save_pipeline_result(edge_neighbors, 'edgeNeighbors', 'pkl')


print 'sort edge points ...',
t = time.time()

edge_coords_sorted = {}
edge_midpoint = {}

for e, pts in edge_coords.iteritems():
    X = pts.astype(np.float)
    c = X.mean(axis=0)
    edge_midpoint[e] = c
    Xc = X - c
    U,S,V = np.linalg.svd(np.dot(Xc.T, Xc))
    u1 = U[:,0]
    projs = np.dot(Xc,u1)
    order = projs.argsort()
    if Xc[order[0],0] > Xc[order[-1],0]:
        order = order[::-1]
    edge_coords_sorted[e] = X[order].astype(np.int)

print 'done in', time.time() - t, 'seconds'

dm.save_pipeline_result(edge_coords_sorted, 'edgeCoords', 'pkl')
dm.save_pipeline_result(edge_midpoint, 'edgeMidpoints', 'pkl')

print 'compute edge vectors ...',
t = time.time()

dedge_vectors = defaultdict(float)

for e in edge_coords.iterkeys():
    c = edge_midpoint[e]
    i, j = e
    if i == -1:
        vector_ji = sp_centroids[j, ::-1] - c
    elif j == -1:
        vector_ji = c - sp_centroids[i, ::-1]
    else:
        vector_ji = sp_centroids[i, ::-1] - sp_centroids[j, ::-1]

    dedge_vectors[(i,j)] = vector_ji/np.linalg.norm(vector_ji)
    dedge_vectors[(j,i)] = -dedge_vectors[(i,j)]

dedge_vectors.default_factory = None

print 'done in', time.time() - t, 'seconds'

dm.save_pipeline_result(dedge_vectors, 'edgeVectors', 'pkl')

print 'compute dedge neighbors ...',
t = time.time()

dedge_neighbors = defaultdict(set)
for (i,j), es in edge_neighbors.iteritems():

    if len(es) == 0:
        print 'WARNING: edge (%d,%d) has no neighbors'%(i,j)
        ls = []
    else:
        ls = set.union(*[{(a,b),(b,a)} for a,b in es])
    
    dedge_neighbors[(i,j)] |= set((a,b) for a,b in ls 
                                  if not (i==b or j==a) and\
                                  np.dot(dedge_vectors[(i,j)], dedge_vectors[(a,b)]) > -.5) - {(j,i),(i,j)}

    dedge_neighbors[(j,i)] |= set((a,b) for a,b in ls 
                                  if not (j==b or i==a) and\
                                  np.dot(dedge_vectors[(j,i)], dedge_vectors[(a,b)]) > -.5) - {(j,i),(i,j)}

dedge_neighbors.default_factory = None

print 'done in', time.time() - t, 'seconds'

dm.save_pipeline_result(dedge_neighbors, 'dedgeNeighbors', 'pkl')
