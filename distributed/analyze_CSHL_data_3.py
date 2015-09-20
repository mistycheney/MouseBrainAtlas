#! /usr/bin/env python

from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.util import img_as_float, img_as_ubyte, pad
from skimage.transform import integral_image

import numpy as np

from joblib import Parallel, delayed

from scipy.signal import fftconvolve
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial.distance import cdist, pdist

import matplotlib.pyplot as plt

from utilities2015 import *

import os, sys
import cv2
import time

centroids = dm.load_pipeline_result('textons', 'npy')
n_texton = len(centroids)

t = time.time()
print 'assign textons ...',

# kmeans = MiniBatchKMeans(n_clusters=n_reduced_texton, batch_size=1000, init=reduced_centroids[:, :20], max_iter=1)
# kmeans.fit(features_rotated[:, :20])
# labels = kmeans.labels_

def first_last_tuples_distribute_over(first_sec, last_sec, n_host):
    secs_per_job = (last_sec - first_sec + 1)/float(n_host)
    first_last_tuples = [(int(first_sec+i*secs_per_job), int(first_sec+(i+1)*secs_per_job-1) if i != n_host - 1 else last_sec) for i in range(n_host)]
    return first_last_tuples

label_list = []
for f, l in first_last_tuples_distribute_over(0, len(features_rotated), 3):
    print f, l
    D = cdist(features_rotated[f:l+1, :20], centroids[:, :20])
    labels = np.argmin(D, axis=1)
    label_list.append(labels)
labels = np.concatenate(label_list)

texton_map = labels.reshape((h,w))

colors = (np.loadtxt('../visualization/100colors.txt') * 255).astype(np.uint8)
textonmap_viz = colors[texton_map]

print 'done in', time.time() - t, 'seconds'

# del features_rotated
# del kmeans
del label_list
del D
del labels

dm.save_pipeline_result(texton_map, 'textonmap', 'npy')
dm.save_pipeline_result(textonmap_viz, 'textonmapViz', 'jpg')

texton_hists = {}
segmentation = np.zeros((h,w), np.int)
rss, css = np.mgrid[0:h:50, 0:w:50]
for gi, (rs, cs) in enumerate(zip(rss.flat, css.flat)):
    segmentation[rs:rs+50, cs:cs+50] = gi
    hist = np.bincount(texton_map[rs:rs+50, cs:cs+50].flat, minlength=n_texton)
    texton_hists[gi] = hist/float(np.sum(hist))
    
segmentation_viz = colors[segmentation%len(colors)]

dm.save_pipeline_result(segmentation_viz, 'segmentaionViz', 'jpg')
dm.save_pipeline_result(np.asarray(texton_hists.values()), 'texHist', 'npy')
dm.save_pipeline_result(segmentation, 'segmentation', 'npy')


n_superpixels = len(np.unique(segmentation))



print 'computing sp properties ...',
t = time.time()

sp_all_props = regionprops(segmentation + 1, cache=True)

def obtain_props_worker(spp):
    return spp.centroid, spp.area, spp.bbox

sp_props = Parallel(n_jobs=16)(delayed(obtain_props_worker)(spp) for spp in sp_all_props)
sp_centroids, sp_areas, sp_bbox = map(np.asarray, zip(*sp_props))

dm.save_pipeline_result(sp_centroids, 'spCentroids', 'npy')
dm.save_pipeline_result(sp_areas, 'spAreas', 'npy')
dm.save_pipeline_result(sp_bbox, 'spBbox', 'npy')

print 'done in', time.time() - t, 'seconds'




from collections import defaultdict

print 'computing neighbors ...',
t = time.time()


def diff_offset(x_offset, y_offset):

    h, w = segmentation.shape

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
    r = np.c_[d[nzs], ys[nzs] + y_offset, xs[nzs] + x_offset] # [sp_label1, sp_label2, y, x]

    return r

diffs = np.vstack([diff_offset(1,0), diff_offset(0,1), diff_offset(1,1), diff_offset(1,-1)])

edge_coords = defaultdict(set)
neighbors = [set() for _ in range(n_superpixels)]

for i, j, y, x in diffs:
    edge_coords[frozenset([i,j])].add((x,y))
    if i != -1:
        neighbors[i].add(j)
    if j != -1:
        neighbors[j].add(i)
        
edge_coords = dict((e, np.array(list(pts))) for e, pts in edge_coords.iteritems())

# check symmetry; note that this CANNOT check if neighbors is complete
A = np.zeros((n_superpixels, n_superpixels))
for i, nbrs in enumerate(neighbors):
    q = list([j for j in nbrs if j != -1])
    A[i, q] = 1    
assert np.all(A == A.T), 'neighbor list is not symmetric'

dm.save_pipeline_result(neighbors, 'neighbors', 'pkl')
dm.save_pipeline_result(edge_coords, 'edgeCoords', 'pkl')

print 'done in', time.time() - t, 'seconds'


from skimage.segmentation import mark_boundaries

print 'generating segmentation visualization ...',
t = time.time()

img_superpixelized = mark_boundaries(dm.image[ymin:ymax+1, xmin:xmax+1], segmentation)
img_superpixelized = img_as_ubyte(img_superpixelized)
dm.save_pipeline_result(img_superpixelized, 'segmentationWithoutText', 'jpg')

for s in range(n_superpixels):
    cv2.putText(img_superpixelized, str(s), 
                tuple(np.floor(sp_centroids[s][::-1]).astype(np.int) - np.array([10,-10])), 
                cv2.FONT_HERSHEY_DUPLEX, .5, ((255,0,255)), 1)

dm.save_pipeline_result(img_superpixelized, 'segmentationWithText', 'jpg')

emptycanvas_superpixelized = mark_boundaries(np.ones((h,w)), segmentation, 
                                             color=(0,0,0), outline_color=None)

alpha_channel = ~ emptycanvas_superpixelized.all(axis=2)
rgba = np.dstack([emptycanvas_superpixelized, alpha_channel])

print 'done in', time.time() - t, 'seconds'

dm.save_pipeline_result(rgba, 'segmentationTransparent', 'png', is_rgb=True)


print 'computing edge neighbors ...',
t = time.time()

edge_map = -1 * np.ones_like(segmentation, np.int)

for ei, pts in enumerate(edge_coords.itervalues()):
    edge_map[pts[:,1], pts[:,0]] = ei
    
edges = edge_coords.keys()

xs, ys = np.mgrid[-5:5, -5:5]

def compute_edge_neighbors_worker(pts):
    nbrs = set(edge_map[np.maximum(0, np.minimum(h-1, (pts[:,1] + ys[:,:,None]).flat)), 
                        np.maximum(0, np.minimum(w-1, (pts[:,0] + xs[:,:,None]).flat))])
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