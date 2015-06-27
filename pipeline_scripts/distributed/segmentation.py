
import os
import argparse
import sys
from joblib import Parallel, delayed

sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], 'pipeline_scripts'))

if os.environ['DATASET_VERSION'] == '2014':
    from utilities2014 import *
elif os.environ['DATASET_VERSION'] == '2015':
    from utilities import *

parser = argparse.ArgumentParser(
formatter_class=argparse.RawDescriptionHelpFormatter,
description='Execute feature extraction pipeline',
epilog="""
The following command processes image RS141_x5_0001.tif using the specified parameters.
python %s RS141 1 -g blueNisslWide -s blueNisslRegular -v blueNissl
"""%(os.path.basename(sys.argv[0]), ))

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("slice_ind", type=int, help="slice index")
parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
args = parser.parse_args()

dm = DataManager(data_dir=os.environ['GORDON_DATA_DIR'], 
    repo_dir=os.environ['GORDON_REPO_DIR'], 
    result_dir=os.environ['GORDON_RESULT_DIR'], labeling_dir=os.environ['GORDON_LABELING_DIR'])

dm.set_gabor_params(gabor_params_id='blueNisslWide')
dm.set_segmentation_params(segm_params_id='blueNisslRegular')
dm.set_vq_params(vq_params_id='blueNissl')

dm.set_image(args.stack_name, 'x5', args.slice_ind)

#============================================================

# Over-segment the image into superpixels using SLIC (http://ivrg.epfl.ch/research/superpixels)

from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_ubyte, pad
import cv2

try:
    masked_segmentation_relabeled = dm.load_pipeline_result('segmentation', 'npy')
    print "segmentation.npy already exists, skip"

except Exception as e:

    masked_image = dm.image.copy()
    masked_image[~dm.mask] = 0

    segmentation = slic(gray2rgb(masked_image), n_segments=int(dm.segm_params['n_superpixels']), 
                                            max_iter=10, 
                                            compactness=float(dm.segm_params['slic_compactness']), 
                                            sigma=float(dm.segm_params['slic_sigma']), 
                                            enforce_connectivity=True)

    n = len(np.unique(segmentation))

    def f(i):
            m = dm.mask[segmentation == i]
            return np.count_nonzero(m)/float(len(m)) < .8

    q = Parallel(n_jobs=16)(delayed(f)(i) for i in range(n))

    masked_segmentation = segmentation.copy()
    for i in np.where(q)[0]:
        masked_segmentation[segmentation == i] = -1

    from skimage.segmentation import relabel_sequential

    # segmentation starts from 0
    masked_segmentation_relabeled, fw, inv = relabel_sequential(masked_segmentation + 1)

    # make background label -1
    masked_segmentation_relabeled -= 1

    dm.save_pipeline_result(masked_segmentation_relabeled, 'segmentation', 'npy')


try:
    sp_properties = dm.load_pipeline_result('spProps', 'npy')
    print "spProps.npy already exists, skip"
    sp_centroids = sp_properties[:,:2]

except Exception as e:

    sp_props = regionprops(masked_segmentation_relabeled + 1, intensity_image=dm.image, cache=True)

    def obtain_props_worker(i):
            return sp_props[i].centroid, sp_props[i].area, sp_props[i].mean_intensity, sp_props[i].bbox

    r = Parallel(n_jobs=16)(delayed(obtain_props_worker)(i) for i in range(len(sp_props)))

    sp_centroids = np.array([s[0] for s in r])
    sp_areas = np.array([s[1] for s in r])
    sp_mean_intensity = np.array([s[2] for s in r])
    sp_bbox = np.array([s[3] for s in r]) 

    # n_superpixel x 8: (cx, cy, area, mean_intensity, ymin, xmin, ymax, xmax)

    sp_properties = np.column_stack([sp_centroids, sp_areas, sp_mean_intensity, sp_bbox])

    dm.save_pipeline_result(sp_properties, 'spProps', 'npy')


n_superpixels = len(np.unique(masked_segmentation_relabeled)) - 1

img_superpixelized = mark_boundaries(dm.image, masked_segmentation_relabeled)
img_superpixelized_with_text = img_as_ubyte(img_superpixelized)

dm.save_pipeline_result(img_superpixelized_with_text, 'segmentationWithoutText', 'jpg')

for s in range(n_superpixels):
    cv2.putText(img_superpixelized_with_text, str(s), 
                 tuple(np.floor(sp_centroids[s][::-1]).astype(np.int) - np.array([10,-10])), 
                 cv2.FONT_HERSHEY_DUPLEX,
                 .5, ((255,0,255)), 1)

dm.save_pipeline_result(img_superpixelized_with_text, 'segmentationWithText', 'jpg')


emptycanvas_superpixelized = mark_boundaries(np.ones_like(dm.image), masked_segmentation_relabeled, 
                                             color=(0,0,0), outline_color=None)
alpha_channel = ~ emptycanvas_superpixelized.all(axis=2)
a = np.dstack([emptycanvas_superpixelized, alpha_channel])

dm.save_pipeline_result(a, 'segmentationTransparent', 'png', is_rgb=True)

# Compute neighbor lists and connectivity matrix

# if dm.check_pipeline_result('neighbors', 'pkl') and dm.check_pipeline_result('edgeCoords', 'pkl'):
#     print "neighbors.pkl already exists, skip"

# else:

from collections import defaultdict

d1 = np.diff(masked_segmentation_relabeled, axis=0).astype(np.bool)
d2 = np.diff(masked_segmentation_relabeled, axis=1).astype(np.bool)
d3 = (masked_segmentation_relabeled[1:,1:]-masked_segmentation_relabeled[:-1,:-1]).astype(np.bool)
d4 = (masked_segmentation_relabeled[1:,:-1]-masked_segmentation_relabeled[:-1,1:]).astype(np.bool)
d5 = (masked_segmentation_relabeled[2:,:]-masked_segmentation_relabeled[:-2,:]).astype(np.bool)
d6 = (masked_segmentation_relabeled[:,2:]-masked_segmentation_relabeled[:,:-2]).astype(np.bool)
d7 = (masked_segmentation_relabeled[2:,2:]-masked_segmentation_relabeled[:-2,:-2]).astype(np.bool)
d8 = (masked_segmentation_relabeled[2:,:-2]-masked_segmentation_relabeled[:-2,2:]).astype(np.bool)

neighbors = [set() for _ in range(n_superpixels)]
edge_coords = defaultdict(set)

def f(d, y1_offset=0, x1_offset=0, y2_offset=0, x2_offset=0):
    ys, xs = np.where(d)
    y1s = ys + y1_offset
    x1s = xs + x1_offset
    y2s = ys + y2_offset
    x2s = xs + x2_offset
    for y1,x1,y2,x2 in zip(y1s,x1s,y2s,x2s):
        l1 = masked_segmentation_relabeled[y1,x1]
        l2 = masked_segmentation_relabeled[y2,x2]
        if l1 != -1:
            neighbors[l1].add(l2)
            edge_coords[frozenset([l1,l2])].add((y1,x1))
            edge_coords[frozenset([l1,l2])].add((y2,x2))
        if l2 != -1:
            neighbors[l2].add(l1)
            edge_coords[frozenset([l1,l2])].add((y1,x1))
            edge_coords[frozenset([l1,l2])].add((y2,x2))
    
f(d1, 0,0,1,0)
f(d2, 0,0,0,1)
f(d3, 0,0,1,1)
f(d4, 0,1,1,0)
f(d5, 0,0,2,0)
f(d6, 0,0,0,2)
f(d7, 0,0,2,2)
f(d8, 0,2,2,0)

# check symmetry; note that this CANNOT check if neighbors is complete
A = np.zeros((n_superpixels, n_superpixels))
for i, nbrs in enumerate(neighbors):
    q = list([j for j in nbrs if j != -1])
    A[i, q] = 1    
assert np.all(A == A.T), 'neighbor list is not symmetric'

for i in range(n_superpixels):
    neighbors[i] -= {i}
    
dm.save_pipeline_result(neighbors, 'neighbors', 'pkl')
dm.save_pipeline_result(edge_coords, 'edgeCoords', 'pkl')

border_len_thresh = 20
neighbors_long_border = [set([nbr for nbr in nbrs if len(edge_coords[frozenset([n,nbr])]) > border_len_thresh]) 
                             for n, nbrs in enumerate(neighbors)]
dm.save_pipeline_result(neighbors_long_border, 'neighborsLongBorder', 'pkl')