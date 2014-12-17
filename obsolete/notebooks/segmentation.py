# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# Over-segment the image into superpixels using SLIC (http://ivrg.epfl.ch/research/superpixels)

from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_ubyte

img_rgb = gray2rgb(dm.image)

try:
    segmentation = dm.load_pipeline_result('segmentation', 'npy')
    
except Exception as e:
    segmentation = slic(img_rgb, n_segments=int(dm.segm_params['n_superpixels']), 
                        max_iter=10, 
                        compactness=float(dm.segm_params['slic_compactness']), 
                        sigma=float(dm.segm_params['slic_sigma']), 
                        enforce_connectivity=True)
    print 'segmentation computed'
    
    dm.save_pipeline_result(segmentation.astype(np.int16), 'segmentation', 'npy')

# <codecell>

from skimage.segmentation import relabel_sequential

try:
    cropped_segmentation_relabeled = dm.load_pipeline_result('cropSegmentation', 'npy')
except:
    # segmentation starts from 0
    cropped_segmentation = crop_borders(segmentation)
    n_superpixels = len(np.unique(cropped_segmentation))
    cropped_segmentation[~cropped_mask] = -1
    cropped_segmentation_relabeled, fw, inv = relabel_sequential(cropped_segmentation + 1)

    # make background label -1
    cropped_segmentation_relabeled -= 1
    dm.save_pipeline_result(cropped_segmentation_relabeled, 'cropSegmentation', 'npy')

# <codecell>

sp_props = regionprops(cropped_segmentation_relabeled + 1, intensity_image=cropped_img, cache=True)

def obtain_props_worker(i):
    return sp_props[i].centroid, sp_props[i].area, sp_props[i].mean_intensity, sp_props[i].bbox

r = Parallel(n_jobs=16)(delayed(obtain_props_worker)(i) for i in range(len(sp_props)))
sp_centroids = np.array([s[0] for s in r])
sp_areas = np.array([s[1] for s in r])
sp_mean_intensity = np.array([s[2] for s in r])
sp_bbox = np.array([s[3] for s in r])

sp_properties = np.column_stack([sp_centroids, sp_areas, sp_mean_intensity, sp_bbox])

dm.save_pipeline_result(sp_properties, 'cropSpProps', 'npy')

n_superpixels = len(np.unique(cropped_segmentation_relabeled))

img_superpixelized = mark_boundaries(gray2rgb(cropped_img), cropped_segmentation_relabeled)
img_superpixelized_text = img_as_ubyte(img_superpixelized)

# background label (-1) is not displayed
for s in range(n_superpixels - 1):
    img_superpixelized_text = cv2.putText(img_superpixelized_text, str(s), 
                      tuple(np.floor(sp_centroids[s][::-1]).astype(np.int)), 
                      cv2.FONT_HERSHEY_COMPLEX_SMALL,
                      .5, ((255,0,255)), 1)

dm.save_pipeline_result(img_superpixelized_text, 'cropSegmentation', 'tif')

# <codecell>

# Compute neighbor lists and connectivity matrix

from skimage.morphology import disk
from skimage.filter.rank import gradient
# from scipy.sparse import coo_matrix

try:
    neighbors = dm.load_pipeline_result('neighbors', 'npy')

except:

    edge_map = gradient(cropped_segmentation_relabeled.astype(np.uint8), disk(3))
    neighbors = [set() for i in range(n_superpixels)]

    for y,x in zip(*np.nonzero(edge_map)):
        neighbors[cropped_segmentation_relabeled[y,x]] |= set(cropped_segmentation_relabeled[y-2:y+2,x-2:x+2].ravel())

    for i in range(n_superpixels):
        neighbors[i] -= set([i])

    dm.save_pipeline_result(neighbors, 'neighbors', 'npy')

