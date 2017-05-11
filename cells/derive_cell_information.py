#! /usr/bin/env python

import sys
import os
import time

import numpy as np
from multiprocess import Pool
from skimage.filters import threshold_otsu
from scipy.ndimage.morphology import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.measure import regionprops, label
from skimage.transform import resize

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from data_manager import *
from metadata import *
from annotation_utilities import *
from registration_utilities import *
from cell_utilities import *

########################################

import argparse
import json

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Detect cells using cell profiler.')

parser.add_argument("stack", type=str, help="Stack")
parser.add_argument("filenames", type=str, help="filenames")
args = parser.parse_args()

########################################

stack = args.stack
filenames = json.loads(args.filenames)

min_blob_area = 10
max_blob_area = 10000
alg = 'cellprofiler'

def find_contour_worker(msk):
    if msk.shape[0] == 1:
        # if mask is a straight line, append another line to it.
        msk = np.vstack([msk, np.ones((msk.shape[1],))])
    elif msk.shape[1] == 1:
        msk = np.c_[msk, np.ones((msk.shape[0],))]
    return find_contour_points(msk, sample_every=1)[1][0]

output_dir = create_if_not_exists(os.path.join(DETECTED_CELLS_ROOTDIR, stack))

def detect_cell_one_section(fn):
    
    if is_invalid(fn):
        return
        
    fn_output_dir = create_if_not_exists(os.path.join(output_dir, fn))

    sys.stderr.write('Processing image %s\n' % fn)

    # Load mask
    t = time.time()
    mask_tb = DataManager.load_thumbnail_mask_v2(stack=stack, fn=fn)
    mask = resize(mask_tb, metadata_cache['image_shape'][stack][::-1]) > .5
    sys.stderr.write('Load mask: %.2f\n' % (time.time() - t) )

    if alg == 'myown':
        
        img_filename = DataManager.get_image_filepath(stack=stack, fn=fn, resol='lossless', version='cropped')

        img = imread(img_filename)
        sys.stderr.write('Load image: %.2f\n' % (time.time() - t) )

        t = time.time()
        im = rgb2gray(img)
        sys.stderr.write('Convert to gray: %.2f\n' % (time.time() - t) )

        t = time.time()

        thresh = threshold_otsu(im)
        binary = im < thresh
        binary[~mask] = 0

        sys.stderr.write('threshold: %.2f\n' % (time.time() - t) )
        
        t = time.time()
        dt = distance_transform_edt(binary)
        sys.stderr.write('distance transform: %.2f\n' % (time.time() - t) )

        t = time.time()
        local_maxi = peak_local_max(dt, labels=binary, footprint=np.ones((10, 10)), indices=False)
        sys.stderr.write('local max: %.2f\n' % (time.time() - t) )

        t = time.time()
        markers = label(local_maxi)
        sys.stderr.write('label: %.2f\n' % (time.time() - t) )

        t = time.time()
        labelmap = watershed(-dt, markers, mask=binary)
        sys.stderr.write('watershed: %.2f\n' % (time.time() - t) )
        
    elif alg == 'cellprofiler':
        labelmap = load_cell_data(stack=stack, fn=fn, what='image_inverted_labelmap_cellprofiler', ext='bp')
        labelmap[~mask] = 0
    
    elif alg == 'farsight':
        labelmap = load_cell_data(stack=stack, fn=fn, what='image_inverted_labelmap_farsight', ext='bp')
        labelmap[~mask] = 0
    
    else:
        raise 'Algorithm not recognized.'
    
    t = time.time()
    props = regionprops(labelmap.astype(np.int32))
    sys.stderr.write('regionprops: %.2f\n' % (time.time() - t) )

    valid_blob_indices = [i for i, p in enumerate(props) if p.area > min_blob_area and p.area < max_blob_area]
    sys.stderr.write('%d blobs identified.\n' % len(valid_blob_indices))
    
    # Get blobs
    t = time.time()
    valid_blob_coords = [props[i].coords for i in valid_blob_indices] # r,c
    fp = get_cell_data_filepath(stack=stack, fn=fn, what='blobCoords', ext='hdf')
    # Global coordinates of all pixels in the mask.
    save_hdf_v2(valid_blob_coords, fp)
    upload_to_s3(fp)
    sys.stderr.write('Save blob coords: %.2f\n' % (time.time() - t) )
    
    # Generate masks
    t = time.time()

    cell_masks = []
    cell_mask_centers = []
    for i, coords in enumerate(valid_blob_coords):
#         bar.value = i
        ymin, xmin = coords.min(axis=0)
        ymax, xmax = coords.max(axis=0)
        cell_mask = np.zeros((ymax+1-ymin, xmax+1-xmin), np.bool)
        cell_mask[coords[:,0]-ymin, coords[:,1]-xmin] = 1
        yc, xc = np.mean(np.where(cell_mask), axis=1)
        cell_masks.append(cell_mask)
        cell_mask_centers.append([xc, yc])
    
    fp = get_cell_data_filepath(stack=stack, fn=fn, what='blobMasks', ext='hdf')
    # binary mask of each blob.
    save_hdf_v2(cell_masks, fp)
    upload_to_s3(fp)
    
    fp = get_cell_data_filepath(stack=stack, fn=fn, what='blobMaskCenters', ext='bp')
    # blob centroid in bounding box
    bp.pack_ndarray_file(np.array(cell_mask_centers), fp)
    upload_to_s3(fp)
    
    sys.stderr.write('Save blob masks: %.2f\n' % (time.time() - t) )
    
    # Other blob attributes
    t = time.time()
    
    
    # Must use serial rather than multiprocess because this is nested in a worker process that is being parallelized.
    
    # pool = Pool(NUM_CORES)
    # valid_blob_contours = pool.map(find_contour_worker, cell_masks)
    # pool.terminate()
    # pool.join()
    valid_blob_contours = [find_contour_worker(msk) for msk in cell_masks]
        
    fp = get_cell_data_filepath(stack=stack, fn=fn, what='blobContours', ext='hdf')
    # contour coordinates relative to the bounding box
    save_hdf_v2(valid_blob_contours, fp)
    upload_to_s3(fp)

    sys.stderr.write('Save blob contours, save: %.2f\n' % (time.time() - t) )
    
    t = time.time()

    valid_blob_orientations = np.array([props[i].orientation for i in valid_blob_indices])
    valid_blob_centroids = np.array([props[i].centroid for i in valid_blob_indices])[:,::-1] # r,c -> x,y
    valid_blob_majorAxisLen = np.array([props[i].major_axis_length for i in valid_blob_indices])
    valid_blob_minorAxisLen = np.array([props[i].minor_axis_length for i in valid_blob_indices])

    fp = get_cell_data_filepath(stack=stack, fn=fn, what='blobOrientations', ext='bp')
    # in radian, relative to horizontal axis of the image
    bp.pack_ndarray_file(valid_blob_orientations, fp)
    upload_to_s3(fp)
    
    fp = get_cell_data_filepath(stack=stack, fn=fn, what='blobCentroids', ext='bp')
    # centroid coordinates of each blob relative to the whole image
    bp.pack_ndarray_file(valid_blob_centroids, fp)
    upload_to_s3(fp)
    
    fp = get_cell_data_filepath(stack=stack, fn=fn, what='blobMajorAxisLen', ext='bp')
    # in pixels
    bp.pack_ndarray_file(valid_blob_majorAxisLen, fp)
    upload_to_s3(fp)
    
    fp = get_cell_data_filepath(stack=stack, fn=fn, what='blobMinorAxisLen', ext='bp')
    # in pixels
    bp.pack_ndarray_file(valid_blob_minorAxisLen, fp)
    upload_to_s3(fp)
    
    blob_contours_global = [(valid_blob_contours[i] - cell_mask_centers[i] + valid_blob_centroids[i]).astype(np.int)
                            for i in range(len(valid_blob_coords))]
    blob_contours_global_fp = get_cell_data_filepath(stack=stack, fn=fn, what='blobContoursGlobal_%(alg)s' % {'alg':alg}, ext='hdf')
    # Contour coordinates of each blob relative to the whole image.
    save_hdf_v2(blob_contours_global, blob_contours_global_fp)
    upload_to_s3(blob_contours_global_fp)
    
    
    # Compute cell sizes
    cell_sizes = np.reshape(cell_masks, (cell_masks.shape[0], -1)).sum(axis=1)
    cell_sizes_fp = get_cell_data_filepath('cellSizes', stack=stack, sec=sec, ext='bp')
    bp.pack_ndarray_file(cell_sizes, cell_sizes_fp)

#     large_cell_indices = np.where(cell_sizes > cell_size_thresh)[0]
#     print 'Got %s large cells.' % len(large_cell_indices)

#     large_cell_indices_fp = get_cell_data_filepath('largeCellIndices', stack=stack, sec=sec, ext='bp')
#     bp.pack_ndarray_file(large_cell_indices, large_cell_indices_fp)

#     def compute_hu_moments(i):
#         b = cells_aligned_padded[i].astype(np.uint8)
#         m = moments(b, order=1)
#         hu = moments_hu(moments_normalized(moments_central(b, cc=m[0,1]/m[0,0], cr=m[1,0]/m[0,0])))    
#         return hu

#     t = time.time()

# # Using parallel sometimes causes stall. Not much faster than sequential anyway.
# #     pool = Pool(8)
# #     large_cell_hu_moments = np.array(pool.map(compute_hu_moments, large_cell_indices))
# #     pool.close()
# #     pool.join()

#     large_cell_hu_moments = np.array([compute_hu_moments(i) for i in large_cell_indices])

#     sys.stderr.write('Compute hu moments: %.2f seconds.\n' % (time.time()-t)) # 3-9s

#     cell_orientations = load_cell_data(what='blobOrientations', stack=stack, sec=sec, ext='bp')
#     cell_mirrors = load_cell_data(what='cells_aligned_mirrorDirections', stack=stack, sec=sec, ext='bp')

#     large_cell_features = np.c_[cell_orientations[large_cell_indices],
#                                      cell_mirrors[large_cell_indices],
#                                      cell_sizes[large_cell_indices], 
#                                      large_cell_hu_moments]

#     large_cell_features_fp = get_cell_data_filepath('largeCellFeatures', stack=stack, sec=sec, ext='bp')
#     bp.pack_ndarray_file(large_cell_features, large_cell_features_fp)
    

# This part is better moved to a separate script because it takes much more memory than other parts and cannot use
# the same number of processes.

#     w, h = metadata_cache['image_shape'][stack]
#     blob_contour_viz = np.zeros((h, w, 4), np.uint8)
#     for cnt in blob_contours_global:
#         blob_contour_viz[np.minimum(h-1, np.maximum(cnt[:,1], 0)), 
#                          np.minimum(w-1, np.maximum(cnt[:,0], 0))] = (255,0,0,255)
#     viz_fp = get_cell_data_filepath(stack=stack, fn=fn, what='blobContours_viz_%(alg)s' % {'alg': alg}, ext='png')
#     imsave(viz_fp, blob_contour_viz)
#     upload_to_s3(viz_fp)
    
#     sys.stderr.write('Compute blob properties, save: %.2f\n' % (time.time() - t) )
    
# for fn in filenames:
#     detect_cell_one_section(fn)
    
pool = Pool(NUM_CORES/2)
pool.map(detect_cell_one_section, filenames)
pool.close()
pool.join()