import numpy as np
from joblib import Parallel, delayed
from utilities2015 import *
import time

from itertools import chain
from operator import itemgetter, attrgetter
from skimage.measure import label, regionprops
from collections import Counter

histogram_map = None
# mask = None

def compute_distance_to_centroids(centroids_xy, centroids_texture, spacing, w_spatial, h, w,
                                 ymins, ymaxs, xmins, xmaxs, window_spatial_distances):
    
    n = len(centroids_xy)
    
    ds = [None for _ in range(n)]   # list of n sx1 arrays (s = 2 spacing x 2 spacing)
        
    for ci in range(n):
    
        ymin = ymins[ci]
        xmin = xmins[ci]
        ymax = ymaxs[ci]
        xmax = xmaxs[ci]
        
        cx, cy = centroids_xy[ci].astype(np.int)
                
        crop_window_x_min = spacing-cx if cx-spacing < 0 else 0
        crop_window_y_min = spacing-cy if cy-spacing < 0 else 0
        crop_window_x_max = 2*spacing - (cx+spacing - (w - 1)) if cx+spacing > w - 1 else 2*spacing
        crop_window_y_max = 2*spacing - (cy+spacing - (h - 1)) if cy+spacing > h - 1 else 2*spacing
        
        spatial_ds = window_spatial_distances[crop_window_y_min:crop_window_y_max+1,
                                              crop_window_x_min:crop_window_x_max+1].reshape((-1,))

        texture_ds = chi2s([centroids_texture[ci]], 
                           histogram_map[ymin:ymax+1, xmin:xmax+1].reshape((-1, histogram_map.shape[-1])))

        # try:
        ds[ci] = w_spatial * spatial_ds + texture_ds
        # except:
        #     print ci, cx, cy
        #     print crop_window_x_min, crop_window_x_max, crop_window_y_min, crop_window_y_max
        #     print xmin, xmax, ymin, ymax
        #     print spatial_ds.size, texture_ds.size
        #     print
        #     raise

    return ds

# def compute_new_centroids(sps, assignments, hist_map):
#     centroids = [None for _ in range(len(sps))]
#     for i, sp_i in enumerate(sps):
#         rs, cs = np.where(assignments == sp_i)
#         centroids[i] = np.r_[np.c_[rs, cs].mean(axis=0)[::-1], hist_map[rs, cs].mean(axis=0)]
#     return centroids

import warnings

def slic_texture(hist_map, mask, spacing=200, w_spatial=0.001, max_iter=5):

    global histogram_map
    histogram_map = hist_map
    # global mask
    # mask = mask1

    h, w, n_texton = hist_map.shape
    
    sp_ys, sp_xs = np.mgrid[0:h:spacing, 0:w:spacing]
    sp_is_valid = mask[sp_ys, sp_xs]
    centroids_xy = np.c_[sp_xs[sp_is_valid], sp_ys[sp_is_valid]]
    centroids_textures = hist_map[centroids_xy[:,1], centroids_xy[:,0]].reshape((-1, hist_map.shape[-1]))

    n_superpixels = len(centroids_xy)
    sys.stderr.write('%d superpixels\n'%n_superpixels)

    number_sps_per_worker = int(n_superpixels / 128)
    print 'number_sps_per_worker', number_sps_per_worker

    # sp_size = h*w/n_superpixels
    # roi_per_worker = 10000
    # number_sps_per_worker = roi_per_worker / sp_size

    ys, xs = np.mgrid[-spacing:spacing+1, -spacing:spacing+1]
    window_spatial_distances = np.sqrt(ys**2 + xs**2)
    
    for iter_i in range(max_iter):

        print 'iteration', iter_i

        cxs = centroids_xy[:, 0].astype(np.int)
        cys = centroids_xy[:, 1].astype(np.int)
        window_xmins = np.maximum(0, cxs - spacing)
        window_xmaxs = np.minimum(w-1, cxs + spacing)
        window_ymins = np.maximum(0, cys - spacing)
        window_ymaxs = np.minimum(h-1, cys + spacing)

        # for sp in range(n_superpixels):
        #     # if np.all(np.isnan(histogram_map[window_ymins[sp]:window_ymaxs[sp]+1, window_xmins[sp]:window_xmaxs[sp]+1])):
        #     if np.all(np.isnan(histogram_map[centroids_xy[sp,1], centroids_xy[sp,0]])):
        #         print 'masked', sp, centroids_xy[sp]

        # sys.exit(0)

        assignments = -1 * np.ones((h, w), np.int16)
        distances = np.inf * np.ones((h, w), np.float16)

        t = time.time()            

        sys.stderr.write('compute distance\n')
        
        res = Parallel(n_jobs=8)(delayed(compute_distance_to_centroids)(centroids_xy[si:ei], 
                                                                        centroids_textures[si:ei], 
                                                                        spacing=spacing, w_spatial=w_spatial, 
                                                                        h=h, w=w, 
                                                                        ymins=window_ymins[si:ei], 
                                                                        ymaxs=window_ymaxs[si:ei], 
                                                                        xmins=window_xmins[si:ei], 
                                                                        xmaxs=window_xmaxs[si:ei],
                                                                        window_spatial_distances=window_spatial_distances)
                                    for si, ei in zip(np.arange(0, n_superpixels, number_sps_per_worker), 
                                        np.arange(0, n_superpixels, number_sps_per_worker) + number_sps_per_worker))

        sys.stderr.write('done in %.2f seconds\n' % (time.time() - t))
        os.system('rm -r /dev/shm/joblib*')

        t = time.time()

        sys.stderr.write('aggregate\n')

        for sp_i, new_ds in enumerate(chain(*res)):
            
            ymin = window_ymins[sp_i]
            xmin = window_xmins[sp_i]
            ymax = window_ymaxs[sp_i]
            xmax = window_xmaxs[sp_i]

            new_distances = new_ds.reshape((ymax+1-ymin, xmax+1-xmin))

            to_update = np.zeros((ymax+1-ymin, xmax+1-xmin), np.bool)

            valid_pixels = ~ np.isnan(new_distances)

            to_update[valid_pixels] = (new_distances[valid_pixels] < distances[ymin:ymax+1, xmin:xmax+1][valid_pixels])

            # if np.count_nonzero(to_update[valid_pixels]) == 0:
            #     print sp_i
            #     print new_distances[valid_pixels]
            #     print distances[ymin:ymax+1, xmin:xmax+1][valid_pixels]
            #     print 

            distances[ymin:ymax+1, xmin:xmax+1][to_update] = new_distances[to_update]
            assignments[ymin:ymax+1, xmin:xmax+1][to_update] = sp_i
    
        del res, distances, valid_pixels, to_update, new_distances, new_ds

        sys.stderr.write('done in %.2f seconds\n' % (time.time() - t))
                
        sys.stderr.write('update assignment\n')
        t = time.time()

        # print len(np.unique(assignments)), np.max(assignments), np.min(assignments)

        props = regionprops(assignments+1)
        sp_coords = map(attrgetter('coords'), props)
        sp_centroid = np.asarray(map(attrgetter('centroid'), props))
        
        centroids_textures = [hist_map[coords[:,0], coords[:,1]].mean(axis=0) for coords in sp_coords]
        
        centroids_xy_new = sp_centroid[:, ::-1]

        sys.stderr.write('total centroid location change = %d\n' % 
                         np.sum(np.abs(centroids_xy_new - centroids_xy)))

        centroids_xy = centroids_xy_new

        sys.stderr.write('done in %.2f seconds\n' % (time.time() - t))

        del props, sp_coords, sp_centroid


    import gc
    collected = gc.collect()
    print "Garbage collector: collected %d objects." % (collected)

    sys.stderr.write('enforce connectivity\n')
    t = time.time()

    # connect broken regions
    sp_area_limit = 2000

    assignments = label(assignments, connectivity=1)

    sp_all_props = regionprops(assignments + 1, cache=True)
        # (row, col), a, (min_row, min_col, max_row, max_col),(rows, cols)

    # sp_props = Parallel(n_jobs=1)(delayed(_obtain_props_worker)(spp) for spp in sp_all_props)
    # sp_areas, sp_bbox, spp_coords = map(np.asarray, zip(*sp_props))

    sp_areas = np.asarray([spp.area for spp in sp_all_props])
    # sp_bboxes = [spp.bbox for spp in sp_all_props]
    # spp_coords = np.asarray([spp.coords for spp in sp_all_props])

    for i in np.where(sp_areas < sp_area_limit)[0]:
        min_row, min_col, max_row, max_col = sp_all_props[i].bbox
        c = Counter([assignments[min(h-1, max_row+5), min(w-1, max_col+5)], 
            assignments[max(0, min_row-5), max(0, min_col-5)], 
            assignments[max(0, min_row-5), min(w-1, max_col+5)], 
            assignments[min(h-1, max_row+5), max(0, min_col-5)]])
        assignments[sp_all_props[i].coords[:,0], sp_all_props[i].coords[:,1]] = c.most_common()[0][0]


    sys.stderr.write('done in %.2f seconds\n' % (time.time() - t))

    # sp_all_props = regionprops(component_labels + 1, cache=True)
    # # sp_props = Parallel(n_jobs=1)(delayed(_obtain_props_worker)(spp) for spp in sp_all_props)
    # # sp_areas, _, _ = map(np.asarray, zip(*sp_props))

    # sp_areas = np.asarray([spp.area for spp in sp_all_props])
    # sp_bboxes = np.asarray([spp.bbox for spp in sp_all_props])
    # sp_coords = np.asarray([spp.coords for spp in sp_all_props])

    # assert np.all(sp_areas > sp_area_limit)

    return assignments

# def _obtain_props_worker(spp):
#     return spp.area, spp.bbox, spp.coords

# def enforce_connectivity(seg, sp_area_limit=2000):

#     from skimage.measure import label, regionprops
#     from collections import Counter

#     h, w = seg.shape[:2]

#     component_labels = label(seg, connectivity=1)
#     sp_all_props = regionprops(component_labels + 1, cache=True)
#         # (row, col), a, (min_row, min_col, max_row, max_col),(rows, cols)

#     sp_props = Parallel(n_jobs=1)(delayed(_obtain_props_worker)(spp) for spp in sp_all_props)
#     sp_areas, sp_bbox, spp_coords = map(np.asarray, zip(*sp_props))

#     for i in np.where(sp_areas < sp_area_limit)[0]:
#         min_row, min_col, max_row, max_col = sp_bbox[i]
#         c = Counter([component_labels[min(h-1, max_row+5), min(w-1, max_col+5)], 
#             component_labels[max(0, min_row-5), max(0, min_col-5)], 
#             component_labels[max(0, min_row-5), min(w-1, max_col+5)], 
#             component_labels[min(h-1, max_row+5), max(0, min_col-5)]])
#         component_labels[spp_coords[i][:,0], spp_coords[i][:,1]] = c.most_common()[0][0]

#     sp_all_props = regionprops(component_labels + 1, cache=True)
#     sp_props = Parallel(n_jobs=1)(delayed(_obtain_props_worker)(spp) for spp in sp_all_props)
#     sp_areas, _, _ = map(np.asarray, zip(*sp_props))

#     assert np.all(sp_areas > sp_area_limit)

#     return component_labels
