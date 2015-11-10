import numpy as np
from joblib import Parallel, delayed
from utilities2015 import *
import time

from itertools import chain
from operator import itemgetter, attrgetter

def compute_distance_to_centroids(centroids_xy, centroids_texture, spacing, w_spatial, hist_map, h, w,
                                 ymins, ymaxs, xmins, xmaxs, window_spatial_distances):
    
    n = len(centroids_xy)
    
    ds = [None for _ in range(n)]
        
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
                           hist_map[ymin:ymax+1, xmin:xmax+1].reshape((-1, hist_map.shape[-1])))
        
        ds[ci] = w_spatial * spatial_ds + texture_ds
            
    return ds

# def compute_new_centroids(sps, assignments, hist_map):
#     centroids = [None for _ in range(len(sps))]
#     for i, sp_i in enumerate(sps):
#         rs, cs = np.where(assignments == sp_i)
#         centroids[i] = np.r_[np.c_[rs, cs].mean(axis=0)[::-1], hist_map[rs, cs].mean(axis=0)]
#     return centroids

def slic_texture(hist_map, spacing=200, w_spatial=0.001, max_iter=5):

    h, w, n_texton = hist_map.shape

    sp_ys, sp_xs = np.mgrid[0:h:spacing, 0:w:spacing]
    
    n_superpixels = len(sp_ys.flat)
    
    centroids_textures = hist_map[0:h:spacing, 0:w:spacing].reshape((-1, n_texton))
    centroids_xy = np.c_[sp_xs.flat, sp_ys.flat]

    ys, xs = np.mgrid[-spacing:spacing+1, -spacing:spacing+1]
    window_spatial_distances = np.sqrt(ys**2 + xs**2)
    
    for iter_i in range(max_iter):

        print 'iteration', iter_i

        cx = centroids_xy[:, 0].astype(np.int)
        cy = centroids_xy[:, 1].astype(np.int)
        window_ymins = np.maximum(0, cy - spacing)
        window_xmins = np.maximum(0, cx - spacing)
        window_ymaxs = np.minimum(h-1, cy + spacing)
        window_xmaxs = np.minimum(w-1, cx + spacing)
                
        assignments = -1 * np.ones((h, w), np.int16)
        distances = np.inf * np.ones((h, w), np.float16)

        sys.stderr.write('%d superpixels\n'%n_superpixels)

        t = time.time()            

        sys.stderr.write('compute distance\n')
        
        res = Parallel(n_jobs=16)(delayed(compute_distance_to_centroids)(centroids_xy[si:ei], 
                                                                         centroids_textures[si:ei], 
                                                                         spacing=spacing, w_spatial=w_spatial, 
                                                                         hist_map=hist_map, h=h, w=w, 
                                                                         ymins=window_ymins[si:ei], 
                                                                         ymaxs=window_ymaxs[si:ei], 
                                                                         xmins=window_xmins[si:ei], 
                                                                         xmaxs=window_xmaxs[si:ei],
                                                window_spatial_distances=window_spatial_distances)
                                    for si, ei in zip(np.arange(0, n_superpixels, n_superpixels/128), 
                                        np.arange(0, n_superpixels, n_superpixels/128) + n_superpixels/128))

        sys.stderr.write('done in %.2f seconds\n' % (time.time() - t))

        t = time.time()

        sys.stderr.write('aggregate\n')

        for sp_i, new_ds in enumerate(chain(*res)):
            
            ymin = window_ymins[sp_i]
            xmin = window_xmins[sp_i]
            ymax = window_ymaxs[sp_i]
            xmax = window_xmaxs[sp_i]

            q = new_ds.reshape((ymax+1-ymin, xmax+1-xmin))
            s = q < distances[ymin:ymax+1, xmin:xmax+1]

            distances[ymin:ymax+1, xmin:xmax+1][s] = q[s]
            assignments[ymin:ymax+1, xmin:xmax+1][s] = sp_i
    
        del res

        sys.stderr.write('done in %.2f seconds\n' % (time.time() - t))
        
                
        sys.stderr.write('update assignment\n')
        t = time.time()

        props = regionprops(assignments+1)
        sp_coords = map(attrgetter('coords'), props)
        sp_centroid = np.asarray(map(attrgetter('centroid'), props))
        
        centroids_textures = [hist_map[coords[:,0], coords[:,1]].mean(axis=0) for coords in sp_coords]
        
        centroids_xy_new = sp_centroid[:, ::-1]

        sys.stderr.write('total centroid location change = %d\n' % 
                         np.sum(np.abs(centroids_xy_new - centroids_xy)))

        centroids_xy = centroids_xy_new

        sys.stderr.write('done in %.2f seconds\n' % (time.time() - t))

    return assignments


def _obtain_props_worker(spp):
    return spp.area, spp.bbox, spp.coords

def enforce_connectivity(seg, sp_area_limit=2000):

    from skimage.measure import label, regionprops
    from collections import Counter

    h, w = seg.shape[:2]

    component_labels = label(seg, connectivity=1)
    sp_all_props = regionprops(component_labels + 1, cache=True)
        # (row, col), a, (min_row, min_col, max_row, max_col),(rows, cols)

    sp_props = Parallel(n_jobs=16)(delayed(_obtain_props_worker)(spp) for spp in sp_all_props)
    sp_areas, sp_bbox, spp_coords = map(np.asarray, zip(*sp_props))

    for i in np.where(sp_areas < sp_area_limit)[0]:
        min_row, min_col, max_row, max_col = sp_bbox[i]
        c = Counter([component_labels[min(h-1, max_row+5), min(w-1, max_col+5)], 
            component_labels[max(0, min_row-5), max(0, min_col-5)], 
            component_labels[max(0, min_row-5), min(w-1, max_col+5)], 
            component_labels[min(h-1, max_row+5), max(0, min_col-5)]])
        component_labels[spp_coords[i][:,0], spp_coords[i][:,1]] = c.most_common()[0][0]

    sp_all_props = regionprops(component_labels + 1, cache=True)
    sp_props = Parallel(n_jobs=16)(delayed(_obtain_props_worker)(spp) for spp in sp_all_props)
    sp_areas, _, _ = map(np.asarray, zip(*sp_props))

    assert np.all(sp_areas > sp_area_limit)

    return component_labels
