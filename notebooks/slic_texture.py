import numpy as np
from joblib import Parallel, delayed
from utilities2015 import *
import time

def compute_distance_to_centroids(centroids, spacing, w_spatial, hist_map, h, w):
    ds = [None for _ in range(centroids.shape[0])]
    
    for centroid_ind, centroid in enumerate(centroids):
        cx = int(centroid[0])
        cy = int(centroid[1])
        ch = centroid[2:]

        ymin = max(0, cy - 2*spacing)
        xmin = max(0, cx - 2*spacing)
        ymax = min(h-1, cy + 2*spacing)
        xmax = min(w-1, cx + 2*spacing)

        ys, xs = np.mgrid[ymin:ymax+1, xmin:xmax+1].astype(np.int)
        spatial_ds = np.squeeze(cdist([[cx, cy]], np.c_[xs.flat, ys.flat], 'euclidean'))

        texture_ds = chi2s([ch], hist_map[ys.flat, xs.flat])
        ds[centroid_ind] = w_spatial * spatial_ds + texture_ds

    return ds

def compute_new_centroids(sps, assignments, hist_map):
    centroids = [None for _ in range(len(sps))]
    for i, sp_i in enumerate(sps):
        rs, cs = np.where(assignments == sp_i)
        centroids[i] = np.r_[np.c_[rs, cs].mean(axis=0)[::-1], hist_map[rs, cs].mean(axis=0)]
    return centroids

def slic_texture(hist_map, spacing=200, w_spatial=0.001, max_iter=1):

    h, w = hist_map.shape[:2]

    from itertools import chain
    from operator import itemgetter

    sp_ys, sp_xs = np.mgrid[0:h:spacing, 0:w:spacing]
    sp_texhists = hist_map[sp_ys.flat, sp_xs.flat]
    centroids = np.c_[sp_xs.flat, sp_ys.flat, sp_texhists]
    n_superpixels = centroids.shape[0]

    for iter_i in range(max_iter):

        print 'iteration', iter_i

        assignments = -1 * np.ones((h, w), np.int16)
        distances = np.inf * np.ones((h, w), np.float16)

        sys.stderr.write('compute_distance_to_centroids\n')
        t = time.time()

        for i in range(0, n_superpixels, 500):

            res = Parallel(n_jobs=16)(delayed(compute_distance_to_centroids)(centroids_p, 
                                                                             spacing=spacing,
                                                                            w_spatial=w_spatial,
                                                                            hist_map=hist_map,
                                                                            h=h, w=w) 
                                      for centroids_p in np.array_split(centroids[i:i+500], 16))

            new_dists = list(chain(*res))
            
            for sp_i, nds in enumerate(new_dists):

                cx = int(centroids[i+sp_i, 0])
                cy = int(centroids[i+sp_i, 1])

                ymin = max(0, cy - 2*spacing)
                xmin = max(0, cx - 2*spacing)
                ymax = min(h-1, cy + 2*spacing)
                xmax = min(w-1, cx + 2*spacing)

                ys, xs = np.mgrid[ymin:ymax+1, xmin:xmax+1].astype(np.int)
                cls = np.c_[ys.flat, xs.flat]

                s = nds < distances[cls[:,0], cls[:,1]]
                distances[cls[s,0], cls[s,1]] = nds[s]
                assignments[cls[s,0], cls[s,1]] = i + sp_i

            del res
            del new_dists

        sys.stderr.write('done in %.2f seconds\n' % (time.time() - t))


        sys.stderr.write('update assignment\n')
        t = time.time()

        centroids_part = Parallel(n_jobs=16)(delayed(compute_new_centroids)(sps, assignments=assignments,
                                                                           hist_map=hist_map) 
                                             for sps in np.array_split(range(n_superpixels), 16))
        centroids_new = np.vstack(centroids_part)

        print 'total centroid location change', np.sum(np.abs(centroids_new[:,:2] - centroids[:,:2]))

        centroids = centroids_new

        sys.stderr.write('done in %.2f seconds\n' % (time.time() - t))

    return assignments

def _obtain_props_worker(spp):
    return spp.area, spp.bbox, spp.coords

def enforce_connectivity(seg):

	from skimage.measure import label, regionprops
	from collections import Counter

	h, w = seg.shape[:2]

	component_labels = label(seg)
	sp_all_props = regionprops(component_labels + 1, cache=True)
	    # (row, col), a, (min_row, min_col, max_row, max_col),(rows, cols)

	sp_props = Parallel(n_jobs=16)(delayed(_obtain_props_worker)(spp) for spp in sp_all_props)
	sp_areas, sp_bbox, spp_coords = map(np.asarray, zip(*sp_props))

	for i in np.where(sp_areas < 2000)[0]:
	    min_row, min_col, max_row, max_col = sp_bbox[i]
	    c = Counter([component_labels[min(h-1, max_row+5), min(w-1, max_col+5)], 
	    	component_labels[max(0, min_row-5), max(0, min_col-5)], 
	        component_labels[max(0, min_row-5), min(w-1, max_col+5)], 
	        component_labels[min(h-1, max_row+5), max(0, min_col-5)]])
	    component_labels[spp_coords[i][:,0], spp_coords[i][:,1]] = c.most_common()[0][0]

	return component_labels
