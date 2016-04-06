import numpy as np
import sys
import os

from skimage.morphology import binary_closing, disk, binary_dilation, binary_erosion, remove_small_holes
from skimage.measure import grid_points_in_poly, subdivide_polygon, approximate_polygon
from skimage.measure import find_contours, regionprops

from shapely.geometry import Polygon

import cv2

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *

def find_contour_points(labelmap):
    '''
    return is (x,y)
    '''

    regions = regionprops(labelmap)

    contour_points = {}

    for r in regions:

        (min_row, min_col, max_row, max_col) = r.bbox

        padded = np.pad(r.filled_image, ((5,5),(5,5)), mode='constant', constant_values=0)

        contours = find_contours(padded, .5, fully_connected='high')
        contours = [cnt.astype(np.int) for cnt in contours if len(cnt) > 10]
        if len(contours) > 0:
#             if len(contours) > 1:
#                 sys.stderr.write('%d: region has more than one part\n' % r.label)
                
            contours = sorted(contours, key=lambda c: len(c), reverse=True)
            contours_list = [c-(5,5) for c in contours]
            contour_points[r.label] = sorted([c[np.arange(0, c.shape[0], 10)][:, ::-1] + (min_col, min_row) 
                                for c in contours_list], key=lambda c: len(c), reverse=True)
            
        elif len(contours) == 0:
#             sys.stderr.write('no contour is found\n')
            continue

    #         viz = np.zeros_like(r.filled_image)
    #         viz[pts_sampled[:,0], pts_sampled[:,1]] = 1
    #         plt.imshow(viz, cmap=plt.cm.gray);
    #         plt.show();
        
    return contour_points


def show_contours(cnts, bg, title):
    viz = bg.copy()
    for cnt in cnts:
        for c in cnt:
            cv2.circle(viz, tuple(c.astype(np.int)), 1, (0,255,0), -1)
        cv2.polylines(viz, [cnt.astype(np.int)], True, (0,255,0), 2)
        
    plt.figure(figsize=(10,10));
    plt.imshow(viz);
#     plt.title(title);
    plt.axis('off');
    plt.show();
    
def show_levelset(levelset, bg, title):
    viz = bg.copy()
    cnts = find_contours(levelset, .5)
    for cnt in cnts:
        for c in cnt[:,::-1]:
            cv2.circle(viz, tuple(c.astype(np.int)), 1, (0,255,0), -1)
    plt.figure(figsize=(10,10));
    plt.imshow(viz);
#     plt.title(title);
    plt.axis('off');
    plt.show();
    
# http://deparkes.co.uk/2015/02/01/find-concave-hull-python/
# http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/

from shapely.ops import cascaded_union, polygonize
from shapely.geometry import MultiLineString
from scipy.spatial import Delaunay
import numpy as np

def alpha_shape(coords, alphas):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    
    tri = Delaunay(coords)
    
    pa = coords[tri.vertices[:,0]]
    pb = coords[tri.vertices[:,1]]
    pc = coords[tri.vertices[:,2]]
    
    a = np.sqrt(np.sum((pa - pb)**2, axis=1))
    b = np.sqrt(np.sum((pb - pc)**2, axis=1))
    c = np.sqrt(np.sum((pc - pa)**2, axis=1))
    s = (a + b + c)/2.
    area = np.sqrt(s*(s-a)*(s-b)*(s-c))
    circum_r = a*b*c/(4.0*area)
        
    geoms = []
        
    for al in alphas:
        edges = tri.vertices[circum_r < 1.0 / al]

        edge_points = []
        for ia, ib, ic in edges:
            edge_points.append(coords[ [ia, ib] ])
            edge_points.append(coords[ [ib, ic] ])
            edge_points.append(coords[ [ic, ia] ])

        m = MultiLineString(edge_points)
        triangles = list(polygonize(m))
        r = cascaded_union(triangles)
        
        geoms.append(r)

        
#     edges = tri.vertices[circum_r < 1.0/alpha]

# # slightly slower than below
# #     edge_points = list(chain(*[[coords[ [ia, ib] ], coords[ [ib, ic] ], coords[ [ic, ia] ]]
# #                    for ia, ib, ic in edges]))
    
#     edge_points = []
#     for ia, ib, ic in edges:
#         edge_points.append(coords[ [ia, ib] ])
#         edge_points.append(coords[ [ib, ic] ])
#         edge_points.append(coords[ [ic, ia] ])

#     m = MultiLineString(edge_points)
#     triangles = list(polygonize(m))
#     r = cascaded_union(triangles)
    
    return geoms

def less(center):
    def less_helper(a, b):
        if (a[0] - center[0] >= 0 and b[0] - center[0] < 0):
            return 1;
        if (a[0] - center[0] < 0 and b[0] - center[0] >= 0):
            return -1;
        if (a[0] - center[0] == 0 and b[0] - center[0] == 0):
            if (a[1] - center[1] >= 0 or b[1] - center[1] >= 0):
                return 2*int(a[1] > b[1]) - 1;
            return 2*int(b[1] > a[1]) - 1

        # compute the cross product of vectors (center -> a) x (center -> b)
        det = (a[0] - center[0]) * (b[1] - center[1]) - (b[0] - center[0]) * (a[1] - center[1])
        if (det < 0):
            return 1;
        if (det > 0):
            return -1;

        # points a and b are on the same line from the center
        # check which point is closer to the center
        d1 = (a[0] - center[0]) * (a[0] - center[0]) + (a[1] - center[1]) * (a[1] - center[1])
        d2 = (b[0] - center[0]) * (b[0] - center[0]) + (b[1] - center[1]) * (b[1] - center[1])
        return 2*int(d1 > d2) - 1
    
    return less_helper

def sort_vertices_counterclockwise(cnt):
    # http://stackoverflow.com/a/6989383
    center = cnt.mean(axis=0)
    return sorted(cnt, cmp=less(center))


def contour_to_concave_hull(cnt, levelset, alphas):
    
    xmin, ymin = cnt.min(axis=0)
    xmax, ymax = cnt.max(axis=0)
    
#     if levelset is None:
    
#         h, w = (ymax-ymin+1, xmax-xmin+1)
#         inside_ys, inside_xs = np.where(grid_points_in_poly((h, w), cnt[:, ::-1]-(ymin,xmin))) 
#         n = inside_ys.size
#         random_indices = np.random.choice(range(n), min(5000, n), replace=False)
#         inside_points = np.c_[inside_xs[random_indices], inside_ys[random_indices]] + (xmin, ymin)

#     else:
        
    xs, ys = np.meshgrid(np.arange(xmin, xmax+1), np.arange(ymin, ymax+1))
    gridpoints = np.c_[xs.flat, ys.flat]
    inside_indices = np.where(levelset[gridpoints[:,1], gridpoints[:,0]] > 0)[0]
    n = inside_indices.size
    random_indices = np.random.choice(range(n), min(3000, n), replace=False)
    inside_points = gridpoints[inside_indices[random_indices]]
        

    geoms = alpha_shape(inside_points, alphas)
    
    base_area = np.sum(levelset)
    errs = np.array([(r.area if r.type == 'Polygon' else max([rr.area for rr in r])) - base_area for r in geoms])
    
#     plt.plot(errs);
#     plt.xticks(range(len(errs)), alphas);
#     plt.show();
    
#     plt.plot(np.abs(errs));
#     plt.xticks(range(len(errs)), alphas);
#     plt.show();
    
    c = np.argmin(np.abs(errs))
    r = geoms[c]
    
#     num_comps = np.array([1 if r.type == 'Polygon' else len(r) for r in geoms])
#     n = num_comps[-1]
#     while True:
#         c = np.min(np.where((num_comps == n) & (errs > 0)))
#         if errs[c] < 1e5:
#             break
#         n += 1
    
    if r.type == 'Polygon':
        concave_hull = r
    else:
        concave_hull = r[np.argmax([rr.area for rr in r])]
    
    # the heuristic rule here is:
    # merge two parts into one if the loss of including extraneous area is not larger 
    # than the loss of sacrificing all parts other than the largest one
    
    if not hasattr(concave_hull, 'exterior'):
        sys.stderr.write('No concave hull produced.\n')
        return None

    if concave_hull.exterior.length < 20 * 3:
        point_interval = concave_hull.exterior.length / 4
    else:
        point_interval = 20
    new_cnt_subsampled = np.array([concave_hull.exterior.interpolate(r, normalized=True).coords[:][0] 
                         for r in np.arange(0, 1, point_interval/concave_hull.exterior.length)], 
               dtype=np.int)

    return new_cnt_subsampled, alphas[c]


def pad_scoremap(stack, sec, l, scoremaps_rootdir, bg_size):

    scoremaps_dir = os.path.join(scoremaps_rootdir, stack, '%04d'%sec)

    try:
#         scoremap_whole = bp.unpack_ndarray_file(os.path.join(scoremaps_dir, 
#                                                    '%(stack)s_%(sec)04d_roi1_denseScoreMapLossless_%(label)s.bp' % \
#                                                    {'stack': stack, 'sec': sec, 'label': l}))

        scoremap_whole = load_hdf(os.path.join(scoremaps_dir, 
                                                   '%(stack)s_%(sec)04d_roi1_denseScoreMapLossless_%(label)s.hdf' % \
                                                   {'stack': stack, 'sec': sec, 'label': l}))
    
    except:
        sys.stderr.write('No scoremap of %s exists\n' % (l))
        return None


    dataset = stack + '_' + '%04d'%sec + '_roi1'

    interpolation_xmin, interpolation_xmax, \
    interpolation_ymin, interpolation_ymax = np.loadtxt(os.path.join(scoremaps_dir, 
                                                                     '%(dataset)s_denseScoreMapLossless_%(label)s_interpBox.txt' % \
                                    {'dataset': dataset, 'label': l})).astype(np.int)

    h, w = bg_size
    
    dense_scoremap_lossless = np.zeros((h, w), np.float32)
    dense_scoremap_lossless[interpolation_ymin:interpolation_ymax+1,
                            interpolation_xmin:interpolation_xmax+1] = scoremap_whole

    return dense_scoremap_lossless

def load_initial_contours(initCnts_dir, 
                          stack,
                          test_volume_atlas_projected=None, 
                          force=True,
                          z_xy_ratio_downsampled=None,
                         volume_limits=None,
                         labels=None):

    if os.path.exists(initCnts_dir + '/initCntsAllSecs_%s.pkl' % stack) and not force:

        init_cnts_allSecs = pickle.load(open(initCnts_dir + '/initCntsAllSecs_%s.pkl' % stack, 'r'))

    else:

        ngbr = 3
        sss = np.empty((2*ngbr,), np.int)
        sss[1::2] = -np.arange(1, ngbr+1)
        sss[::2] = np.arange(1, ngbr+1)

        init_cnts_allSecs = {}

        first_detect_sec, last_detect_sec = detect_bbox_range_lookup[stack]
        
        volume_xmin, volume_xmax, volume_ymin, volume_ymax, volume_zmin, volume_zmax = volume_limits
        
        for sec in range(first_detect_sec, last_detect_sec+1):

            z = int(z_xy_ratio_downsampled*sec) - volume_zmin

            projected_annotation_labelmap = test_volume_atlas_projected[..., z]

            init_cnts = find_contour_points(projected_annotation_labelmap) # downsampled 16
            init_cnts = dict([(labels[label_ind], (cnts[0]+(volume_xmin, volume_ymin))*2) 
                              for label_ind, cnts in init_cnts.iteritems()])

            # copy annotations of undetected classes from neighbors
            Ls = []
            for ss in sss:
                sec2 = sec + ss
                z2 = int(z_xy_ratio_downsampled*sec2) - volume_zmin
                if z2 >= test_volume_atlas_projected.shape[2] or z2 < 0:
                    continue

                init_cnts2 = find_contour_points(test_volume_atlas_projected[..., z2]) # downsampled 16
                init_cnts2 = dict([(labels[label_ind], (cnts[0]+(volume_xmin, volume_ymin))*2) 
                                  for label_ind, cnts in init_cnts2.iteritems()])
                Ls.append(init_cnts2)

            for ll in Ls:
                for l, c in ll.iteritems():
                    if l not in init_cnts:
                        init_cnts[l] = c

            init_cnts_allSecs[sec] = init_cnts

        pickle.dump(init_cnts_allSecs, open(initCnts_dir + '/initCntsAllSecs_%s.pkl' % stack, 'w'))
        
    return init_cnts_allSecs


def surr_points(arr):
    poly = Polygon(arr)
    p1 = points_in_polygon(list(poly.buffer(10, resolution=2).exterior.coords))
    p2 = points_in_polygon(list(poly.exterior.coords))
    surr_pts = pts_arr_setdiff(p1, p2)
    return surr_pts

def points_in_polygon(polygon):
    pts = np.array(polygon, np.int)
    xmin, ymin = pts.min(axis=0)
    ymax, ymax = pts.max(axis=0)
    nz_ys, nz_xs =np.where(grid_points_in_poly((ymax-ymin+1, xmax-xmin+1), pts-[xmin, ymin]))
    nz2 = np.c_[nz_xs + xmin, nz_ys + ymin]
    return nz2

def pts_arr_setdiff(nz1, nz2):
    # http://stackoverflow.com/a/11903368
    a1_rows = nz1.view([('', nz1.dtype)] * nz1.shape[1])
    a2_rows = nz2.view([('', nz2.dtype)] * nz2.shape[1])
    surr_nzs = np.setdiff1d(a1_rows, a2_rows).view(nz1.dtype).reshape(-1, nz1.shape[1])
    return surr_nzs