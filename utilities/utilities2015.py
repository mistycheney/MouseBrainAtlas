import matplotlib
matplotlib.use('Agg')

import os
import csv
import sys
from operator import itemgetter
from subprocess import check_output, call
import json
import cPickle as pickle
import datetime

from skimage.io import imread, imsave
from skimage.filters import threshold_otsu, threshold_adaptive, gaussian_filter
from skimage.color import color_dict, gray2rgb, label2rgb, rgb2gray
from skimage.segmentation import clear_border
from skimage.morphology import binary_dilation, binary_erosion, watershed, remove_small_objects
from skimage.measure import regionprops, label
from skimage.restoration import denoise_bilateral
from skimage.util import img_as_ubyte, img_as_float
from skimage.transform import rescale
from scipy.spatial.distance import cdist, pdist
import numpy as np
import matplotlib.pyplot as plt
try:
    import cv2
except:
    sys.stderr.write('Cannot load cv2.\n')
#from tables import open_file, Filters, Atom
import bloscpack as bp

from ipywidgets import FloatProgress
from IPython.display import display


def shell_escape(s):
    """
    Escape a string (treat it as a single complete string) in shell commands.
    """
    from tempfile import mkstemp
    fd, path = mkstemp()
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(s)
        cmd = r"""cat %s | sed -e "s/'/'\\\\''/g; 1s/^/'/; \$s/\$/'/" """ % path
        escaped_str = check_output(cmd, shell=True)
    finally:
        os.remove(path)

    return escaped_str

def plot_histograms(hists, bins, titles=None, ncols=4, xlabel='', ylabel='', suptitle='', normalize=False, cellsize=(2, 1.5), **kwargs):
    """
    cellsize: (w,h) for each cell
    """

    if isinstance(hists, dict):
        titles = hists.keys()
        hists = hists.values()

    if normalize:
        hists = hists/np.sum(hists, axis=1).astype(np.float)[:,None]

    if titles is None:
        titles = ['' for _ in range(len(hists))]

    n = len(hists)
    nrows = int(np.ceil(n/float(ncols)))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, \
                            figsize=(ncols*cellsize[0], nrows*cellsize[1]), **kwargs)
    axes = axes.flatten()

    width = np.abs(np.mean(np.diff(bins)))*.8

    c = 0
    for name_u, h in zip(titles, hists):

        axes[c].bar(bins, h/float(h.sum()), width=width);
        axes[c].set_xlabel(xlabel);
        axes[c].set_ylabel(ylabel);
        axes[c].set_title(name_u);
        c += 1

    for i in range(c, len(axes)):
        axes[i].axis('off')

    plt.suptitle(suptitle);
    plt.tight_layout();
    plt.show()

def save_pickle(obj, fp):
    with open(fp, 'w') as f:
        pickle.dump(obj, f)

def save_json(obj, fp):
    with open(fp, 'w') as f:
        json.dump(obj, f)

def load_json(fp):
    with open(fp, 'r') as f:
        return json.load(f)

def load_pickle(fp):
    with open(fp, 'r') as f:
        obj = pickle.load(f)

    return obj

def one_liner_to_arr(line, func):
    return np.array(map(func, line.strip().split()))

def array_to_one_liner(arr):
    return ' '.join(map(str, arr)) + '\n'

def show_progress_bar(min, max):
    bar = FloatProgress(min=min, max=max)
    display(bar)
    return bar

from enum import Enum

class PolygonType(Enum):
    CLOSED = 'closed'
    OPEN = 'open'
    TEXTURE = 'textured'
    TEXTURE_WITH_CONTOUR = 'texture with contour'
    DIRECTION = 'directionality'

def create_parent_dir_if_not_exists(fp):
    create_if_not_exists(os.path.dirname(fp))

def create_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def execute_command(cmd):
    print cmd

    try:
        retcode = call(cmd, shell=True)
        if retcode < 0:
            print >>sys.stderr, "Child was terminated by signal", -retcode
        else:
            print >>sys.stderr, "Child returned", retcode
        return retcode
    except OSError as e:
        print >>sys.stderr, "Execution failed:", e
        raise e

def draw_arrow(image, p, q, color, arrow_magnitude=9, thickness=5, line_type=8, shift=0):
    # adapted from http://mlikihazar.blogspot.com.au/2013/02/draw-arrow-opencv.html

    import cv2

    # draw arrow tail
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # calc angle of the arrow
    angle = np.arctan2(p[1]-q[1], p[0]-q[0])
    # starting point of first line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle + np.pi/4)))
    # draw first half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # starting point of second line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle - np.pi/4)))
    # draw second half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)


def save_hdf_v2(data, fn, key='data'):
    import pandas
    create_parent_dir_if_not_exists(fn)
    pandas.Series(data=data).to_hdf(fn, key, mode='w')

def load_hdf_v2(fn, key='data'):
    import pandas
    return pandas.read_hdf(fn, key)

def save_hdf(data, fn, complevel=9, key='data'):
    filters = Filters(complevel=complevel, complib='blosc')
    with open_file(fn, mode="w") as f:
        _ = f.create_carray('/', key, Atom.from_dtype(data.dtype), filters=filters, obj=data)

def load_hdf(fn, key='data'):
    """
    Used by loading features.
    """
    with open_file(fn, mode="r") as f:
        data = f.get_node('/'+key).read()
    return data


def unique_rows(a, return_index=True):
    # http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    unique_a = a[idx]
    if return_index:
        return unique_a, idx
    else:
        return unique_a

def unique_rows2(a):
    ind = np.lexsort(a.T)
    return a[np.concatenate(([True],np.any(a[ind[1:]]!=a[ind[:-1]],axis=1)))]


def order_nodes(sps, neighbor_graph, verbose=False):

    from networkx.algorithms import dfs_successors, dfs_postorder_nodes


    subg = neighbor_graph.subgraph(sps)
    d_suc = dfs_successors(subg)

    x = [(a,b) for a,b in d_suc.iteritems() if len(b) == 2]

    if verbose:
        print 'root, two_leaves', x

    if len(x) == 0:
        trav = list(dfs_postorder_nodes(subg))
    else:
        if verbose:
            print 'd_succ'
            for it in d_suc.iteritems():
                print it

        root, two_leaves = x[0]

        left_branch = []
        right_branch = []

        c = two_leaves[0]
        left_branch.append(c)
        while c in d_suc:
            c = d_suc[c][0]
            left_branch.append(c)

        c = two_leaves[1]
        right_branch.append(c)
        while c in d_suc:
            c = d_suc[c][0]
            right_branch.append(c)

        trav = left_branch[::-1] + [root] + right_branch

        if verbose:
            print 'left_branch', left_branch
            print 'right_branch', right_branch

    return trav

def find_score_peaks(scores, min_size = 4, min_distance=10, threshold_rel=.3, threshold_abs=0, peakedness_lim=0,
                    peakedness_radius=1, verbose=False):

    from skimage.feature import peak_local_max

    scores2 = scores.copy()
    scores2[np.isnan(scores)] = np.nanmin(scores)
    scores = scores2

    if len(scores) > min_size:

        scores_shifted = scores[min_size-1:]
        scores_shifted_positive = scores_shifted - scores_shifted.min()

        peaks_shifted = np.atleast_1d(np.squeeze(peak_local_max(scores_shifted_positive,
                                    min_distance=min_distance, threshold_abs=threshold_abs-scores_shifted.min(), exclude_border=False)))

        # print peaks_shifted

        if len(peaks_shifted) == 0:
            high_peaks_sorted = np.array([np.argmax(scores)], np.int)
            high_peaks_peakedness = np.inf

        else:
            peaks_shifted = peaks_shifted[scores_shifted[peaks_shifted] >= np.max(scores_shifted) - threshold_rel]
            peaks_shifted = np.unique(np.r_[peaks_shifted, np.argmax(scores_shifted)])

            if verbose:
                print 'raw peaks', np.atleast_1d(np.squeeze(min_size - 1 + peaks_shifted))

            if len(peaks_shifted) > 0:
                peaks = min_size - 1 + peaks_shifted
            else:
                peaks = np.array([np.argmax(scores[min_size-1:]) + min_size-1], np.int)

            peakedness = np.zeros((len(peaks),))
            for i, p in enumerate(peaks):
                nbrs = np.r_[scores[max(min_size-1, p-peakedness_radius):p], scores[p+1:min(len(scores), p+1+peakedness_radius)]]
                assert len(nbrs) > 0
                peakedness[i] = scores[p]-np.mean(nbrs)

            if verbose:
                print 'peakedness', peakedness
                print 'filtered peaks', np.atleast_1d(np.squeeze(peaks))

            high_peaks = peaks[peakedness > peakedness_lim]
            high_peaks = np.unique(np.r_[high_peaks, min_size - 1 + np.argmax(scores_shifted)])

            high_peaks_order = scores[high_peaks].argsort()[::-1]
            high_peaks_sorted = high_peaks[high_peaks_order]

            high_peaks_peakedness = np.zeros((len(high_peaks),))
            for i, p in enumerate(high_peaks):
                nbrs = np.r_[scores[max(min_size-1, p-peakedness_radius):p], scores[p+1:min(len(scores), p+1+peakedness_radius)]]
                assert len(nbrs) > 0
                high_peaks_peakedness[i] = scores[p]-np.mean(nbrs)

    else:
        high_peaks_sorted = np.array([np.argmax(scores)], np.int)
        high_peaks_peakedness = np.inf

    return high_peaks_sorted, high_peaks_peakedness


# def find_z_section_map(stack, volume_zmin, downsample_factor = 16):
#
#     section_thickness = 20 # in um
#     xy_pixel_distance_lossless = 0.46
#     xy_pixel_distance_tb = xy_pixel_distance_lossless * 32 # in um, thumbnail
#     # factor = section_thickness/xy_pixel_distance_lossless
#
#     xy_pixel_distance_downsampled = xy_pixel_distance_lossless * downsample_factor
#     z_xy_ratio_downsampled = section_thickness / xy_pixel_distance_downsampled
#
#     # build annotation volume
#     section_bs_begin, section_bs_end = section_range_lookup[stack]
#     print section_bs_begin, section_bs_end
#
#     map_z_to_section = {}
#     for s in range(section_bs_begin, section_bs_end+1):
#         for z in range(int(z_xy_ratio_downsampled*s) - volume_zmin,
#                        int(z_xy_ratio_downsampled*(s+1)) - volume_zmin + 1):
#             map_z_to_section[z] = s
#
#     return map_z_to_section

def fit_ellipse_to_points(pts):

    pts = np.array(list(pts) if isinstance(pts, set) else pts)

    c0 = pts.mean(axis=0)

    coords0 = pts - c0

    U,S,V = np.linalg.svd(np.dot(coords0.T, coords0)/coords0.shape[0])
    v1 = U[:,0]
    v2 = U[:,1]
    s1 = np.sqrt(S[0])
    s2 = np.sqrt(S[1])

    return v1, v2, s1, s2, c0


def scores_to_vote(scores):
    vals = np.unique(scores)
    d = dict(zip(vals, np.linspace(0, 1, len(vals))))
    votes = np.array([d[s] for s in scores])
    votes = votes/votes.sum()
    return votes


def display_image(vis, filename='tmp.jpg'):

    if vis.dtype != np.uint8:
        imsave(filename, img_as_ubyte(vis))
    else:
        imsave(filename, vis)

    from IPython.display import FileLink
    return FileLink(filename)


def pad_patches_to_same_size(vizs, pad_value=0, keep_center=False, common_shape=None):
    """
    If patch size is larger than common shape, crop to common shape.
    """

    # If common_shape is not given, use the largest of all data
    if common_shape is None:
        common_shape = np.max([p.shape[:2] for p in vizs], axis=0)

    dt = vizs[0].dtype
    ndim = vizs[0].ndim

    if ndim == 2:
        common_box = (pad_value*np.ones((common_shape[0], common_shape[1]))).astype(dt)
    elif ndim == 3:
        common_box = (pad_value*np.ones((common_shape[0], common_shape[1], 3))).astype(dt)

    patches_padded = []
    for p in vizs:
        patch_padded = common_box.copy()

        if keep_center:

            top_margin = (common_shape[0] - p.shape[0])/2
            if top_margin < 0:
                ymin = 0
                ymax = common_shape[0]-1
                ymin2 = -top_margin
                ymax2 = -top_margin+common_shape[0]-1
            else:
                ymin = top_margin
                ymax = top_margin + p.shape[0] - 1
                ymin2 = 0
                ymax2 = p.shape[0]-1

            left_margin = (common_shape[1] - p.shape[1])/2
            if left_margin < 0:
                xmin = 0
                xmax = common_shape[1]-1
                xmin2 = -left_margin
                xmax2 = -left_margin+common_shape[1]-1
            else:
                xmin = left_margin
                xmax = left_margin + p.shape[1] - 1
                xmin2 = 0
                xmax2 = p.shape[1]-1

            patch_padded[ymin:ymax+1, xmin:xmax+1] = p[ymin2:ymax2+1, xmin2:xmax2+1]
#             patch_padded[top_margin:top_margin+p.shape[0], left_margin:left_margin+p.shape[1]] = p
        else:
            # assert p.shape[0] < common_shape[0] and p.shape[1] < common_shape[1]
            patch_padded[:p.shape[0], :p.shape[1]] = p

        patches_padded.append(patch_padded)

    return patches_padded

# def pad_patches_to_same_size(vizs, pad_value=0, keep_center=False, common_shape=None):
#
#     # If common_shape is not given, use the largest of all data
#     if common_shape is None:
#         common_shape = np.max([p.shape[:2] for p in vizs], axis=0)
#
#     dt = vizs[0].dtype
#     ndim = vizs[0].ndim
#
#     if ndim == 2:
#         common_box = (pad_value*np.ones((common_shape[0], common_shape[1]))).astype(dt)
#     elif ndim == 3:
#         common_box = (pad_value*np.ones((common_shape[0], common_shape[1], 3))).astype(dt)
#
#     patches_padded = []
#     for p in vizs:
#         patch_padded = common_box.copy()
#
#         if keep_center:
#             top_margin = (common_shape[0] - p.shape[0])/2
#             left_margin = (common_shape[1] - p.shape[1])/2
#             patch_padded[top_margin:top_margin+p.shape[0], left_margin:left_margin+p.shape[1]] = p
#         else:
#             patch_padded[:p.shape[0], :p.shape[1]] = p
#         patches_padded.append(patch_padded)
#
#     return patches_padded

def display_volume_sections(vol, every=5, ncols=5, **kwargs):
    zmin, zmax = bbox_3d(vol)[4:]
    zs = range(zmin+1, zmax, every)
    vizs = [vol[..., z] for z in zs]
    titles = ['z=%d' % z  for z in zs]
    display_images_in_grids(vizs, nc=ncols, titles=titles, **kwargs)

def display_images_in_grids(vizs, nc, titles=None, export_fn=None, maintain_shape=True, **kwargs):

    if maintain_shape:

        vizs = pad_patches_to_same_size(vizs)

    n = len(vizs)
    nr = int(np.ceil(n/float(nc)))
    aspect_ratio = vizs[0].shape[1]/float(vizs[0].shape[0]) # width / height

    fig, axes = plt.subplots(nr, nc, figsize=(nc*5*aspect_ratio, nr*5))
    axes = axes.flatten()

    for i in range(len(axes)):
        if i >= n:
            axes[i].axis('off');
        else:
            axes[i].imshow(vizs[i], **kwargs);
            if titles is not None:
                axes[i].set_title(titles[i], fontsize=30);
            axes[i].set_xticks([]);
            axes[i].set_yticks([]);

    fig.tight_layout();

    if export_fn is not None:
        create_if_not_exists(os.path.dirname(export_fn))
        plt.savefig(export_fn);
        plt.close(fig)
    else:
        plt.show();

# <codecell>

# import numpy as np
# from scipy.ndimage.filters import maximum_filter
# from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

# def detect_peaks(image):
#     """
#     Takes an image and detect the peaks usingthe local maximum filter.
#     Returns a boolean mask of the peaks (i.e. 1 when
#     the pixel's value is the neighborhood maximum, 0 otherwise)
#     """

#     # define an 8-connected neighborhood
#     neighborhood = generate_binary_structure(2,2)

#     #apply the local maximum filter; all pixel of maximal value
#     #in their neighborhood are set to 1
#     local_max = maximum_filter(image, footprint=neighborhood)==image
#     #local_max is a mask that contains the peaks we are
#     #looking for, but also the background.
#     #In order to isolate the peaks we must remove the background from the mask.

#     #we create the mask of the background
#     background = (image==0)

#     #a little technicality: we must erode the background in order to
#     #successfully subtract it form local_max, otherwise a line will
#     #appear along the background border (artifact of the local maximum filter)
#     eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

#     #we obtain the final mask, containing only peaks,
#     #by removing the background from the local_max mask
#     detected_peaks = local_max - eroded_background

#     return detected_peaks

# <codecell>

# def visualize_cluster(scores, cluster='all', title='', filename=None):
#     '''
#     Generate black and white image with the cluster of superpixels highlighted
#     '''

#     vis = scores[segmentation]
#     if cluster != 'all':
#         cluster_selection = np.equal.outer(segmentation, cluster).any(axis=2)
#         vis[~cluster_selection] = 0

#     plt.matshow(vis, cmap=plt.cm.Greys_r);
#     plt.axis('off');
#     plt.title(title)
#     if filename is not None:
#         plt.savefig(os.path.join(result_dir, 'stages', filename + '.png'), bbox_inches='tight')
# #     plt.show()
#     plt.close();


def paint_superpixels_on_image(superpixels, segmentation, img):
    '''
    Highlight a cluster of superpixels on the real image
    '''

    cluster_map = -1*np.ones_like(segmentation)
    for s in superpixels:
        cluster_map[segmentation==s] = 1
    vis = label2rgb(cluster_map, image=img)
    return vis

def paint_superpixel_groups_on_image(sp_groups, segmentation, img, colors):
    '''
    Highlight multiple superpixel groups with different colors on the real image
    '''

    cluster_map = -1*np.ones_like(segmentation)
    for i, sp_group in enumerate(sp_groups):
        for j in sp_group:
            cluster_map[segmentation==j] = i
    vis = label2rgb(cluster_map, image=img, colors=colors)
    return vis

# <codecell>

def kl(a,b):
    m = (a!=0) & (b!=0)
    return np.sum(a[m]*np.log(a[m]/b[m]))

def js(u,v):
    m = .5 * (u + v)
    r = .5 * (kl(u,m) + kl(v,m))
    return r

# <codecell>

def chi2(u,v):
    """
    Compute Chi^2 distance between two distributions.

    Empty bins are ignored.

    """

    u[u==0] = 1e-6
    v[v==0] = 1e-6
    r = np.sum(((u-v)**2).astype(np.float)/(u+v))

    # m = (u != 0) & (v != 0)
    # r = np.sum(((u[m]-v[m])**2).astype(np.float)/(u[m]+v[m]))

    # r = np.nansum(((u-v)**2).astype(np.float)/(u+v))
    return r


def chi2s(h1s, h2s):
    """
    h1s is n x n_texton
    MUST be float type
    """
    return np.sum((h1s-h2s)**2/(h1s+h2s+1e-10), axis=1)

# def chi2s(h1s, h2s):
#     '''
#     h1s is n x n_texton
#     '''
#     s = (h1s+h2s).astype(np.float)
#     with np.errstate(divide='ignore', invalid='ignore'):
#         ss = (h1s-h2s)**2/s
#     ss[s==0] = 0
#     return np.sum(ss, axis=1)


def alpha_blending(src_rgb, dst_rgb, src_alpha, dst_alpha):


    if src_rgb.dtype == np.uint8:
        src_rgb = img_as_float(src_rgb)

    if dst_rgb.dtype == np.uint8:
        dst_rgb = img_as_float(dst_rgb)

    if isinstance(src_alpha, float) or  isinstance(src_alpha, int):
        src_alpha = src_alpha * np.ones((src_rgb.shape[0], src_rgb.shape[1]))

    if isinstance(dst_alpha, float) or  isinstance(dst_alpha, int):
        dst_alpha = dst_alpha * np.ones((dst_rgb.shape[0], dst_rgb.shape[1]))

    out_alpha = src_alpha + dst_alpha * (1. - src_alpha)
    out_rgb = (src_rgb * src_alpha[..., None] +
               dst_rgb * dst_alpha[..., None] * (1. - src_alpha[..., None])) / out_alpha[..., None]

    out = np.zeros((src_rgb.shape[0], src_rgb.shape[1], 4))

    out[..., :3] = out_rgb
    out[..., 3] = out_alpha

    return out


def bbox_2d(img):
    """
    Returns:
        (xmin, xmax, ymin, ymax)
    """

    if np.count_nonzero(img) == 0:
        raise Exception('bbox2d: Image is empty.')
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return cmin, cmax, rmin, rmax

def bbox_3d(img):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    try:
        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]
    except:
        raise Exception('Input is empty.\n')

    return cmin, cmax, rmin, rmax, zmin, zmax



# def sample(points, num_samples):
#     n = len(points)
#     sampled_indices = np.random.choice(range(n), min(num_samples, n), replace=False)
#     return points[sampled_indices]

def apply_function_to_nested_list(func, l):
    """
    Func applies to the list consisting of all elements of l, and return a list.
    l: a list of list
    """
    from itertools import chain
    result = func(list(chain(*l)))
    csum = np.cumsum(map(len, l))
    new_l = [result[(0 if i == 0 else csum[i-1]):csum[i]] for i in range(len(l))]
    return new_l


def apply_function_to_dict(func, d):
    """
    Args:
        func:
            a function that takes as input the list consisting of a flatten list of values of `d`, and return a list.
        d (dict {key: list}): 
    """
    from itertools import chain
    result = func(list(chain(*d.values())))
    csum = np.cumsum(map(len, d.values()))
    new_d = {k: result[(0 if i == 0 else csum[i-1]):csum[i]] for i, k in enumerate(d.keys())}
    return new_d

def smart_map(data, keyfunc, func):
    """
    Args:
        data (list): data
        keyfunc (func): a lambda function for data key.
        func (func): a function that takes f(key, group). `group` is a sublist of `data`.
                    func does a global operation using key.
                    then does a same operation on each entry of group.
                    return a list.
    """

    from itertools import groupby
    from multiprocess import Pool

    keyfunc_with_enum = lambda (i, x): keyfunc(x)

    grouping = groupby(sorted(enumerate(data), key=keyfunc_with_enum), keyfunc_with_enum)
    grouping_noidx = {}
    grouping_idx = {}
    for k, group in grouping:
        grouping_idx[k], grouping_noidx[k] = zip(*group)

    pool = Pool(15)
    results_by_key = pool.map(lambda (k, group): func(k, group), grouping_noidx.iteritems())
    pool.close()
    pool.join()

    # results_by_key = [func(k, group) for k, group in grouping_noidx.iteritems()]

    keys = grouping_noidx.keys()
    results_inOrigOrder = {i: res for k, results in zip(keys, results_by_key)
                           for i, res in zip(grouping_idx[k], results)}

    return results_inOrigOrder.values()

boynton_colors = dict(blue=(0,0,255),
    red=(255,0,0),
    green=(0,255,0),
    yellow=(255,255,0),
    magenta=(255,0,255),
    pink=(255,128,128),
    gray=(128,128,128),
    brown=(128,0,0),
    orange=(255,128,0))

kelly_colors = dict(vivid_yellow=(255, 179, 0),
                    strong_purple=(128, 62, 117),
                    vivid_orange=(255, 104, 0),
                    very_light_blue=(166, 189, 215),
                    vivid_red=(193, 0, 32),
                    grayish_yellow=(206, 162, 98),
                    medium_gray=(129, 112, 102),

                    # these aren't good for people with defective color vision:
                    vivid_green=(0, 125, 52),
                    strong_purplish_pink=(246, 118, 142),
                    strong_blue=(0, 83, 138),
                    strong_yellowish_pink=(255, 122, 92),
                    strong_violet=(83, 55, 122),
                    vivid_orange_yellow=(255, 142, 0),
                    strong_purplish_red=(179, 40, 81),
                    vivid_greenish_yellow=(244, 200, 0),
                    strong_reddish_brown=(127, 24, 13),
                    vivid_yellowish_green=(147, 170, 0),
                    deep_yellowish_brown=(89, 51, 21),
                    vivid_reddish_orange=(241, 58, 19),
                    dark_olive_green=(35, 44, 22))

high_contrast_colors = boynton_colors.values() + kelly_colors.values()

import randomcolor

def random_colors(count):
    rand_color = randomcolor.RandomColor()
    random_colors = [map(int, rgb_str[4:-1].replace(',', ' ').split())
                     for rgb_str in rand_color.generate(luminosity="bright", count=count, format_='rgb')]
    return random_colors

def read_dict_from_txt(fn, converter=np.float, key_converter=np.int):
    d = {}
    with open(fn, 'r') as f:
        for line in f.readlines():
            items = line.split()
            if len(items) == 2:
                d[key_converter(items[0])] = converter(items[1])
            else:
                d[key_converter(items[0])] = np.array(map(converter, items[1:]))
    return d

def write_dict_to_txt(d, fn, fmt='%f'):
    with open(fn, 'w') as f:
        for k, vals in d.iteritems():
            f.write(k + ' ' +  (' '.join([fmt]*len(vals))) % tuple(vals) + '\n')
