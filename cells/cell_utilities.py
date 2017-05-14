import numpy as np
from multiprocess import Pool
import sys
import os
import pandas
import bloscpack as bp
# from matplotlib.path import Path

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from data_manager import *

###############

n_radial = 4
# radial_bins = np.logspace(0, 2, 10, base=10)
radial_bins = np.linspace(0, 100, n_radial+1)

n_angular = 8
angular_bins = np.linspace(-np.pi, np.pi, n_angular+1)

n_orientation_bins = 10
orientation_bins = np.linspace(-np.pi/2, np.pi/2, n_orientation_bins+1)

size_bins = np.r_[np.linspace(0, 3000, 10), np.inf]

n_edge_length_bins = 10
edge_length_bins = np.r_[np.linspace(0, 100, n_edge_length_bins), np.inf]

n_edge_direction_bins = 10
edge_direction_bins = np.linspace(-np.pi/2, np.pi/2, n_edge_direction_bins+1)

###############

def visualize_blob_contour(binary_img, rgb_img):
    viz = rgb_img.copy()
    for cnt in find_contour_points(binary_img)[1]:
        cv2.polylines(viz, [cnt.astype(np.int)], isClosed=True, color=(255,0,0), thickness=2)
    return viz

tree = None # important, otherwise multiprocess.Pool does not work

def normalize_angle(a):
    """
    Force angle into -pi/2 to pi/2.
    """
    if a > np.pi/2:
        a = a - np.pi
    elif a < -np.pi/2:
        a = a + np.pi
    return a

def load_data(fp):

    if fp.endswith('bp'):
        data = bp.unpack_ndarray_file(fp)
    elif fp.endswith('jpg'):
        data = imread(fp)
    elif fp.endswith('hdf'):
        data = load_hdf_v2(fp).tolist()
    elif fp.endswith('pkl'):
        data = pickle.load(open(fp, 'r'))
    else:
        raise Exception('Not recognized.')

    return data

def load_cell_classifier_data(what, stack, sec=None, fn=None, anchor_fn=None, ext=None):
    fp = get_cell_classifier_data_filepath(what, stack, sec, fn, anchor_fn, ext)
    download_from_s3(fp)
    return load_data(fp)

def get_cell_classifier_data_filepath(what, stack, sec=None, fn=None, anchor_fn=None, ext=None):

    if fn is None:
        assert sec is not None
        fn = metadata_cache['sections_to_filenames'][stack][sec]
        if is_invalid(fn):
            sys.stderr.write('This section is invalid.\n')
            return

    if anchor_fn is None:
        anchor_fn = metadata_cache['anchor_fn'][stack]

    features_dir = create_if_not_exists(os.path.join(CELL_FEATURES_CLF_ROOTDIR, 'features_per_section'))

    fn_template = '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(what)s.%(ext)s'
    fp = os.path.join(features_dir, stack, fn, fn_template % \
                    {'fn':fn, 'anchor_fn':anchor_fn, 'what':what, 'ext':ext})

    return fp

def get_cell_data_filepath(what, stack, sec=None, fn=None, ext=None):

    if fn is None:
        assert sec is not None
        fn = metadata_cache['sections_to_filenames'][stack][sec]
        if fn in ['Placeholder', 'Nonexisting', 'Rescan']:
            sys.stderr.write('This section is invalid.\n')
            return

    if what == 'orientation':
        fn_template = '%(fn)s_blobOrientations.bp'
    elif what == 'major':
        fn_template = '%(fn)s_blobMajorAxisLen.bp'
    elif what == 'minor':
        fn_template = '%(fn)s_blobMinorAxisLen.bp'
    elif what == 'mask_center':
        fn_template = '%(fn)s_blobMaskCenters.bp'
    elif what == 'mask':
        fn_template = '%(fn)s_blobMasks.hdf'
    elif what == 'centroid':
        fn_template = '%(fn)s_blobCentroids.bp'
    elif what == 'contours':
        fn_template = '%(fn)s_blobContours.hdf'
    elif what == 'image':
        fn_template = '%(fn)s_image.jpg'
    elif what == 'cells_aligned':
        fn_template = '%(fn)s_cells_aligned.hdf'
    elif what == 'cells_aligned_h':
        fn_template = '%(fn)s_cells_aligned_h.hdf'
    elif what == 'cells_aligned_v':
        fn_template = '%(fn)s_cells_aligned_v.hdf'
    elif what == 'cells_aligned_d':
        fn_template = '%(fn)s_cells_aligned_d.hdf'
    elif what == 'cells_aligned_padded':
        fn_template = '%(fn)s_cells_aligned_padded.bp'
    elif what == 'cells_aligned_padded_h':
        fn_template = '%(fn)s_cells_aligned_padded_h.bp'
    elif what == 'cells_aligned_padded_v':
        fn_template = '%(fn)s_cells_aligned_padded_v.bp'
    elif what == 'cells_aligned_padded_d':
        fn_template = '%(fn)s_cells_aligned_padded_d.bp'
    elif what == 'neighbors':
        fn_template = '%(fn)s_neighbors.hdf'
    elif what == 'neighbor_vectors':
        fn_template = '%(fn)s_neighbor_vectors.hdf'
    else:
        fn_template = '%(fn)s_' + what + '.' + ext
    # else:
    #     raise Exception('Not recognized.')

    fp = os.path.join(DETECTED_CELLS_ROOTDIR, stack, fn, fn_template % {'fn': fn})
    return fp

def load_cell_data(what, stack, sec=None, fn=None, ext=None):

    fp = get_cell_data_filepath(what, stack, sec=sec, fn=fn, ext=ext)
    if fp is None:
        raise 'Cannot load data for section %d.'
    download_from_s3(fp)

    data = load_data(fp)
    return data

def allocate_radial_angular_bins(vectors, anchor_direction, angular_bins, radial_bins):
    """
    Return bins starting from 0.
    """

    if isinstance(angular_bins, int):
        angular_bins = np.linspace(-np.pi, np.pi, angular_bins)

    distances = np.sqrt(np.sum(vectors**2, axis=1))
    radial_bin_indices = np.digitize(distances, radial_bins) - 1

    angles = np.arctan2(vectors[:, 1], vectors[:, 0]) # -pi, pi
    angles_relative_to_anchor = angles - anchor_direction

    angles_relative_to_anchor[angles_relative_to_anchor > np.pi] -= 2*np.pi
    angles_relative_to_anchor[angles_relative_to_anchor < -np.pi] += 2*np.pi

    angular_bin_indices = np.digitize(angles_relative_to_anchor, angular_bins) - 1

    return radial_bin_indices, angular_bin_indices

def allocate_bins_overlap(vs, bins):

    bins = np.array(bins)

    left_edges = bins[:,0]
    right_edges = bins[:,1]
    n_bins = len(bins)

    left = np.searchsorted(left_edges, vs)
    right = np.searchsorted(right_edges, vs)

    allo = [None for i in range(len(vs))]

    a = left - right == 1
    single = np.where(a)[0]
    dual = list(set(range(len(vs))) - set(single))

    for i in single:
        allo[i] = right[i]

    for i in dual:
        allo[i] = (right[i], left[i]-1)

    return allo

def allocate_radial_angular_bins_overlap(vectors, anchor_direction, angular_bins, radial_bins):
    """
    radial_bins: list of (left edge, right edge)
    """

    distances = np.sqrt(np.sum(vectors**2, axis=1))
    radial_bin_indices = allocate_bins_overlap(distances, radial_bins)

    angles = np.arctan2(vectors[:, 1], vectors[:, 0]) # -pi, pi
    angles_relative_to_anchor = angles - anchor_direction
    angular_bin_indices = allocate_bins_overlap(angles_relative_to_anchor, angular_bins)

    return radial_bin_indices, angular_bin_indices

selected_cell_arrays = None
selected_cell_sizes = None

def set_selected_cell_arrays(a, s):
    global selected_cell_arrays
    global selected_cell_sizes
    selected_cell_arrays = a
    selected_cell_sizes = s

# selected_cell_arrays = None
# selected_cell_arrays_h = None
# selected_cell_arrays_v = None
# selected_cell_arrays_d = None
# selected_cell_sizes = None
#
# def set_selected_cell_arrays(a, h, v, d, s):
#     global selected_cell_arrays
#     global selected_cell_arrays_h
#     global selected_cell_arrays_v
#     global selected_cell_arrays_d
#     global selected_cell_sizes
#     selected_cell_arrays = a
#     selected_cell_arrays_h = h
#     selected_cell_arrays_v = v
#     selected_cell_arrays_d = d
#     selected_cell_sizes = s
#

def parallel_cdist(data1, data2, n_rows_per_job=100):

    from scipy.spatial.distance import cdist

    data1 = np.array(data1)
    data2 = np.array(data2)

    pool = Pool(12)

    start_indices = np.arange(0, data1.shape[0], n_rows_per_job)
    end_indices = start_indices + n_rows_per_job - 1

    partial_distance_matrices = pool.map(lambda (si, ei): cdist(data1[si:ei+1].copy(), data2), zip(start_indices, end_indices))
    pool.close()
    pool.join()

    distance_matrix = np.concatenate(partial_distance_matrices)
    return distance_matrix

def kmeans(data, seed_indices, n_iter=100):

    n_classes = len(seed_indices)
    n_data = len(data)

    centroids = data[seed_indices]
    for i in range(n_iter):
        sys.stderr.write('iter %d\n' % i)
        point_centroid_distances = parallel_cdist(data, centroids)

        if i > 0:
            prev_nearest_centroid_indices = nearest_centroid_indices.copy()

        nearest_centroid_indices = np.argmin(point_centroid_distances, axis=1)

        if i > 0:
            n_change = np.count_nonzero(prev_nearest_centroid_indices != nearest_centroid_indices)
            change_ratio = float(n_change) / n_data

            if change_ratio < .01:
                break

        centroids = [data[nearest_centroid_indices == c].mean(axis=0) for c in range(n_classes)]

    return nearest_centroid_indices, np.asarray(centroids)

def compute_jaccard_x_vs_list_v2(x, t, x_size=None, t_sizes=None):
    """
    t: n x d
    x: 1 x d - boolean array
    """

    if t_sizes is None:
        t_sizes = np.sum(t, axis=1)

    if x_size is None:
        x_size = np.count_nonzero(x)

    intersections_with_i = t[:, x].sum(axis=1)
    unions_with_i = t_sizes + x_size - intersections_with_i
    return intersections_with_i.astype(np.float)/unions_with_i

# def compute_jaccard_x_vs_list(x, t, x_size=None, t_sizes=None, x_h=None, x_v=None, x_d=None):
#     """
#     t: n x d
#     x: 1 x d - boolean array
#     """
#
#     t = np.array(t)
#
#     x_size = np.count_nonzero(x)
#     t_sizes = np.sum(t, axis=1)
#     n = len(t)
#
#     intersections_with_x = t[:, x].sum(axis=1) # nx1
#     intersections_with_x_h = t[:, x_h].sum(axis=1)
#     intersections_with_x_v = t[:, x_v].sum(axis=1)
#     intersections_with_x_d = t[:, x_d].sum(axis=1)
#
#     intersections_all_mirrors = np.c_[intersections_with_x, intersections_with_x_h,
#                                     intersections_with_x_v, intersections_with_x_d] # nx4
#     temp = t_sizes + x_size # nx1
#
#     unions_with_x = temp - intersections_with_x
#     unions_with_x_h = temp - intersections_with_x_h
#     unions_with_x_v = temp - intersections_with_x_v
#     unions_with_x_d = temp - intersections_with_x_d
#
#     unions_all_mirrors = np.c_[unions_with_x, unions_with_x_h, unions_with_x_v, unions_with_x_d] # nx4
#
#     jaccards = intersections_all_mirrors.astype(np.float) / unions_all_mirrors # nx4
#     best_mirrors = np.argmax(jaccards, axis=1)
#     best_jaccards = jaccards[range(n), best_mirrors]
#
#     return best_jaccards, best_mirrors

# def compute_jaccard_i_vs_list(i, indices):
#
#     global selected_cell_arrays
#     global selected_cell_arrays_h
#     global selected_cell_arrays_v
#     global selected_cell_arrays_d
#     global selected_cell_sizes
#
#     if indices == 'all':
#         intersections_with_i = selected_cell_arrays[:, selected_cell_arrays[i]].sum(axis=1)
#         intersections_with_i_h = selected_cell_arrays_h[:, selected_cell_arrays_h[i]].sum(axis=1)
#         intersections_with_i_v = selected_cell_arrays_v[:, selected_cell_arrays_v[i]].sum(axis=1)
#         intersections_with_i_d = selected_cell_arrays_d[:, selected_cell_arrays_d[i]].sum(axis=1)
#
#         unions_with_i = selected_cell_sizes[i] + selected_cell_sizes - intersections_with_i
#         unions_with_i_h = selected_cell_sizes[i] + selected_cell_sizes - intersections_with_i_h
#         unions_with_i_v = selected_cell_sizes[i] + selected_cell_sizes - intersections_with_i_v
#         unions_with_i_d = selected_cell_sizes[i] + selected_cell_sizes - intersections_with_i_d
#
#     else:
#         intersections_with_i = selected_cell_arrays[indices, selected_cell_arrays[i]].sum(axis=1)
#         intersections_with_i_h = selected_cell_arrays_h[indices, selected_cell_arrays_h[i]].sum(axis=1)
#         intersections_with_i_v = selected_cell_arrays_v[indices, selected_cell_arrays_v[i]].sum(axis=1)
#         intersections_with_i_d = selected_cell_arrays_d[indices, selected_cell_arrays_d[i]].sum(axis=1)
#
#         unions_with_i = selected_cell_sizes[i] + selected_cell_sizes[indices] - intersections_with_i
#         unions_with_i_h = selected_cell_sizes[i] + selected_cell_sizes[indices] - intersections_with_i_h
#         unions_with_i_v = selected_cell_sizes[i] + selected_cell_sizes[indices] - intersections_with_i_v
#         unions_with_i_d = selected_cell_sizes[i] + selected_cell_sizes[indices] - intersections_with_i_d
#
#     intersections_all_poses = np.c_[intersections_with_i, intersections_with_i_h, intersections_with_i_v, intersections_with_i_d] # nx4
#     unions_all_poses = np.c_[unions_with_i, unions_with_i_h, unions_with_i_v, unions_with_i_d] # nx4
#     jaccards_all_poses = intersections_all_poses.astype(np.float)/unions_all_poses
#
#     n = len(jaccards_all_poses)
#
#     best_mirrors = np.argmax(jaccards_all_poses, axis=1)
#     best_jaccards = jaccards_all_poses[range(n), best_mirrors]
#
#     return best_jaccards, best_mirrors


def compute_jaccard_i_vs_list(i, indices):

    global selected_cell_arrays
    global selected_cell_sizes

    if indices == 'all':
        intersections_with_i = selected_cell_arrays[:, selected_cell_arrays[i]].sum(axis=1)
        unions_with_i = selected_cell_sizes[i] + selected_cell_sizes - intersections_with_i
    else:
        intersections_with_i = selected_cell_arrays[indices, selected_cell_arrays[i]].sum(axis=1)
        unions_with_i = selected_cell_sizes[i] + selected_cell_sizes[indices] - intersections_with_i

    jaccards = intersections_with_i.astype(np.float)/unions_with_i
    return jaccards

def compute_jaccard_i_vs_all(i):
    scores = compute_jaccard_i_vs_list(i, 'all')
    return scores

# def compute_jaccard_i_vs_all(i, return_poses=False):
#     scores, poses = compute_jaccard_i_vs_list(i, 'all')
#
#     if return_poses:
#         return scores, poses
#     else:
#         return scores

def compute_jaccard_pairwise(indices, square_form=True, parallel=True, return_poses=False):
    n = len(indices)

    if parallel:
        pool = Pool(16)
        scores_poses_tuples = pool.map(lambda x: compute_jaccard_i_vs_list(x[0],x[1]),
                                   [(indices[i], indices[i+1:]) for i in range(n)])
        pool.close()
        pool.join()
    else:
        scores_poses_tuples = [compute_jaccard_i_vs_list(indices[i], indices[i+1:]) for i in range(n)]

    pairwise_scores = np.array([scores for scores, poses in scores_poses_tuples])

    if square_form:
        pairwise_scores = squareform(np.concatenate(pairwise_scores))

    if return_poses:
        poses = np.array([poses for scores, poses in scores_poses_tuples])
        return pairwise_scores, poses
    else:
        return pairwise_scores

def compute_jaccard_list_vs_all(seed_indices):
    pool = Pool(14)
    affinities_to_seeds = np.array(pool.map(lambda i: compute_jaccard_i_vs_all(i), seed_indices))
    pool.close()
    pool.join()
    return affinities_to_seeds

# def compute_jaccard_list_vs_all(seed_indices, return_poses=False):
#
#     pool = Pool(14)
#
#     if return_poses:
#         scores_poses_tuples = pool.map(lambda i: compute_jaccard_i_vs_all(i, return_poses=True), seed_indices)
#         affinities_to_seeds = np.array([scores for scores, poses in scores_poses_tuples])
#     else:
#         affinities_to_seeds = np.array(pool.map(lambda i: compute_jaccard_i_vs_all(i), seed_indices))
#
#     pool.close()
#     pool.join()
#
#     if return_poses:
#         poses = np.array([poses for scores, poses in scores_poses_tuples])
#         return affinities_to_seeds, poses
#     else:
#         return affinities_to_seeds
