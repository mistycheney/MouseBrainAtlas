import numpy as np
from multiprocess import Pool

selected_cell_arrays = None
selected_cell_arrays_h = None
selected_cell_arrays_v = None
selected_cell_arrays_d = None
selected_cell_sizes = None

def set_selected_cell_arrays(a, h, v, d, s):
    global selected_cell_arrays
    global selected_cell_arrays_h
    global selected_cell_arrays_v
    global selected_cell_arrays_d
    global selected_cell_sizes
    selected_cell_arrays = a
    selected_cell_arrays_h = h
    selected_cell_arrays_v = v
    selected_cell_arrays_d = d
    selected_cell_sizes = s


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

def compute_jaccard_x_vs_list(t, x, t_sizes, x_size):
    """
    t: n x d
    x: 1 x d - boolean array
    """

    intersections_with_i = t[:, x].sum(axis=1)
    unions_with_i = t_sizes + x_size - intersections_with_i
    return intersections_with_i.astype(np.float)/unions_with_i


def compute_jaccard_i_vs_list(i, indices):

    global selected_cell_arrays
    global selected_cell_arrays_h
    global selected_cell_arrays_v
    global selected_cell_arrays_d
    global selected_cell_sizes

    if indices == 'all':
        intersections_with_i = selected_cell_arrays[:, selected_cell_arrays[i]].sum(axis=1)
        intersections_with_i_h = selected_cell_arrays_h[:, selected_cell_arrays_h[i]].sum(axis=1)
        intersections_with_i_v = selected_cell_arrays_v[:, selected_cell_arrays_v[i]].sum(axis=1)
        intersections_with_i_d = selected_cell_arrays_d[:, selected_cell_arrays_d[i]].sum(axis=1)

        unions_with_i = selected_cell_sizes[i] + selected_cell_sizes - intersections_with_i

    else:
        intersections_with_i = selected_cell_arrays[indices, selected_cell_arrays[i]].sum(axis=1)
        intersections_with_i_h = selected_cell_arrays_h[indices, selected_cell_arrays_h[i]].sum(axis=1)
        intersections_with_i_v = selected_cell_arrays_v[indices, selected_cell_arrays_v[i]].sum(axis=1)
        intersections_with_i_d = selected_cell_arrays_d[indices, selected_cell_arrays_d[i]].sum(axis=1)

        unions_with_i = selected_cell_sizes[i] + selected_cell_sizes[indices] - intersections_with_i

    intersections_all_poses = np.c_[intersections_with_i, intersections_with_i_h, intersections_with_i_v, intersections_with_i_d] # nx4
    max_poses = np.argmax(intersections_all_poses, axis=1)
    max_intersection = intersections_all_poses[range(len(intersections_with_i)), max_poses]
    return max_intersection.astype(np.float)/unions_with_i, max_poses


def compute_jaccard_i_vs_all(i, return_poses=False):
    scores, poses = compute_jaccard_i_vs_list(i, 'all')

    if return_poses:
        return scores, poses
    else:
        return scores

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


def compute_jaccard_list_vs_all(seed_indices, return_poses=False):

    pool = Pool(14)

    if return_poses:
        scores_poses_tuples = pool.map(lambda i: compute_jaccard_i_vs_all(i, return_poses=True), seed_indices)
        affinities_to_seeds = np.array([scores for scores, poses in scores_poses_tuples])
    else:
        affinities_to_seeds = np.array(pool.map(lambda i: compute_jaccard_i_vs_all(i), seed_indices))

    pool.close()
    pool.join()

    if return_poses:
        poses = np.array([poses for scores, poses in scores_poses_tuples])
        return affinities_to_seeds, poses
    else:
        return affinities_to_seeds
