from __future__ import division

import itertools

import numpy as np
import scipy.stats
from scipy import ndimage

from skimage.segmentation import find_boundaries
from skimage.morphology import skeletonize

from _shape_context import _pixel_graph, chi2_distance


def pixel_graph(img):
    assert len(np.unique(img)) <= 2
    return _pixel_graph(img.astype(np.uint8))


# TODO: move asserts to tests

def _sample_single_contour(img, n_points):
    """Samples pixels, in order, along a contour.

    Parameters
    ----------
    img : np.ndarray
        A binary image with nonzero elements along a connected
        contour.
    n_points : the number of points to return

    Returns
    -------
    points : list of tuples
        An ordered list of (x, y) points, starting from the point
        closest to the origin.

    """
    # Right now, just does a depth-first search. This is not optimal
    # because backtracking can put some pixels very far out of
    # order. a better approach would be to find a locally stable
    # sorting. However, this seems to work well enough for now.
    graph = pixel_graph(img)
    if len(graph) == 0:
        return []
    visited = set()
    unvisited = set(graph.keys())
    stacked = set()
    stack = []
    order = []

    # start from pixel closest to origin
    x, y = np.nonzero(img)
    start = (x[0], y[0])
    stack.append(start)
    
    while unvisited:
        # assert len(visited) + len(unvisited) == len(graph)
        try:
            node = stack.pop()
        except:
            node = unvisited.pop()
        # assert not node in visited
        order.append(node)
        visited.add(node)
        unvisited.remove(node)
        neighbors = graph[node]
        for n in neighbors - stacked - visited:
            stack.append(n)
            stacked.add(n)
        # assert len(visited) + len(unvisited) == len(graph)
        # assert len(visited & unvisited) == 0
    # assert len(order) == len(graph)
    stride = int(np.ceil(len(order) / n_points))
    return order[::stride]


def sample_points(img, n_points=100):
    """Sample points along edges in a binary image.

    Returns an array of shape ``(n_points, 2)`` in image coordinates.

    If there are several disconnected contours, they are sampled
    seperately and appended in order of their minimum distance to the
    origin of ``img`` in NumPy array coordinates.

    """
    # FIXME: what if contour crosses itself? for example: an infinity
    # symbol?
    assert img.ndim == 2
    assert n_points > 0

    boundaries = skeletonize(find_boundaries(img))

    # reorder along curves; account for holes and disconnected lines
    # with connected components.
    labels, n_labels = ndimage.label(boundaries, structure=np.ones((3, 3)))
    n_labeled_pixels = labels.sum()
    all_labels = range(1, n_labels + 1)
    curve_n_pixels = list((labels == lab).sum() for lab in all_labels)
    curve_n_points = list(int(np.ceil((n / n_labeled_pixels) * n_points))
                          for n in curve_n_pixels)

    # sample a linear subset of each connected curve
    samples = list(_sample_single_contour(labels == lab, n_points)
                   for lab, n_points in zip(all_labels, curve_n_points))

    # append them together. They should be in order, because
    # ndimage.label() labels in order.
    points = list(itertools.chain(*samples))
    return np.vstack(points)


def euclidean_dists_angles(points):
    """Returns symmetric pairwise ``dists`` and ``angles`` arrays."""
    # TODO: rotation invariance; compute angles relative to tangent
    n = len(points)
    dists = scipy.spatial.distance.pdist(points, 'euclidean')
    dists = scipy.spatial.distance.squareform(dists)
    # TODO: repeated effort to compute angles this way
    rows = np.hstack([points] * n).reshape(n, n, 2)
    cols = np.vstack([points] * n).reshape(n, n, 2)
    pairs = rows - cols
    x = pairs[:, :, 0]
    y = pairs[:, :, 1]
    angles = np.arctan2(y, x)
    return dists, angles


def shape_context(dists, angles, n_radial_bins=5, n_polar_bins=12):
    """Compute shape context descriptors for all given points.

    If ``dists`` and ``angles`` are Euclidean, this corresponds to the
    original shape context (Belongie, Malik, and Puzicha, 2002). If
    they are the inner-distance and inner angle, this corresponds to
    the inner-distance shape context (Ling and Jacobs, 2007).

    Parameters
    ----------
    dists : (N, N) ndarray
        ``dists[i, j]`` is the distance between points ``i`` and ``j``.

    angles : (N, N) ndarray
        ``angles[i, j]`` is the distance between points ``i`` and ``j``.

    n_radial_bins : int
        number of radial bins in histogram

    n_polar_bins : int
        number of polar bins in histogram

    Returns
    -------
    shape_contexts : ndarray
        The shape context descriptor for each point. Has shape
        ``(n_points, radial_bins, polar_bins)``.

    """
    assert dists.shape[0] == dists.shape[1]
    assert dists.ndim == 2
    assert dists.shape == angles.shape

    # ensure distances are symmetric
    # assert (dists.T == dists).all()

    n_points = dists.shape[0]

    r_array = np.logspace(0, 1, n_radial_bins, base=10) / 10.0
    r_array = np.hstack(([0], r_array))[1:]
    theta_array = np.linspace(-np.pi, np.pi, n_polar_bins + 1)[1:]
    result = np.zeros((n_points, n_radial_bins, n_polar_bins),
                      dtype=np.int)

    # normalize distances
    dists = dists / dists.max()

    for i in range(n_points):
        for j in range(i + 1, n_points):
            r_idx = np.searchsorted(r_array, dists[i, j])
            theta_idx = np.searchsorted(theta_array, angles[i, j])
            result[i, r_idx, theta_idx] += 1

            theta_idx = np.searchsorted(theta_array, angles[j, i])
            result[j, r_idx, theta_idx] += 1

    # ensure all points were counted
    # assert (result.reshape(n_points, -1).sum(axis=1) == (n_points - 1)).all()
    return result


def shape_distance(a_descriptors, b_descriptors, penalty=0.3, backtrace=False):
    """Computes the distance between two shapes.

    The distance is defined as the minimal cost of aligning an ordered
    sequence of shape context descriptors along their contours. For
    more information, see Ling and Jacobs, 2007.

    Uses dynamic programming to find best alignment of sampled points.

    If ``backtrace`` is True, also returns alignment.

    """
    # FIXME: Assumes the sequences' starting and ending points are aligned.
    # TODO: this could probably be optimized.
    # TODO: write a visualization of the alignment found in this function.

    assert a_descriptors.ndim == 3
    assert b_descriptors.ndim == 3
    assert a_descriptors.shape[1:] == b_descriptors.shape[1:]

    n_rows = a_descriptors.shape[0]
    n_cols = b_descriptors.shape[0]

    a_descriptors = a_descriptors.reshape(n_rows, -1)
    b_descriptors = b_descriptors.reshape(n_cols, -1)

    table = np.zeros((n_rows, n_cols))

    # TODO: perhaps precomputing all pairwise distances would be
    # faster
    d = lambda i, j: chi2_distance(a_descriptors[i],
                                   b_descriptors[j])

    # initialize outer elements
    table[0, 0] = d(0, 0)

    for i in range(1, n_rows):
        match = i * penalty + d(i, 0)
        mismatch = table[i - 1, 0] + penalty
        table[i, 0] = min(match, mismatch)

    for j in range(1, n_cols):
        match = j * penalty + d(0, j)
        mismatch = table[0, j - 1] + penalty
        #table[i, 0] = min(match, mismatch)
	table[0,j] = min(match, mismatch)

    # fill in the rest of the table
    for i in range(1, n_rows):
        for j in range(1, n_cols):
            match = table[i - 1, j - 1] + d(i, j)
            mismatch = min(table[i - 1, j],
                           table[i, j - 1]) + penalty
            table[i, j] = min(match, mismatch)

    # tracing optimal alignment is not necessary. we are just
    # interested in the final cost.
    if not backtrace:
        return table[-1, -1]

    i = n_rows - 1
    j = n_cols - 1

    alignment = []
    while i > 0 or j > 0:
        if i == 0 or j == 0:
            break
        
        val = table[i - 1, j - 1]
        up = table[i - 1, j]
        left = table[i, j - 1]

        if val <= left and val <= up:
            alignment.append((i, j))
            i = i - 1
            j = j - 1
        elif left < up:
            j -= 1
        else:
            i -= 1
    return table[-1, -1], alignment[::-1]


def full_shape_distance(img1, img2, n_points=100):
    """A convenience function to compute the distance between two binary images."""
    points1 = sample_points(img1, n_points=n_points)
    dists1, angles1 = euclidean_dists_angles(points1)
    descriptors1 = shape_context(dists1, angles1)

    points2 = sample_points(img2, n_points=n_points)
    dists2, angles2 = euclidean_dists_angles(points2)
    descriptors2 = shape_context(dists2, angles2)

    d = shape_distance(descriptors1, descriptors2)
    return d


def dists_to_affinities(dists, neighbors=10, alpha=0.27):
    """Compute an affinity matrix for a given distance matrix."""
    affinities = np.zeros_like(dists)
    sorted_rows = np.sort(dists, axis=1)
    for i in range(dists.shape[0]):
        for j in range(i, dists.shape[1]):
            nn_dists = np.hstack((sorted_rows[i, 1:neighbors],
                                  sorted_rows[j, 1:neighbors]))
            sigma = alpha * nn_dists.mean()
            sim = scipy.stats.norm.pdf(dists[i, j], loc=0, scale=alpha * sigma)
            affinities[i, j] = sim
            affinities[j, i] = sim

    # normalize each row
    return affinities / affinities.sum(axis=1)


def graph_transduction(i, affinities, max_iters=5000):
    """Compute new affinities for a query based on graph transduction.

    The ``i``th element of ``affinities`` is the query; the rest are its
    candidate matches.

    Method as described in "Learning context-sensitive shape
    similarity by graph transduction." by Bai, Yang, et. al. (2010).

    """
    f = np.zeros((affinities.shape[0], 1))
    f[i] = 1
    for _ in range(max_iters):
        f = np.dot(affinities, f)
        f[i] = 1
    return f.ravel()


# TODO: for each shape, retrieve its nearest neighbors and only do
# graph transduction on them.

def compute_new_affinities(affinities):
    """Computes all new pairwise affinities by graph transduction."""
    result = list(graph_transduction(i, affinities) for i in range(affinities.shape[0]))
    return np.vstack(affinities)
