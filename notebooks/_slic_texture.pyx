#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
from libc.float cimport DBL_MAX
from cpython cimport bool

import numpy as np
cimport numpy as cnp

from skimage.util import regular_grid


def _slic_cython(double[:, :, :, ::1] image_zyx,
                 double[:, ::1] segments,
                 float step,
                 Py_ssize_t max_iter,
                 double[::1] spacing):

    # initialize on grid
    cdef Py_ssize_t depth, height, width
    depth = image_zyx.shape[0]
    height = image_zyx.shape[1]
    width = image_zyx.shape[2]

    cdef Py_ssize_t n_segments = segments.shape[0]
    # number of features [X, Y, Z, ...]
    cdef Py_ssize_t n_features = segments.shape[1]

    # approximate grid size for desired n_segments
    cdef Py_ssize_t step_z, step_y, step_x
    slices = regular_grid((depth, height, width), n_segments)
    step_z, step_y, step_x = [int(s.step) for s in slices]

    cdef Py_ssize_t[:, :, ::1] nearest_segments \
        = np.empty((depth, height, width), dtype=np.intp)
    cdef double[:, :, ::1] distance \
        = np.empty((depth, height, width), dtype=np.double)
    cdef Py_ssize_t[::1] n_segment_elems = np.zeros(n_segments, dtype=np.intp)

    cdef Py_ssize_t i, c, k, x, y, z, x_min, x_max, y_min, y_max, z_min, z_max
    cdef char change
    cdef double dist_center, cx, cy, cz, dy, dz

    cdef double sz, sy, sx
    sz = spacing[0]
    sy = spacing[1]
    sx = spacing[2]

    # The colors are scaled before being passed to _slic_cython so
    # max_color_sq can be initialised as all ones
    cdef double[::1] max_dist_color = np.ones(n_segments, dtype=np.double)
    cdef double dist_color

    # The reference implementation (Achanta et al.) calls this invxywt
    cdef double spatial_weight = float(1) / (step ** 2)

    for i in range(max_iter):
        change = 0
        distance[:, :, :] = DBL_MAX

        # iterate over all segments
        for k in range(n_segments):

            # segment coordinate centers
            cz = segments[k, 0]
            cy = segments[k, 1]
            cx = segments[k, 2]

            # calculate the window around segment center
            z_min = <Py_ssize_t>max(cz - 2 * step_z, 0)
            z_max = <Py_ssize_t>min(cz + 2 * step_z + 1, depth)
            y_min = <Py_ssize_t>max(cy - 2 * step_y, 0)
            y_max = <Py_ssize_t>min(cy + 2 * step_y + 1, height)
            x_min = <Py_ssize_t>max(cx - 2 * step_x, 0)
            x_max = <Py_ssize_t>min(cx + 2 * step_x + 1, width)

            for z in range(z_min, z_max):
                dz = (sz * (cz - z)) ** 2   # sq distance from segment center in z axis
                for y in range(y_min, y_max):
                    dy = (sy * (cy - y)) ** 2   # sq distance from segment center in y axis
                    for x in range(x_min, x_max):
                        dx = sx * (cx - x)) ** 2    # sq distance from segment center in x axis
                        dist_center = (dz + dy + dx) * spatial_weight   # sq distance from segment center
                        
                        dist_color = 0
                        for c in range(3, n_features):
                            dist_color += (image_zyx[z, y, x, c - 3]
                                            - segments[k, c]) ** 2
                        
                        dist_center += dist_color

                        if distance[z, y, x] > dist_center:
                            nearest_segments[z, y, x] = k
                            distance[z, y, x] = dist_center
                            change = 1

        # stop if no pixel changed its segment
        if change == 0:
            break

        # recompute segment centers

        # sum features for all segments
        n_segment_elems[:] = 0
        segments[:, :] = 0
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    k = nearest_segments[z, y, x]
                    n_segment_elems[k] += 1
                    segments[k, 0] += z
                    segments[k, 1] += y
                    segments[k, 2] += x
                    for c in range(3, n_features):
                        segments[k, c] += image_zyx[z, y, x, c - 3]

        # divide by number of elements per segment to obtain mean
        for k in range(n_segments):
            for c in range(n_features):
                segments[k, c] /= n_segment_elems[k]

    return np.asarray(nearest_segments)


def _enforce_label_connectivity_cython(Py_ssize_t[:, :, ::1] segments,
                                       Py_ssize_t n_segments,
                                       Py_ssize_t min_size,
                                       Py_ssize_t max_size):
    """ Helper function to remove small disconnected regions from the labels

    Parameters
    ----------
    segments : 3D array of int, shape (Z, Y, X)
        The label field/superpixels found by SLIC.
    n_segments: int
        Number of specified segments
    min_size: int
        Minimum size of the segment
    max_size: int
        Maximum size of the segment. This is done for performance reasons,
        to pre-allocate a sufficiently large array for the breadth first search
    Returns
    -------
    connected_segments : 3D array of int, shape (Z, Y, X)
        A label field with connected labels starting at label=1
    """

    # get image dimensions
    cdef Py_ssize_t depth, height, width
    depth = segments.shape[0]
    height = segments.shape[1]
    width = segments.shape[2]

    # neighborhood arrays
    cdef Py_ssize_t[::1] ddx = np.array((1, -1, 0, 0, 0, 0), dtype=np.intp)
    cdef Py_ssize_t[::1] ddy = np.array((0, 0, 1, -1, 0, 0), dtype=np.intp)
    cdef Py_ssize_t[::1] ddz = np.array((0, 0, 0, 0, 1, -1), dtype=np.intp)

    # new object with connected segments initialized to -1
    cdef Py_ssize_t[:, :, ::1] connected_segments \
        = -1 * np.ones_like(segments, dtype=np.intp)

    cdef Py_ssize_t current_new_label = 0
    cdef Py_ssize_t label = 0

    # variables for the breadth first search
    cdef Py_ssize_t current_segment_size = 1
    cdef Py_ssize_t bfs_visited = 0
    cdef Py_ssize_t adjacent

    cdef Py_ssize_t zz, yy, xx

    cdef Py_ssize_t[:, ::1] coord_list = np.zeros((max_size, 3), dtype=np.intp)

    # loop through all image
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if connected_segments[z, y, x] >= 0:
                    continue
                # find the component size
                adjacent = 0
                label = segments[z, y, x]
                connected_segments[z, y, x] = current_new_label
                current_segment_size = 1
                bfs_visited = 0
                coord_list[bfs_visited, 0] = z
                coord_list[bfs_visited, 1] = y
                coord_list[bfs_visited, 2] = x

                #perform a breadth first search to find
                # the size of the connected component
                while bfs_visited != current_segment_size:
                    for i in range(6):
                        zz = coord_list[bfs_visited, 0] + ddz[i]
                        yy = coord_list[bfs_visited, 1] + ddy[i]
                        xx = coord_list[bfs_visited, 2] + ddx[i]
                        if (0 <= xx < width and
                                0 <= yy < height and
                                0 <= zz < depth):
                            if (segments[zz, yy, xx] == label and
                                    connected_segments[zz, yy, xx] == -1):
                                connected_segments[zz, yy, xx] = \
                                    current_new_label
                                coord_list[current_segment_size, 0] = zz
                                coord_list[current_segment_size, 1] = yy
                                coord_list[current_segment_size, 2] = xx
                                current_segment_size += 1
                            elif (connected_segments[zz, yy, xx] >= 0 and
                                  connected_segments[zz, yy, xx] != current_new_label):
                                adjacent = connected_segments[zz, yy, xx]
                    bfs_visited += 1

                # change to an adjacent one, like in the original paper
                if current_segment_size < min_size:
                    for i in range(current_segment_size):
                        connected_segments[coord_list[i, 0],
                                           coord_list[i, 1],
                                           coord_list[i, 2]] = adjacent
                else:
                    current_new_label += 1

    return np.asarray(connected_segments)
