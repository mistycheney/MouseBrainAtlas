#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
from libc.float cimport DBL_MAX
from cpython cimport bool

import numpy as np
cimport numpy as cnp

from _regular_grid import regular_grid

def _slic_cython(double[:, :, ::1] image_yx,
                 double[:, ::1] segments,
                 float step,
                 Py_ssize_t max_iter,
                 double[::1] spacing,
                 bint slic_zero):

    # initialize on grid
    cdef Py_ssize_t height, width
    height = image_yx.shape[0]
    width = image_yx.shape[1]

    cdef Py_ssize_t n_segments = segments.shape[0]
    # number of features [X, Y, Z, ...]
    cdef Py_ssize_t n_features = segments.shape[1]

    # approximate grid size for desired n_segments
    cdef Py_ssize_t step_y, step_x
    slices = regular_grid((height, width), n_segments)
    step_y, step_x = [int(s.step) for s in slices]

    cdef Py_ssize_t[:, ::1] nearest_segments \
        = np.empty((height, width), dtype=np.intp)
    cdef double[:, ::1] distance \
        = np.empty((height, width), dtype=np.double)
    cdef Py_ssize_t[::1] n_segment_elems = np.zeros(n_segments, dtype=np.intp)

    cdef Py_ssize_t i, c, k, x, y, x_min, x_max, y_min, y_max
    cdef char change
    cdef double dist_center, cx, cy, dy

    cdef double sy, sx
    sy = spacing[0]
    sx = spacing[1]
    
    # The colors are scaled before being passed to _slic_cython so
    # max_color_sq can be initialised as all ones
    cdef double[::1] max_dist_color = np.ones(n_segments, dtype=np.double)
    cdef double dist_color

    # The reference implementation (Achanta et al.) calls this invxywt
    cdef double spatial_weight = float(1) / (step ** 2)

#     with nogil:
    for i in range(max_iter):
        change = 0
        distance[:, :] = DBL_MAX

        # assign pixels to segments
        for k in range(n_segments):

            # segment coordinate centers
            cy = segments[k, 0]
            cx = segments[k, 1]

            # compute windows
            y_min = <Py_ssize_t>max(cy - 2 * step_y, 0)
            y_max = <Py_ssize_t>min(cy + 2 * step_y + 1, height)
            x_min = <Py_ssize_t>max(cx - 2 * step_x, 0)
            x_max = <Py_ssize_t>min(cx + 2 * step_x + 1, width)

            for y in range(y_min, y_max):
                dy = (sy * (cy - y)) ** 2
                for x in range(x_min, x_max):
                    dist_center = (dy + (sx * (cx - x)) ** 2) * spatial_weight

                    dist_color = 0
                    for c in range(2, n_features):
                        dist_color += (image_yx[y, x, c-2] - segments[k, c]) ** 2
#                     print dist_color

                    dist_center += dist_color / 2.

                    if distance[y, x] > dist_center:
                        nearest_segments[y, x] = k
                        distance[y, x] = dist_center
                        change = 1

        # stop if no pixel changed its segment
        if change == 0:
            break

        # recompute segment centers

        # sum features for all segments
        n_segment_elems[:] = 0
        segments[:, :] = 0
        for y in range(height):
            for x in range(width):
                k = nearest_segments[y, x]
                n_segment_elems[k] += 1
                segments[k, 0] += y
                segments[k, 1] += x
                for c in range(2, n_features):
                    segments[k, c] += image_yx[y, x, c-2]

        # divide by number of elements per segment to obtain mean
        for k in range(n_segments):
            for c in range(n_features):
                segments[k, c] /= n_segment_elems[k]
                
        # If in SLICO mode, update the color distance maxima
#             if slic_zero:
#                 for z in range(depth):
#                     for y in range(height):
#                         for x in range(width):

#                             k = nearest_segments[z, y, x]
#                             dist_color = 0

#                             for c in range(3, n_features):
#                                 dist_color += (image_zyx[z, y, x, c - 3] -
#                                                segments[k, c]) ** 2

#                             # The reference implementation seems to only change
#                             # the color if it increases from previous iteration
#                             if max_dist_color[k] < dist_color:
#                                 max_dist_color[k] = dist_color

    return np.asarray(nearest_segments)


def _enforce_label_connectivity_cython(Py_ssize_t[:, ::1] segments,
                                       Py_ssize_t n_segments,
                                       Py_ssize_t min_size,
                                       Py_ssize_t max_size):


    # get image dimensions
    cdef Py_ssize_t height, width
    height = segments.shape[0]
    width = segments.shape[1]

    # neighborhood arrays
    cdef Py_ssize_t[::1] ddx = np.array((1, -1, 0, 0, 0, 0), dtype=np.intp)
    cdef Py_ssize_t[::1] ddy = np.array((0, 0, 1, -1, 0, 0), dtype=np.intp)

    # new object with connected segments initialized to -1
    cdef Py_ssize_t[:, ::1] connected_segments \
        = -1 * np.ones_like(segments, dtype=np.intp)

    cdef Py_ssize_t current_new_label = 0
    cdef Py_ssize_t label = 0

    # variables for the breadth first search
    cdef Py_ssize_t current_segment_size = 1
    cdef Py_ssize_t bfs_visited = 0
    cdef Py_ssize_t adjacent

    cdef Py_ssize_t yy, xx

    cdef Py_ssize_t[:, ::1] coord_list = np.zeros((max_size, 2), dtype=np.intp)

    for y in range(height):
        for x in range(width):
            if connected_segments[y, x] >= 0:
                continue
            # find the component size
            adjacent = 0
            label = segments[y, x]
            connected_segments[y, x] = current_new_label
            current_segment_size = 1
            bfs_visited = 0
            coord_list[bfs_visited, 0] = y
            coord_list[bfs_visited, 1] = x

            #perform a breadth first search to find
            # the size of the connected component
            while bfs_visited < current_segment_size < max_size:
                for i in range(6):
                    yy = coord_list[bfs_visited, 0] + ddy[i]
                    xx = coord_list[bfs_visited, 1] + ddx[i]
                    if (0 <= xx < width and
                            0 <= yy < height):
                        if (segments[yy, xx] == label and
                                connected_segments[yy, xx] == -1):
                            connected_segments[yy, xx] = \
                                current_new_label
                            coord_list[current_segment_size, 0] = yy
                            coord_list[current_segment_size, 1] = xx
                            current_segment_size += 1
                            if current_segment_size >= max_size:
                                break
                        elif (connected_segments[yy, xx] >= 0 and
                              connected_segments[yy, xx] != current_new_label):
                            adjacent = connected_segments[yy, xx]
                bfs_visited += 1

            # change to an adjacent one, like in the original paper
            if current_segment_size < min_size:
                for i in range(current_segment_size):
                    connected_segments[coord_list[i, 0],
                                       coord_list[i, 1]] = adjacent
            else:
                current_new_label += 1

    return np.asarray(connected_segments)
