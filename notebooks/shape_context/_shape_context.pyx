# encoding: utf-8
#cython: cdivision=True
#cython: boundscheck=False
# cython: wraparound=False

from collections import defaultdict
import numpy as np

cimport numpy as np
cimport cython

np.import_array()


def _pixel_graph(char[:, :] img):
    """ Create an 8-way pixel connectivity graph for a binary image."""
    cdef int m, n
    cdef int i, j, imod, jmod
    adj = defaultdict(set)
    m = img.shape[0]
    n = img.shape[1]
    for i in range(m):
        for j in range(n):
            for imod in (-1, 0, 1):
                for jmod in (-1, 0, 1):
                    if imod == jmod == 0:
                        continue
                    if i + imod < 0 or i + imod >= m:
                        continue
                    if j + jmod < 0 or j + jmod >= n:
                        continue
                    if img[i, j] and img[i + imod, j + jmod]:
                        adj[i, j].add((i + imod, j + jmod))
                        adj[i + imod, j + jmod].add((i, j))
    return adj


def chi2_distance(long[:] x,
                  long[:] y):
    """Chi-square histogram distance.

    Ignores bins with no elements.

    """
    cdef long i
    cdef long n = len(x)
    cdef long x_elt
    cdef long y_elt
    cdef double x_max = 0
    cdef double y_max = 0
    cdef double x_norm_elt
    cdef double y_norm_elt
    cdef double d
    cdef double s
    cdef double result = 0
    for i in range(n):
        x_max = max(x_max, x[i])
        y_max = max(y_max, y[i])
    if x_max == 0 or y_max == 0:
        raise Exception('empty histogram')
    for i in range(n):
        x_elt = x[i]
        y_elt = y[i]
        if x_elt == 0 and y_elt == 0:
            continue
        x_norm_elt = x_elt / x_max
        y_norm_elt = y_elt / y_max
        d = x_norm_elt - y_norm_elt
        s = x_norm_elt + y_norm_elt
        result += (d * d) / s
    return result / 2
