#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
from libc.math cimport sqrt

import numpy as np


MAX_FLOAT64 = np.finfo(np.float64).max


cdef double sqeuclidean_dist(double [::1] point0, double[::1] point1):
    """Compute the squared euclidean distance between two points.

    Parameters
    ----------
    point0, point1 : 1D arrays of double, same size
        The two points for which to compare the distance.

    Returns
    -------
    dist : double
        The squared euclidean distance between point0 and point1.
    """
    cdef Py_ssize_t i, dim = point0.shape[0]
    cdef double diff, dist = 0.
    for i in range(dim):
        diff = point0[i] - point1[i]
        dist += diff * diff
    return dist


cdef inline double dmin(double x, double y):
    return x if x < y else y

cdef inline double dmax(double x, double y):
    return x if x > y else y

def hausdorff_distance_onesided(double[:, ::1] points_src,
                                double[:, ::1] points_dst):
    """Compute the one-sided Haussdorff distance between two sets of points.

    This is defined as the longest trip that one can make from a point
    in `points_src` to its nearest point in `points_dst`.
    """
    cdef double maxmin = 0.
    cdef double currmin = MAX_FLOAT64, d = 0.
    cdef Py_ssize_t i_src, i_dst

    for i_src in range(points_src.shape[0]):
        currmin = MAX_FLOAT64
        for i_dst in range(points_dst.shape[0]):
            d = sqeuclidean_dist(points_src[i_src], points_dst[i_dst])
            currmin = dmin(currmin, d)
            if currmin < maxmin:
                break  # trip from current point already shorter than curr. max
        maxmin = dmax(maxmin, currmin)
    return sqrt(maxmin)
