#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
from libc.math cimport sqrt

import numpy as np


cdef double MAX_FLOAT64 = np.finfo(np.float64).max


cdef double sqeuclidean_dist(double[:, ::1] points0, Py_ssize_t pt0,
                             double[:, ::1] points1, Py_ssize_t pt1):
    """Compute the squared euclidean distance between two points.

    Parameters
    ----------
    points0 : 2D array of double, shape (M, K)
        The array containing the source point.
    pt0 : int
        The index of the source point within ``points0``.
    points1 : 2D array of double, shape (N, K)
        The array containing the destination point.
    pt1 : int
        The index of the destination point within ``points1``.

    Returns
    -------
    dist : double
        The squared euclidean distance between point0 and point1.
    """
    cdef Py_ssize_t i, dim = points0.shape[1]
    cdef double diff, dist = 0.
    for i in range(dim):
        diff = points0[pt0, i] - points1[pt1, i]
        dist += diff * diff
    return dist


cdef inline double dmin(double x, double y):
    return x if x < y else y

cdef inline double dmax(double x, double y):
    return x if x > y else y


def hausdorff_distance_onesided(double[:, ::1] points_src,
                                double[:, ::1] points_dst):
    """Compute the one-sided Hausdorff distance between two sets of points.

    This is defined as the longest trip that one can make from a point
    in `points_src` to its nearest point in `points_dst`.

    An equivalent expression using the function ``distance_matrix`` from
    ``scipy.spatial`` would be:

        ``np.max(np.min(distance_matrix(points_src, points_dst), axis=1))``

    This function, however, contains several performance improvements
    over that solution.

    Parameters
    ----------
    points_src, points_dst : 2D arrays of double, shape (M, K) and (N, K)
        Two arrays of spatial coordinates in K dimensions.

    Returns
    -------
    dist : double
        The one-sided Hausdorff distance between points in ``points_src``
        and points in ``points_dst``.
    """
    cdef double maxmin = 0.
    cdef double currmin = MAX_FLOAT64, d = 0.
    cdef Py_ssize_t i_src, i_dst

    for i_src in range(points_src.shape[0]):
        currmin = MAX_FLOAT64
        for i_dst in range(points_dst.shape[0]):
            d = sqeuclidean_dist(points_src, i_src, points_dst, i_dst)
            currmin = dmin(currmin, d)
            if currmin < maxmin:
                break  # trip from current point already shorter than curr. max
        maxmin = dmax(maxmin, currmin)
    return sqrt(maxmin)
