"""
This module defines miscellaneous utility functions used
throughout the library.

"""

# import math
import operator as _op
import functools as _func
import cmath as _cm


def round_pow2(n):
    """
    Rounds a number to the nearest larger or equal power of 2

    Parameters
    ----------
    n: int, float
        The number to be rounded. Must be positive.

    Returns
    -------
    int
        The nearest power of 2 larger or equal to n, that is
        m such that n <= m=2^k.

    """
    p2 = 1 if n > 1 else 2
    while p2 < n:
        p2 *= 2
    return p2

    # Solution 2
    # return int(math.pow(2, math.ceil(math.log2(n))))


def is_pow2(n):
    """
    Check whether a number is power of 2

    Parameters
    ----------
    n: int
        A positive integer number

    Returns
    -------
    bool

    """
    return n > 0 and ((n & (n - 1)) == 0)
    # Solution 2
    # return math.log2(n) % 1 == 0


def product(a):
    """
    Computes the product of all elements in an iterable

    Parameters
    ----------
    a: iterable

    Returns
    -------
    scalar
        The product of all elements in the iterable

    """
    return _func.reduce(_op.mul, a)


def floor_pow2(n):
    """
    Returns the nearest power of 2 <= n

    """
    k = 0
    n >>= 1
    while n:
        k += 1
        n >>= 1
    return k


def swap2(i, j, x):
    """
    Swaps two elements in a sequence

    Parameters
    ----------
    i: int
        The first index in x that must be swapped
    j: int
        The second index in x that must be swapped
    x: list[]
        A sequence with the elements i,j to be swapped

    Returns
    -------
    None

    """
    t = x[i]
    x[i] = x[j]
    x[j] = t


def to_meshgrid(x):
    """
    Converts a 2D grid of points (x,y) into a mesh grid

    Parameters
    ----------
    x: list[list]
        A 2D array of 2-tuples (x,y)

    Returns
    -------
    tuple
        A 2-tuple with the X and Y coordinates of the mesh

    """
    dim1r, dim2r = range(len(x)), range(len(x[0]))
    x1 = [[x[n][m][1] for m in dim2r] for n in dim1r]
    x2 = [[x[n][m][0] for m in dim2r] for n in dim1r]
    return x1, x2

    # Nr, Mr = range(len(X)), range(len(X[0]))
    # N = [[0 for _ in Mr] for _ in Nr]
    # M = [[0 for _ in Mr] for _ in Nr]
    #
    # for n in Nr:
    #     for m in Mr:
    #         N[n][m] = X[n][m][0]
    #         M[n][m] = X[n][m][1]
    #
    # return N, M


def all_same(v, array):
    """
    Checks whether all elements in an array ar equal to a given value

    Parameters
    ----------
    v: scalar
        The value to be checked against
    array: list[]
        The array to be checked

    Returns
    -------
    bool
        True if all elements in the array are equal to the given one

    """
    return all(x == v for x in array) and len(array) > 0
    # Solution 2:
    #
    # array.count(v) == len(array) and len(array) > 0

    # Solution 3:
    #
    # for x in array:
    #     if x != v:
    #         return False
    # return True


def cmin(a, b):
    """
    Min function for complex numbers

    Note: there are typically two approaches to defining an
          ordering function (min/max) for complex numbers:
          1) consider them as pairs (Re, Im) and compare
             these pairs by Re and Im respectively
          2) consider their polar representation (Mag, Phi)
             and compare by magnitude and phase
          Here we use method 2

    Parameters
    ----------
    a: complex
    b: complex

    Returns
    -------
    complex

    """
    ac, bc = (abs(a), _cm.phase(a)), (abs(b), _cm.phase(b))
    return a if ac < bc else b


def cmax(a, b):
    """
    Max function for complex numbers

    Parameters
    ----------
    a: complex
    b: complex

    Returns
    -------
    complex

    """
    ac, bc = (abs(a), _cm.phase(a)), (abs(b), _cm.phase(b))
    return a if ac > bc else b


def rmin(a, b):
    """
    Min function for real numbers

    Parameters
    ----------
    a: scalar
    b: scalar

    Returns
    -------
    scalar

    """
    # seems to be much faster than the built-in
    return a if a < b else b


def rmax(a, b):
    """
    Max function for real numbers

    Parameters
    ----------
    a: scalar
    b: scalar

    Returns
    -------
    scalar

    """
    # seems to be much faster than the built-in
    return a if a > b else b


def get_min_max_f(array):
    """
    Gets the min/max functions based on array type

    This is a refactored convenience function that returns the
    min() and max() functions based on the type of elements in
    the given array.

    Parameters
    ----------
    array: list[], list[list]

    Returns
    -------
    function
        A 2-tuple (min, max) with the min() and max() functions

    """
    try:
        is_cplx = type(array[0][0]) is complex
    except TypeError:
        is_cplx = type(array[0]) is complex
    return (cmin, cmax) if is_cplx else (rmin, rmax)


def isiterable(obj):
    """
    Check whether a given object is iterable

    Parameters
    ----------
    obj: Iterable

    Returns
    -------
    bool

    """
    try:
        _ = iter(obj)
    except TypeError:
        return False
    return True

