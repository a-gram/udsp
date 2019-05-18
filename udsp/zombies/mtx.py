"""
This module defines functions and classes for matrix-vector
manipulation and algebra.

"""

import functools as _fun
import operator as _op
from . import utils as _utl


def mat_new(rows, cols, init=0):
    """
    Creates a new initialized matrix

    Parameters
    ----------
    rows: int
        The number of rows
    cols: int
        The number of columns
    init: scalar, callable, optional
        The initializer expression. It can be a scalar value or a
        callable object (function, lambda, etc.) that will be invoked
        for each element in the matrix. The callable object must have
        the signature f(n, m), where the arguments n,m indicate the
        indices of rows and columns respectively.

    Returns
    -------
    list[list]
        An initialized matrix

    """

    if callable(init):
        return [[init(n, m) for m in range(cols)]
                            for n in range(rows)]
    else:
        return [[init] * cols for _ in range(rows)]


def mat_copy(a):
    """
    Creates a duplicate of a matrix

    Parameters
    ----------
    a: list[list]
        A matrix of scalar values to be copied

    Returns
    -------
    list[list]
        A copy of the given matrix

    """
    if mat_empty(a):
        return a

    return [row.copy() for row in a]


def mat_empty(a):
    """
    Check whether a given matrix is empty

    A matrix is considered "empty" if it is None or has no
    elements.

    Parameters
    ----------
    a: list[list]
        The matrix to be checked

    Returns
    -------
    bool
        True if the matrix is empty, False otherwise

    """
    return a is None or len(a) == 0 or len(a[0]) == 0


def mat_dim(a):
    """
    Returns the dimensions of a matrix

    Note: it is assumed that all the rows have equal size

    Parameters
    ----------
    a: list[list]
        A matrix

    Returns
    -------
    tuple
        A 2-tuple (rows, cols)

    """
    return len(a), len(a[0])


def mat_dims_equal(a, b, full_check=False):
    """
    Checks whether two matrices have equal dimensions

    Parameters
    ----------
    a: list[list]
        A matrix
    b: list[list]
        A matrix
    full_check: bool, optional
        If False (default) then the check on the second dimension (number
        of columns) is done on the first rows only, that is a[0]==b[0] and
        it's assumed that all the others have the same size. If True then
        all the rows are compared.

    Returns
    -------
    bool
        True if the two matrices have equal dimensions, False otherwise.

    """
    if not full_check:
        return len(a) == len(b) and len(a[0]) == len(b[0])
    else:
        return len(a) == len(b) and all(
               map(lambda r1, r2: len(r1) == len(r2), a, b))
        # if not len(a) == len(b):
        #     return False
        # cols_equal = True
        # for n in range(len(a)):
        #     cols_equal &= len(a[n]) == len(b[n])
        # return cols_equal


def dot_product(a, b):
    """
    Computes the dot product of two vectors

    Parameters
    ----------
    a: list[]
        A vector of numeric values
    b: list[]
        A vector of numeric values

    Returns
    -------
    int, float

    """
    if len(a) != len(b):
        raise ValueError(
            "Incompatible vector sizes: len(a) != len(b)"
        )

    # return sum(map(operator.mul, a, b))

    prod = 0
    for i in range(len(a)):
        prod += a[i] * b[i]
    return prod


def mat_product(a, b):
    """
    Computes the product of two matrices

    Parameters
    ----------
    a: list[list]
        A matrix of scalar values
    b: list[list]
        A matrix of scalar values

    Returns
    -------
    list[list]
        A matrix product of a and b

    """
    if len(a[0]) != len(b):
        raise ValueError(
            "Incompatible matrix sizes: ncols(a) != nrows(b)"
        )
    # TODO: there is a nice matrix multiplication operator '@'
    #       that's built-in and is (i believe) faster

    dim1_a, dim2_a = range(len(a)), range(len(a[0]))
    dim1_b, dim2_b = range(len(b)), range(len(b[0]))

    bt = [[b[m][n] for m in dim1_b] for n in dim2_b]

    return [[dot_product(a[n], bt[m]) for m in dim2_b]
                                      for n in dim1_a]


def mat_add(a, b):
    """
    Computes the sum of two matrices, or of a matrix and a scalar

    Parameters
    ----------
    a: list[list]
        A matrix of scalar values
    b: list[list], scalar
        A matrix of scalar values or a scalar value.

    Returns
    -------
    list[list]
        A matrix sum of a and b

    """
    # if type(b) is list:
    #     assert len(a) == len(b) and len(a[0]) == len(b[0])
    #     # return [[a[n][m] - b[n][m] for m in range(len(a[0]))]
    #     #                            for n in range(len(a))]
    #     return [*map(lambda ra, rb: [ai + bi for ai, bi in zip(ra, rb)], a, b)]
    # else:
    #     # return [[a[n][m] - b for m in range(len(a[0]))]
    #     #                      for n in range(len(a))]
    #     return [*map(lambda ra: [ai + b for ai in ra], a)]

    if type(b) is not list:
        b = [b] * len(a)
    return [*map(vec_add, a, b)]


def mat_sub(a, b):
    """
    Computes the difference of two matrices, or of a matrix and a scalar

    Parameters
    ----------
    a: list[list]
        A matrix of scalar values
    b: list[list], scalar
        A matrix of scalar values or a scalar value.

    Returns
    -------
    list[list]
        A matrix difference of a and b

    """
    # if type(b) is list:
    #     assert len(a) == len(b) and len(a[0]) == len(b[0])
    #     # return [[a[n][m] - b[n][m] for m in range(len(a[0]))]
    #     #                            for n in range(len(a))]
    #     return [*map(lambda ra, rb: [ai - bi for ai, bi in zip(ra, rb)], a, b)]
    # else:
    #     # return [[a[n][m] - b for m in range(len(a[0]))]
    #     #                      for n in range(len(a))]
    #     return [*map(lambda ra: [ai - b for ai in ra], a)]

    if type(b) is not list:
        b = [b] * len(a)
    return [*map(vec_sub, a, b)]


def mat_mul(a, b):
    """
    Computes the element-wise (Hadamard) product of two matrices

    Parameters
    ----------
    a: list[list]
        A matrix of scalar values
    b: list[list], scalar
        A matrix of scalar values

    Returns
    -------
    list[list]
        A matrix product of a and b

    """
    # if type(b) is list:
    #     assert len(a) == len(b) and len(a[0]) == len(b[0])
    #     # return [[a[n][m] * b[n][m] for m in range(len(a[0]))]
    #     #                            for n in range(len(a))]
    #     return [*map(lambda ra, rb: [ai * bi for ai, bi in zip(ra, rb)], a, b)]
    # else:
    #     # return [[a[n][m] * b for m in range(len(a[0]))]
    #     #                      for n in range(len(a))]
    #     return [*map(lambda ra: [ai * b for ai in ra], a)]

    if type(b) is not list:
        b = [b] * len(a)
    return [*map(vec_mul, a, b)]


def mat_div(a, b):
    """
    Computes the element-wise quotient of two matrices

    Parameters
    ----------
    a: list[list]
        A matrix of scalar values
    b: list[list], scalar
        A matrix of scalar values

    Returns
    -------
    list[list]
        A matrix quotient of a and b

    """
    # if type(b) is list:
    #     assert len(a) == len(b) and len(a[0]) == len(b[0])
    #     # return [[a[n][m] / b[n][m] for m in range(len(a[0]))]
    #     #                            for n in range(len(a))]
    #     return [*map(lambda ra, rb: [ai / bi for ai, bi in zip(ra, rb)], a, b)]
    # else:
    #     # return [[a[n][m] / b for m in range(len(a[0]))]
    #     #                      for n in range(len(a))]
    #     return [*map(lambda ra: [ai / b for ai in ra], a)]

    if type(b) is not list:
        b = [b] * len(a)
    return [*map(vec_div, a, b)]


def mat_pow(a, p):
    """
    Computes the element-wise power of a matrix

    Parameters
    ----------
    a: list[list]
        A matrix of scalar values
    p: int, float
        The exponent

    Returns
    -------
    list[list]
        The exponential matrix of a

    """
    # return [[a[n][m] ** p for m in range(len(a[0]))]
    #                       for n in range(len(a))]
    # return [*map(lambda ra: [ai ** p for ai in ra], a)]
    return [*map(vec_pow, a, [p] * len(a))]


def mat_sum(a):
    """
    Computes the inner sum of a matrix

    Parameters
    ----------
    a: list[list]
        A matrix of scalar values

    Returns
    -------
    scalar
        The sum of the elements in the matrix

    """
    # suma = 0
    # for row in a:
    #     suma += sum(row)
    # return suma
    return sum(map(vec_sum, a))


def mat_prod(a):
    """
    Computes the "inner product" of a matrix

    NOTE: This can easily overflow. No checks are made.

    Parameters
    ----------
    a: list[list]
        A matrix of scalar values

    Returns
    -------
    scalar
        The product of the elements in the matrix

    """
    # prod = 1
    # for row in a:
    #     for e in row:
    #         prod *= e
    # return prod
    return _fun.reduce(_op.mul, map(vec_prod, a))


def mat_min(a):
    """
    Finds the minimum value in a matrix

    Parameters
    ----------
    a: list[list]
        A matrix of scalar values

    Returns
    -------
    scalar
        The minimum value in the matrix.

    """
    return vec_min([*map(vec_min, a)])

    # Solution 2 (slightly faster than the above)
    #
    # fmin, _ = _utl.get_min_max_f(a)
    #
    # mn = a[0][0]
    #
    # for row in a:
    #     for e in row:
    #         mn = fmin(mn, e)
    # return mn


def mat_max(a):
    """
    Finds the maximum value in a matrix

    Parameters
    ----------
    a: list[list]
        A matrix of scalar values

    Returns
    -------
    scalar
        The maximum value in the matrix.

    """
    return vec_max([*map(vec_max, a)])

    # Solution 2 (slightly faster than the above)
    #
    # mx = a[0][0]
    #
    # _, fmax = _utl.get_min_max_f(a)
    #
    # for row in a:
    #     for e in row:
    #         mx = fmax(mx, e)
    # return mx


def mat_min_max(a):
    """
    Finds the minimum and maximum value in a matrix

    Parameters
    ----------
    a: list[list]
        A matrix of scalar values

    Returns
    -------
    scalar
        A 2-tuple with the minimum and maximum value in the matrix.

    """
    fmin, fmax = _utl.get_min_max_f(a)

    mn = mx = a[0][0]
    for row in a:
        rmn, rmx = vec_min_max(row)
        mn = fmin(mn, rmn)
        mx = fmax(mx, rmx)
    return mn, mx


def mat_abs(a):
    """
    Computes the element-wise absolute value of a matrix

    Parameters
    ----------
    a: list[list]
        A matrix of scalar values

    Returns
    -------
    list[list]
        The absolute matrix of a

    """
    # return [[abs(a[n][m]) for m in range(len(a[0]))]
    #                       for n in range(len(a))]
    return [*map(vec_abs, a)]


def mat_neg(a):
    """
    Negates a matrix element-wise

    Parameters
    ----------
    a: list[list]
        A matrix of scalar values

    Returns
    -------
    list[list]
        The negated matrix of a

    """
    # return [[-a[n][m] for m in range(len(a[0]))]
    #                   for n in range(len(a))]
    return [*map(vec_neg, a)]


def mat_reverse(a, rows=True, cols=True):
    """
    Reverses a matrix along the rows and/or columns

    Parameters
    ----------
    a: list[list]
        A matrix of scalar values
    rows: bool, optional
        If True, the matrix is reversed along the rows
    cols: bool, optional
        If True, the matrix is reversed along the columns

    Returns
    -------
    list[list]
        The reversed matrix

    """
    if rows and cols:
        return [a[n][::-1] for n in reversed(range(len(a)))]
    elif cols:
        return a[::-1]
    elif rows:
        return [row[::-1] for row in a]
    else:
        return mat_copy(a)


def mat_submat(a, r):
    """
    Extracts a sub-matrix from a given matrix

    Parameters
    ----------
    a: list[list]
        A matrix of scalar values
    r: tuple
        A tuple with two pairs (cs, ce), (rs, re) indicating the start
        and end points of the sub-matrix dimensions along the columns and
        rows respectively. If cs>ce (or rs>re) then the sub-matrix will be
        reversed along both dimensions. Values must be positive.

    Returns
    -------
    list[list]
        The sub-matrix

    """
    assert len(r) == 2 and len(r[0]) == 2 and len(r[1]) == 2
    assert min(r[0]) >= 0 and min(r[1]) >= 0

    no, n1 = r[0][0], r[0][1]
    mo, m1 = r[1][0], r[1][1]

    stepn = 1 if no <= n1 else -1
    stepm = 1 if mo <= m1 else -1

    ne = n1 + stepn

    # Solution 1
    #
    # return ([a[n][mo:: stepm] for n in range(no, ne, stepn)]
    #         if mo > m1 == 0 else
    #         [a[n][mo: m1 + stepm: stepm] for n in range(no, ne, stepn)])

    # Solution 2
    #
    b = [None] * (abs(n1 - no) + 1)

    for n in range(no, ne, stepn):
        b[abs(n - no)] = (a[n][mo:: stepm] if mo > m1 == 0 else
                          a[n][mo: m1 + stepm: stepm])
    return b


def mat_to(newtype, a):
    """
    Converts the elements of a matrix to a specified scalar type

    Parameters
    ----------
    newtype: class
        The new scalar type to which the elements of the matrix
        will be converted.
    a: list[list]
        A matrix of scalar values

    Returns
    -------
    list[list]
        A matrix whose elements are converted to the specified type.
        If the matrix is already of the given type, a copy is returned.

    """
    if type(a[0][0]) is not newtype:
        return [[newtype(e) for e in row] for row in a]
    else:
        return [[e for e in row] for row in a]


# ---------------------------------------------------------
#                        VECTORS
# ---------------------------------------------------------


def vec_new(elems, init=0):
    """
    Creates a new initialized vector

    Parameters
    ----------
    elems: int
        The number of elements
    init: scalar, callable, optional
        The initializer expression. It can be a scalar value or a
        callable object (function, lambda, etc.) that will be invoked
        for each element in the matrix. The callable object must have
        the signature f(n), where the argument n indicates the elements.

    Returns
    -------
    list[]
        An initialized vector

    """
    if callable(init):
        return [init(n) for n in range(elems)]
    else:
        return [init] * elems


def vec_copy(a):
    """
    Creates a duplicate (deep copy) of a vector

    Parameters
    ----------
    a: list[]
        The vector to be copied

    Returns
    -------
    list[]
        A copy of the given vector

    """
    return a.copy()


def vec_empty(a):
    """
    Check whether a given vector is empty

    A vector is considered "empty" if it is None or has no
    elements.

    Parameters
    ----------
    a: list[]
        The vector to be checked

    Returns
    -------
    bool
        True if the vector is empty, False otherwise

    """
    return a is None or len(a) == 0


def vec_dims_equal(a, b):
    """
    Checks whether two vectors have equal dimensions

    Parameters
    ----------
    a: list[]
        A vector
    b: list[]
        A vector

    Returns
    -------
    bool
        True if the two vectors have equal dimensions, False otherwise.

    """
    return len(a) == len(b)


def vec_add(a, b):
    """
    Computes the sum of two vectors, or of a vector and a scalar

    Parameters
    ----------
    a: list[]
        A vector of scalar values
    b: list[]
        A vector of scalar values or a scalar value.

    Returns
    -------
    list[]
        A vector sum of a and b

    """
    if type(b) is list:
        assert len(a) == len(b)
        # return [a[n] + b[n] for n in range(len(a))]
        return [*map(lambda ai, bi: ai + bi, a, b)]
    else:
        # return [a[n] + b for n in range(len(a))]
        return [*map(lambda ai: ai + b, a)]


def vec_sub(a, b):
    """
    Computes the difference of two vectors, or of a vector and a scalar

    Parameters
    ----------
    a: list[]
        A vector of scalar values
    b: list[]
        A vector of scalar values or a scalar value.

    Returns
    -------
    list[]
        A vector difference of a and b

    """
    if type(b) is list:
        assert len(a) == len(b)
        # return [a[n] - b[n] for n in range(len(a))]
        return [*map(lambda ai, bi: ai - bi, a, b)]
    else:
        # return [a[n] - b for n in range(len(a))]
        return [*map(lambda ai: ai - b, a)]


def vec_mul(a, b):
    """
    Computes the vector-vector or vector-scalar element-wise product

    Parameters
    ----------
    a: list[]
        A vector of scalar values
    b: list[]
        A vector of scalar values or a scalar value.

    Returns
    -------
    list[]
        A vector product of a and b

    """
    if type(b) is list:
        assert len(a) == len(b)
        # return [a[n] * b[n] for n in range(len(a))]
        return [*map(lambda ai, bi: ai * bi, a, b)]
    else:
        # return [a[n] * b for n in range(len(a))]
        return [*map(lambda ai: ai * b, a)]


def vec_div(a, b):
    """
    Computes the vector-vector or vector-scalar element-wise quotient

    Parameters
    ----------
    a: list[]
        A vector of scalar values
    b: list[]
        A vector of scalar values or a scalar value.

    Returns
    -------
    list[]
        A vector quotient of a and b

    """
    if type(b) is list:
        assert len(a) == len(b)
        # return [a[n] / b[n] for n in range(len(a))]
        return [*map(lambda ai, bi: ai / bi, a, b)]
    else:
        # return [a[n] / b for n in range(len(a))]
        return [*map(lambda ai: ai / b, a)]


def vec_pow(a, p):
    """
    Computes the element-wise power of a vector

    Parameters
    ----------
    a: list[]
        A vector of scalar values
    p: int, float
        The exponent

    Returns
    -------
    list[]
        The exponential vector of a

    """
    # return [a[n] ** p for n in range(len(a))]
    return [*map(lambda ai: ai ** p, a)]


def vec_sum(a):
    """
    Computes the inner sum of a vector

    Parameters
    ----------
    a: list[]
        A vector of scalar values

    Returns
    -------
    scalar
        The sum of the elements in the vector

    """
    return sum(a)


def vec_prod(a):
    """
    Computes the "inner product" of a vector

    NOTE: This can easily overflow. No checks are made.

    Parameters
    ----------
    a: list[]
        A vector of scalar values

    Returns
    -------
    scalar
        The product of the elements in the vector

    """
    prod = 1
    for e in a:
        prod *= e
    return prod


def vec_min(a):
    """
    Finds the minimum value in a vector

    Parameters
    ----------
    a: list[]
        A vector of scalar values

    Returns
    -------
    scalar
        The minimum value in the vector. If the vector is complex then
        the elements are compared by magnitude and phase.

    """
    fmin, _ = _utl.get_min_max_f(a)

    # mn = a[0]
    #
    # for e in a:
    #     mn = fmin(mn, e)
    # return mn

    return _fun.reduce(fmin, a)


def vec_max(a):
    """
    Finds the maximum value in a vector

    Parameters
    ----------
    a: list[]
        A vector of scalar values

    Returns
    -------
    scalar
        The maximum value in the vector. If the vector is complex then
        the elements are compared by magnitude and phase.

    """
    _, fmax = _utl.get_min_max_f(a)

    # mx = a[0]
    #
    # for e in a:
    #     mx = fmax(mx, e)
    # return mx

    return _fun.reduce(fmax, a)


def vec_min_max(a):
    """
    Finds the minimum and maximum value in a vector

    Parameters
    ----------
    a: list[]
        A vector of scalar values

    Returns
    -------
    tuple
        A 2-tuple with the minimum and maximum value. If the vector
        is complex then the elements are compared by magnitude and phase.

    """
    fmin, fmax = _utl.get_min_max_f(a)

    # return _fun.reduce(
    #     lambda x, y: (fmin(x[0], y[0]), fmax(x[1], y[1])),
    #     zip(a, a)
    # )

    mn = mx = a[0]

    for e in a:
        mn = fmin(mn, e)
        mx = fmax(mx, e)
    return mn, mx


def vec_abs(a):
    """
    Computes the element-wise absolute value of a vector

    Parameters
    ----------
    a: list[]
        A vector of scalar values

    Returns
    -------
    list[]
        The absolute vector of a

    """
    # return [abs(a[n]) for n in range(len(a))]
    return [*map(lambda ai: abs(ai), a)]


def vec_neg(a):
    """
    Negates a vector element-wise

    Parameters
    ----------
    a: list[]
       A vector of scalar values

    Returns
    -------
    list[]
        The negated vector of a

    """
    # return [-a[n] for n in range(len(a))]
    return [*map(lambda ai: -ai, a)]


def vec_reverse(a):
    """
    Reverses a vector

    Parameters
    ----------
    a: list[]
        A vector of scalar values

    Returns
    -------
    list[]
        The reversed vector

    """
    return a[::-1]


def vec_subvec(a, r):
    """
    Extracts a sub-vector from a given vector

    Parameters
    ----------
    a: list[]
        A vector of scalar values
    r: tuple
        A pair (ps, pe) indicating the start and end points of the
        sub-vector dimension. If ps>pe then the sub-vector will be
        reversed. Values must be positive.

    Returns
    -------
    list[]
        The sub-vector

    """
    assert len(r) == 2
    assert min(r) >= 0

    step = 1 if r[0] <= r[1] else -1

    return (a[r[0]:: step] if r[0] > r[1] == 0 else
            a[r[0]: r[1] + step: step])


def vec_to(newtype, a):
    """
    Converts the elements of a vector to a specified scalar type

    Parameters
    ----------
    newtype: class
        The new scalar type to which the elements of the vector
        will be converted.
    a: list[list]
        A vector of scalar values

    Returns
    -------
    list[]
        A vector whose elements are converted to the specified type.
        If the vector is already of the given type, a copy is returned.

    """
    if type(a[0]) is not newtype:
        return [newtype(e) for e in a]
    else:
        return [e for e in a]
