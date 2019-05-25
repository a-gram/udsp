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
    if mat_is_void(a):
        return a

    return [row.copy() for row in a]


def mat_is_void(a):
    """
    Check whether a given matrix is void

    A matrix is considered as "void" if it is None or []

    Parameters
    ----------
    a: list[list]
        The matrix to be checked

    Returns
    -------
    bool
        True if the matrix is void, False otherwise

    """
    return a is None or len(a) == 0


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
    if mat_is_void(a):
        return ()
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


def is_mat(a):
    """
    Check whether the given argument is a matrix

    Parameters
    ----------
    a

    Returns
    -------
    bool

    """
    return type(a) is list and len(a) and type(a[0]) is list


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
    float, None

    """
    if vec_is_void(a) or vec_is_void(b):
        return None

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
    b: list[list], list[]
        A matrix of scalar values or a vector

    Returns
    -------
    list[list]
        A matrix product of a and b

    """
    if mat_is_void(a) or mat_is_void(b):
        return []
    if len(a[0]) != len(b):
        raise ValueError(
            "Incompatible matrix sizes: ncols(a) != nrows(b)"
        )

    # TODO: there is a nice matrix multiplication operator '@'
    #       that's built-in and is (i believe) faster

    # arows, acols = range(len(a)), range(len(a[0]))
    # brows, bcols = range(len(b)), range(len(b[0]))
    #
    # bt = [[b[n][m] for n in brows] for m in bcols]
    #
    # return [[dot_product(a[n], bt[m]) for m in bcols]
    #                                   for n in arows]

    # mat-mat product
    if is_mat(b):
        return [[dot_product(row, col) for col in zip(*b)]
                                       for row in a]
    # mat-vector product
    else:
        return [dot_product(row, b) for row in a]


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
        and end points of the sub-matrix along the columns and
        rows respectively. If cs>ce (or rs>re) then the sub-matrix will be
        reversed along the dimensions. All values must be positive.

    Returns
    -------
    list[list]
        The sub-matrix

    """
    if mat_is_void(a):
        return []
    if (not r or len(r) != 2 or len(r[0]) != 2 or len(r[1]) != 2
              or min(r[0]) < 0 or min(r[1]) < 0):
        raise ValueError("Invalid sub matrix dimensions. Must be a "
                         "tuple with 2 pairs ((cs, ce), (rs, re)).")

    nmin, nmax = 0, len(a) - 1
    mmin, mmax = 0, len(a[0]) - 1

    no, n1 = r[0][0], r[0][1]
    mo, m1 = r[1][0], r[1][1]

    # For out of bound ranges, return an empty array
    if (no > nmax and n1 > nmax or no < nmin and n1 < nmin or
        mo > mmax and m1 > mmax or mo < mmin and m1 < mmin):
        return []
    # Clip the limits
    no, n1 = max(min(no, nmax), nmin), max(min(n1, nmax), nmin)
    mo, m1 = max(min(mo, mmax), mmin), max(min(m1, mmax), mmin)

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


def mat_submat_copy(a, b, offset, inplace=True):
    """

    Parameters
    ----------
    a: list[list]
        A matrix of scalar values represnting the main matrix
    b: list[list]
        A matrix of scalar values representing the sub-matrix
    offset: tuple
        A 2-tuple (row, col) indicating the offset within the
        main matrix at which to copy the submatrix
    inplace: bool
        Indicates whether to modify the original matrix or make
        a copy of it

    Returns
    -------
    list[list]
        A matrix with the sub-matrix copied onto it

    """
    if mat_is_void(a):
        return []
    if mat_is_void(b):
        return a
    if not offset or min(offset) < 0:
        raise ValueError("Invalid offset. Must be a 2-tuple of int >=0")

    m = a if inplace else mat_copy(a)
    arows, acols = mat_dim(m)
    brows, bcols = mat_dim(b)
    no, co = offset
    if (no, co) >= (arows, acols):
        return m

    for row in b:
        blen = min(acols - co, bcols)
        m[no][co: co + blen] = row[:blen]
        no += 1
        if no >= arows:
            break
    return m


def mat_flatten(a, mode=(1, False)):
    """
    Flattens (vectorizes) a matrix

    Parameters
    ----------
    a: list[list]
        A matrix of scalar values
    mode: tuple
        A 2-tuple in the form (dim, inverted) indicating how to
        perform the flattening. The 'dim' value must be either 1
        or 2, where 1 means that the matrix is flattened using the
        first dimension (rows) starting from the first. If it's 2
        then it is flattened using the second dimension (columns)
        starting from the first. The 'inverted' value indicates
        whether to invert the order of the flattening, that is
        starting from the last row/column.

    Returns
    -------
    list[]
        A vectorized form of the given matrix

    """
    if mat_is_void(a):
        return []
    if len(mode) < 2 or not mode[0] in (1, 2):
        raise ValueError("Invalid mode. Must be a 2-tuple "
                         "in the form ({1,2}, {True,False})")

    nrows, ncols = mat_dim(a)
    dim, inverted = mode
    vec = [None] * (nrows * ncols)

    if dim == 1:
        it = reversed(a) if inverted else a
        for vo, row in enumerate(it):
            dr = vo * ncols
            vec[dr: dr + ncols] = row
    else:
        it = [[*reversed(row)] for row in a] if inverted else a
        for vo, col in enumerate(zip(*it)):
            dr = vo * nrows
            vec[dr: dr + nrows] = col
    return vec


def mat_unflatten(v, size, mode=(1, False)):
    """
    Restores a vector to a matrix of the given size

    Parameters
    ----------
    v: list[]
        The vector to be "unflattened"
    size: tuple
        A 2-tuple representing the size of the resulting matrix (rows,
        columns)
    mode: tuple
        The unflattening mode. This paramater can be used to restore
        a matrix that's been vectorized with the flatten() method using
        a specific mode. A sort of inverse function. By default, the
        matrix is restored by rows segmenting the vector from start to
        end using the given row size.

    Returns
    -------
    list[list]
        A matrix

    """
    if vec_is_void(v):
        return []
    if not size or len(size) < 2 or min(size) <= 0:
        raise ValueError("Invalid size. Must be a 2-tuple of int > 0")
    if not mode or len(mode) < 2 or not mode[0] in (1, 2):
        raise ValueError("Invalid mode. Must be a 2-tuple "
                         "in the form ({1,2}, {True,False})")
    if len(v) / size[1] != size[0]:
        raise ValueError("Can't unflat the vector to the given size")

    rows, cols = size
    # Mode rows (dim=1, inverted=False)
    if mode == (1, False):
        return [v[r * cols: (r + 1) * cols]
                for r in range(rows)]
    # Mode rows inverted (dim=1, inverted=True)
    elif mode == (1, True):
        return [v[r * cols: (r + 1) * cols]
                for r in reversed(range(rows))]
    # Mode cols (dim=2 inverted=False)
    elif mode == (2, False):
        step = None if rows == 1 or cols == 1 else 2
        return [v[r: (r + 1) if cols == 1 else None: step]
                for r in range(rows)]
    # Mode cols inverted (dim=2 inverted=True)
    elif mode == (2, True):
        # if rows == 1:
        #     return v.copy().reverse()
        # if cols == 1:
        #     return [[e] for e in v]
        # else:
        #     return [v[-r-1:: -2] for r in reversed(range(rows))]

        step = None if cols == 1 else (-1 if rows == 1 else -2)
        return [v[-(r + 1): (rows - r) if cols == 1 else None: step]
                for r in reversed(range(rows))]
    else:
        raise RuntimeError


def mat_pad(a, p, val=0):
    """
    Pads a matrix

    Parameters
    ----------
    a: list[list]
        A matrix of scalar values
    p: tuple
        A 4-tuple (top, bottom, left, right) indicating the padding
        sizes on the first (top, bottom) and second (left, right)
        dimension respectively
    val: scalar
        A constant value used for the padding (default is 0)

    Returns
    -------
    list[list]
        A padded matrix

    """
    if mat_is_void(a):
        return []
    if not p or len(p) < 4 or min(p) < 0:
        raise ValueError(
            "Invalid padding size. Must be a 4-tuple of int >= 0"
        )

    arows, acols = mat_dim(a)
    top, bottom, left, right = p
    ap = mat_new(arows + top + bottom, acols + left + right, val)
    for n, row in enumerate(a):
        ap[top + n][left: left + acols] = row
    return ap


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


def mat_toeplitz(h, g):
    """
    Constructs a Toeplitz matrix from the given sequences

    Parameters
    ----------
    h: list[]
        A sequence defining the matrix for non-negative indices.
        This will define the number of rows.
    g: list[]
        A sequence defining the matrix for negative indices. This
        will define the number of columns.

    Returns
    -------
    list[list]
        A Toeplitz matrix

    """
    rows, cols = len(h), len(g)
    T = mat_new(rows, cols)
    for col in range(cols):
        for row in range(rows):
            i = row - col
            T[row][col] = h[i] if i >= 0 else g[-i]
    return T


def mat_toeplitz_1d(h, x):
    """
    Constructs a Toeplitz matrix for 1D convolutions

    Parameters
    ----------
    h: list[]
        The filter's vector
    x: list[]
        The signal's vector

    Returns
    -------
    list[list]
        A Toeplitz matrix such that y = T(h) * x

    """
    Nh, Nx = len(h), len(x)
    Ny = Nx + Nh - 1
    Trows, Tcols = Ny, Nx
    T = mat_new(Trows, Tcols)
    for i, row in enumerate(T):
        Ts = max(i - Nh + 1, 0)
        Te = min(i + 1, Tcols)
        bs = min(i, Nh-1)
        be = i - Tcols if i >= Tcols else None
        row[Ts: Te] = h[bs: be: -1]
    return T


def mat_toeplitz_2d(h, x):
    """
    Constructs a Toeplitz matrix for 2D convolutions

    Parameters
    ----------
    h: list[list]
        A matrix of scalar values representing the filter (or kernel)
    x: list[list]
        A matrix of scalar values representing the signal

    Returns
    -------
    list[list]
        A doubly block Toeplitz matrix such that y = T(h) * x

    """
    # Calculate the dimensions of the output y
    Nh, Mh = mat_dim(h)
    Nx, Mx = mat_dim(x)
    Ny, My = Nh + Nx - 1, Mh + Mx - 1

    # Pad the filter, if needed
    padn, padm = Ny - Nh, My - Mh
    #if padn or padm:
    #    hp = mat_pad(h, [padn, 0, 0, padm])

    # Dimension of a Toeplitz matrix
    Trows, Tcols = My, Mx
    # Create the Toeplitz matrices
    Tlist = []
    for row in reversed(h):
        t = mat_toeplitz_1d(row, x[0])
        Tlist.append(t)
    for _ in range(padn):
        Tlist.append(mat_new(Trows, Tcols))
    # Dimension of the block Toeplitz matrix (BTM)
    BTrows, BTcols = Ny, Nx
    # Dimension of the doubly block Toeplitz matrix (DBTM)
    DTrows, DTcols = BTrows * Trows, BTcols * Tcols
    # Construct the DBTM
    DBTM = mat_new(DTrows, DTcols)
    for col in range(Nx):
        for row in range(Ny):
            i = row - col
            offset = (row * Trows, col * Tcols)
            block = Tlist[i]
            mat_submat_copy(DBTM, block, offset)
    return DBTM
    # xvec = mat_flatten(x, (1, True))
    # yvec = mat_product(DBTM, xvec)
    #
    # for mat in Tlist:
    #     print(np.array(mat))
    # print(np.array(DBTM))
    # print(np.array(xvec))
    # print(np.array(yvec))


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


def vec_is_void(a):
    """
    Check whether a given vector is empty

    A vector is considered "void" if it is None or has no
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
    scalar, None
        The minimum value in the vector, or None if the vector is empty

    """
    if vec_is_void(a):
        return None

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
    scalar, None
        The maximum value in the vector, or None if the vector is empty

    """
    if vec_is_void(a):
        return None

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
    tuple, None
        A 2-tuple with the minimum and maximum value,
        or None if the vector is empty

    """
    if vec_is_void(a):
        return None

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


def vec_pad(a, p, val=0):
    """
    Pads a vector

    Parameters
    ----------
    a: list[]
        A vector of scalar values
    p: tuple
        A 2-tuple indicating the padding size to the left and right
        of the vector respectively
    val: scalar
        A constant value used for the padding

    Returns
    -------
    list[]
        A padded vector

    """
    if vec_is_void(a):
        return []
    if not p or len(p) < 2 or min(p) < 0:
        raise ValueError(
            "Invalid padding size. Must be a 2-tuple of int >= 0"
        )

    left, right = p
    vp = vec_new(len(a) + left + right, val)
    vp[left: left + len(a)] = a
    return vp


def vec_to(newtype, a):
    """
    Converts the elements of a vector to a specified scalar type

    Parameters
    ----------
    newtype: class
        The new scalar type to which the elements of the vector
        will be converted.
    a: list[]
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


# ---------------------------------------------------------
#                     LINEAR OPERATIONS
# ---------------------------------------------------------


def conv2d(h, x):

    ax = len(h[0]) // 2
    ay = len(h) // 2
    N, M = len(x), len(x[0])
    J, I = len(h), len(h[0])
    y = mat_new(N, M)

    def inrange(pt):
        return (0 <= pt[0] < N) and (0 <= pt[1] < M)

    for n in range(N):
        for m in range(M):
            for j in range(J):
                for i in range(I):
                    p = (n + ay - j, m + ax - i)
                    if inrange(p):
                        y[n][m] += x[p[0]][p[1]] * h[j][i]
    return y


def conv2d_mat(h, x):
    #import numpy as np
    Nh, Mh = mat_dim(h)
    Nx, Mx = mat_dim(x)
    Ny, My = Nh + Nx - 1, Mh + Mx - 1

    toep = mat_toeplitz_2d(h, x)
    xvec = mat_flatten(x, (1, True))
    yvec = mat_product(toep, xvec)
    y = mat_unflatten(yvec, (Ny, My))

    # print(np.array(toep))
    # print(np.array(xvec))
    # print(np.array(yvec))
    # print(np.array(y))

    return y

