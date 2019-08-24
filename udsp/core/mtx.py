"""
This module defines functions and classes for matrix-vector
operations (linear algebra and other manipulations). It is meant
to be used only internally, not as a public interface (consider it
as "private").

Matrices and vectors are implemented using the built-in list class,
that is vectors are basically list[] and matrices list[list].

Elements of matrices and vectors are assumed to be scalar types
(i.e. int, float or complex), all of the same type.

Since lists are heterogeneous containers, there is no enforcement
for all the elements to be of the same type, and the functions
defined here do not perform any check. It is assumed that all
elements are of same type as the first element. Implementation with
the 'array' class would have been nice, but it lacks complex type
support and (surprisingly) performs worse than lists on math operations.

Creating a matrix or a vector should be done by using the functions
defined here. That is, if you're extending/modifying the library
code, do not do this

a = [[1 for m in range(ncols)] for n in range(nrows)]

instead, do this

a = mat_new(nrows, ncols, 1)

Use the matrix-vector API defined here for all operations on matrices
and vectors. This is because the current implementation with lists
may be replaced in future with something more performant and the
modifications needed to the code will be limited to this module if
using this API.

TODO: This module was born "procedural" but it should really
      be redesigned with an OOP approach to abstract away the
      matrix/vector distinction made by these functions.

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

    # Solution 1
    # dot = 0
    # for i in range(len(a)):
    #     dot += a[i] * b[i]
    # return dot

    # Solution 2
    # dot = 0
    # for ai, bi in zip(a, b):
    #     dot += ai * bi
    # return dot

    return sum(map(_op.mul, a, b))


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
    Add two matrices or a matrix and a scalar element-wise

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
    if type(b) is not list:
        b = [b] * len(a)
    return [*map(vec_add, a, b)]


def mat_sub(a, b):
    """
    Subtract two matrices or a matrix and a scalar element-wise

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
    if type(b) is not list:
        b = [b] * len(a)
    return [*map(vec_sub, a, b)]


def mat_mul(a, b):
    """
    Multiply two matrices or a matrix and a scalar element-wise

    Parameters
    ----------
    a: list[list]
        A matrix of scalar values
    b: list[list], scalar
        A matrix of scalar values or a scalar value.

    Returns
    -------
    list[list]
        A matrix product of a and b

    """
    if type(b) is not list:
        b = [b] * len(a)
    return [*map(vec_mul, a, b)]


def mat_div(a, b):
    """
    Divide two matrices or a matrix and a scalar element-wise

    Parameters
    ----------
    a: list[list]
        A matrix of scalar values
    b: list[list], scalar
        A matrix of scalar values or a scalar value.

    Returns
    -------
    list[list]
        A matrix quotient of a and b

    """
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


def mat_extend(m, ext, val=0, mode=None, **kwargs):
    """
    Extends a matrix and assigns a given value to the new elements

    Parameters
    ----------
    m: list[list]
        A matrix of scalar values
    ext: tuple
        A 4-tuple (top, bottom, left, right) indicating the extension
        sizes on the first (top, bottom) and second (left, right)
        dimension respectively
    val: scalar
        A constant value used for the new elements
    mode: str, None
        A string representing the mode in which the newly added
        elements will be evaluated. By default, if none is specified,
        the new elements are all given the specified constant value.
        If a mode is specified, it must be a string indicating one
        of the supported modes, and **kwargs will contain parameters
        relative to that specific mode.

        Currently supported modes:

        "range": Extend a range of definition where each element of the
                 matrix represents a 2D point in a 2D range [a,b]x[c,d].
                 The range is extended by continuing the progression
                 (supposed to be linear) forward and backward, i.e.
                 [..., a-2*ds, a-ds, a,..., b, b+ds, b+2*ds, ...] (and
                 similarly for [c,d]).

                 kwargs:  ds (int) - Step of the progression

        "mirror": Extend the matrix by mirroring it along both dimensions.

                  kwargs:

        "stretch": Extend the matrix by repeating the values at the
                   boundaries along both dimensions.

                   kwargs:

        "repeat": Extend the matrix by repeating it along both dimensions
                  as if it was periodic.

                  kwargs:

    kwargs:
        The arguments relative to the extension mode, if any (see 'mode'
        parameter)

    Returns
    -------
    list[list]
        The given matrix extended with the given value

    """
    if mat_is_void(m):
        return []
    if not ext or len(ext) < 4 or min(ext) < 0:
        raise ValueError(
            "Invalid extension size. Must be a 4-tuple of int >= 0"
        )

    m_rows, m_cols = mat_dim(m)
    top, bottom, left, right = ext
    me_rows = m_rows + top + bottom
    me_cols = m_cols + left + right

    me = mat_new(me_rows, me_cols, val)

    for n, row in enumerate(m):
        me[top + n][left: left + m_cols] = row

    # No mode specified, just return the default
    if mode is None:
        return me

    # Extend the values ranges of m
    if mode == "range":

        ds = kwargs["ds"]
        ns, ms = m[0][0][0], m[0][0][1]

        # Solution 1

        # ext_col = [ms + (j - left) * ds for j in range(me_cols)]
        # ext_row = [ns + (i - top) * ds for i in range(me_rows)]
        #
        # for n, row in enumerate(me):
        #     if top <= n < top + m_rows:
        #         for i in range(0, left):
        #             row[i] = (ext_row[n], ext_col[i])
        #         for i in range(left + m_cols, me_cols):
        #             row[i] = (ext_row[n], ext_col[i])
        #     else:
        #         for i in range(me_cols):
        #             row[i] = (ext_row[n], ext_col[i])

        # Solution 2
        for i in range(me_rows):
            ni = ns + (i - top) * ds
            for j in range(me_cols):
                me[i][j] = (ni, ms + (j - left) * ds)

    elif mode in {"mirror", "stretch", "repeat"}:

        stretch = mode == "stretch"
        mright = left + m_cols
        mtop = top + m_rows

        def ftop(x):
            ii = 0 if stretch else x % m_rows
            return top + ii

        def fbot(x):
            ii = 0 if stretch else x % m_rows
            return mtop - 1 - ii

        def fleft(x):
            ii = 0 if stretch else x % m_cols
            return left + ii

        def fright(x):
            ii = 0 if stretch else x % m_cols
            return mright - 1 - ii

        # Set the extension functions based on mode
        if mode in {"mirror", "stretch"}:
            ext_t, ext_b = ftop, fbot
            ext_l, ext_r = fleft, fright
        else:
            ext_t, ext_b = fbot, ftop
            ext_l, ext_r = fright, fleft

        # Extend top-bottom
        for i in range(top):
            n1 = top - 1 - i
            n2 = ext_t(i)
            me[n1][left: mright] = me[n2][left: mright]
        for i in range(bottom):
            n1 = mtop + i
            n2 = ext_b(i)
            me[n1][left: mright] = me[n2][left: mright]

        # Extend left-right
        for n in range(me_rows):
            for i in range(left):
                m1 = left - 1 - i
                m2 = ext_l(i)
                me[n][m1] = me[n][m2]
            for i in range(right):
                m1 = mright + i
                m2 = ext_r(i)
                me[n][m1] = me[n][m2]

    else:
        raise ValueError("Invalid extension mode.")
    return me


def mat_compose(mats, f):
    """
    Compose a sequence of matrices using the given mapping function

    Parameters
    ----------
    mats: list[list]
        A sequence of matrices with equal dimensions
    f: callable
        A function that performs the mapping f:mats->composite,
        where 'composite' is the resulting matrix. The function
        f(m1[i,j],...,mn[i,j]) is applied to all elements of the
        n matrices for each point.

    Returns
    -------
    list[list]
        The composite matrix

    """
    return [[*map(f, *t)] for t in zip(*mats)]

    # Solution 2
    # Note: in this case f() must be a vector function
    # return [*itertools.starmap(f, [*zip(*mats)])]


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


def mat_bin(m, nbins, f, init=0):
    """
    Accumulate elements of a matrix into bins

    Parameters
    ----------
    m: list[list]
        A matrix of scalar values
    nbins: int
        The number of bins (accumulators) that will be returned as a list.
    f: callable
        A function f(m)->[bo,...,bn] that maps the matrix into the "bin space".
        Accepts one argument (a matrix element) and returns a 2-tuple
        (i, val), where 'i' is the bin index to be incremented and 'val'
        the increment value. If no bin shall be incremented, set i to None.
    init: scalar, optional
        A scalar value used to initialize the bins. Defaults to 0.

    Returns
    -------
    list
        A list of bins with the accumulated values.

    """
    acc = [init] * nbins
    for row in m:
        for e in row:
            i, val = f(e)
            if i is not None:
                acc[i] += val
    return acc


def mat_round(m, mode):
    """
    Round the elements of a matrix using different methods

    Parameters
    ----------
    m: list[list]
        A matrix of scalar values
    mode: {"nearest", "up", "down"}
        The rounding mode, where "nearest" is the standard method,
        "up" is the ceiling method and "down" is the flooring method.

    Returns
    -------
    list[list]
        A matrix with the rounded elements

    """
    return [*map(vec_round, m, [mode] * len(m))]


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
        A Toeplitz matrix T such that y = T(h) * x

    """
    Nh, Nx = len(h), len(x)
    Ny = Nx + Nh - 1
    Trows, Tcols = Ny, Nx
    T = mat_new(Trows, Tcols)
    for i, row in enumerate(T):
        Ts = max(i - Nh + 1, 0)
        Te = min(i + 1, Tcols)
        bs = min(i, Nh - 1)
        be = i - Tcols if i >= Tcols else None
        row[Ts: Te] = h[bs: be: -1]
    return T


def mat_toeplitz_2d(h, x):
    """
    Constructs a Toeplitz matrix for 2D convolutions

    Parameters
    ----------
    h: list[list]
        A matrix of scalar values representing the filter
    x: list[list]
        A matrix of scalar values representing the signal

    Returns
    -------
    list[list]
        A doubly block Toeplitz matrix T such that y = T(h) * x

    """
    # Calculate the dimensions of the arrays
    Nh, Mh = mat_dim(h)
    Nx, Mx = mat_dim(x)
    Ny, My = Nh + Nx - 1, Mh + Mx - 1
    # Pad the filter, if needed
    padn, padm = Ny - Nh, My - Mh
    # Dimensions of a Toeplitz matrix
    Trows, Tcols = My, Mx
    # Dimension of the block Toeplitz matrix (BTM)
    BTrows, BTcols = Ny, Nx
    # Dimension of the doubly block Toeplitz matrix (DBTM)
    DTrows, DTcols = BTrows * Trows, BTcols * Tcols
    # Create the Toeplitz matrices
    Tlist = []
    for row in reversed(h):
        t = mat_toeplitz_1d(row, x[0])
        Tlist.append(t)
    # Padding the blocks, if needed
    Tlist += [None] * padn
    # Construct the DBTM
    DBTM = mat_new(DTrows, DTcols)
    for col in range(BTcols):
        for row in range(BTrows):
            i = row - col
            offset = (row * Trows, col * Tcols)
            block = Tlist[i]
            if block:
                mat_submat_copy(DBTM, block, offset)
    return DBTM


# ---------------------------------------------------------
#                        VECTORS
# ---------------------------------------------------------


def vec_new(elems, init=None):
    """
    Creates a new initialized vector

    Parameters
    ----------
    elems: int
        The number of elements
    init: scalar, callable, collections.Iterable, optional
        The initializer expression. It can be a scalar value, a callable
        object (function, lambda, etc.) that will be invoked for each
        element in the matrix, or an iterable. The callable object must
        have the signature f(n), where n is the index over the elements.

    Returns
    -------
    list[]
        An initialized vector

    """
    if callable(init):
        return [init(n) for n in range(elems)]
    elif _utl.isiterable(init):
        return [i for i in init]
    else:
        return [init or 0] * elems


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
    Add two vectors or a vector and a scalar element-wise

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
    Subtract two vectors or a vector and a scalar element-wise

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
    Multiply two vectors or a vector and a scalar element-wise

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
    Divide two vectors or a vector and a scalar element-wise

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


def vec_extend(v, ext, val=0, mode=None, **kwargs):
    """
    Extends a vector and assigns a given value to the new elements

    Parameters
    ----------
    v: list[]
        A vector of scalar values
    ext: tuple
        A 2-tuple indicating the extension size to the left and right
        of the vector respectively
    val: scalar
        A constant value used for the new elements
    mode: str, None
        A string representing the mode in which the newly added
        elements will be evaluated. By default, if none is specified,
        the new elements are all given the specified constant value.
        If a mode is specified, it must be a string indicating one
        of the supported modes, and **kwargs will contain parameters
        relative to that specific mode.

        Currently supported modes:

        "range": Extend a range of definition where each element of the
                 vector represents a 1D point in a 1D range [a,b].
                 The range is extended by continuing the progression
                 (supposed to be linear) forward and backward, i.e.
                 [..., a-2*ds, a-ds, a,..., b, b+ds, b+2*ds, ...].

                 kwargs:  ds (int) - Step of the progression

                 Examples:

                 ext=(2,3), ds=1: [2, 3, 4] => [0, 1, 2, 3, 4, 5, 6, 7]
                 ext=(1,2), ds=0.5: [5.5, 6, 6.5] => [5, 5.5, 6, 6.5, 7, 7.5]

        "mirror": Extend the vector by mirroring it backwards and forwards.

                  kwargs:

        "stretch": Extend the vector by repeating the first and last value
                   backwards and forwards.

                   kwargs:

        "repeat": Extend the vector by repeating it backwards and forwards
                  as if it was periodic.

                  kwargs:

    kwargs:
        The arguments relative to the extension mode, if any (see 'mode'
        parameter)

    Returns
    -------
    list[]
        A vector extension of the given vector

    """
    if vec_is_void(v):
        return []
    if not ext or len(ext) < 2 or min(ext) < 0:
        raise ValueError(
            "Invalid extension size. Must be a 2-tuple of int >= 0"
        )

    left, right = ext
    dim = len(v) + left + right
    ve = vec_new(dim, val)
    ve[left: left + len(v)] = v

    # No mode specified, just return the default
    if mode is None:
        return ve

    # Extend the values range of v
    if mode == "range":

        ds = kwargs["ds"]
        xs, xe = v[0], v[len(v) - 1]

        # Solution 1
        #
        # Extend backwards
        # for i in range(1, left + 1):
        #     ve[left - i] = xs - (i * ds)
        # # Extend forwards
        # for i in range(right):
        #     ve[dim - right + i] = xe + (i + 1) * ds

        # Extend backwards
        ve[:left] = [xs - (left - i) * ds
                     for i in range(left)]
        # Extend forwards
        ve[left + len(v):] = [xe + (i + 1) * ds
                              for i in range(right)]

    elif mode in {"mirror", "stretch", "repeat"}:

        stretch = mode == "stretch"
        N = len(v)

        def fleft(x):
            ii = 0 if stretch else x % N
            return left + ii

        def fright(x):
            ii = 0 if stretch else x % N
            return left + N - 1 - ii

        # Set the extension functions based on mode
        if mode in {"mirror", "stretch"}:
            ext_l, ext_r = fleft, fright
        else:
            ext_l, ext_r = fright, fleft

        # Extend backwards
        for i in range(left):
            n1 = left - 1 - i
            n2 = ext_l(i)
            ve[n1] = ve[n2]
        # Extend forwards
        for i in range(right):
            n1 = left + N + i
            n2 = ext_r(i)
            ve[n1] = ve[n2]

    else:
        raise ValueError("Invalid extension mode.")
    return ve


def vec_compose(vecs, f):
    """
    Compose a sequence of vectors using the given mapping function

    Parameters
    ----------
    vecs: list[]
        A sequence of vectors with equal dimensions
    f: callable
        A function that performs the mapping f:vecs->composite,
        where 'composite' is the resulting vector. The function
        f(v1[i],...,vn[i]) is applied to all elements of the
        n vectors for each point.

    Returns
    -------
    list[]
        The composite vector

    """
    return [*map(f, *vecs)]


def vec_to(newtype, v):
    """
    Converts the elements of a vector to a specified scalar type

    Parameters
    ----------
    newtype: class
        The new scalar type to which the elements of the vector
        will be converted.
    v: list[]
        A vector of scalar values

    Returns
    -------
    list[]
        A vector whose elements are converted to the specified type.
        If the vector is already of the given type, a copy is returned.

    """
    if type(v[0]) is not newtype:
        return [newtype(e) for e in v]
    else:
        return [e for e in v]


def vec_bin(v, nbins, f, init=0):
    """
    Accumulate elements of a vector into bins

    Parameters
    ----------
    v: list[]
        A vector of scalar values
    nbins: int
        The number of bins (accumulators) that will be returned as a list.
    f: callable
        A function f(v)->[bo,...,bn] that maps the vector into the "bin space".
        Accepts one argument (a vector element) and returns a 2-tuple
        (i, val), where 'i' is the bin index to be incremented and 'val'
        the increment value. If no bin shall be incremented, set i to None.
    init: scalar, optional
        A scalar value used to initialize the bins. Defaults to 0.

    Returns
    -------
    list
        A list of bins with the accumulated values.

    """
    acc = [init] * nbins
    for e in v:
        i, val = f(e)
        if i is not None:
            acc[i] += val
    return acc


def vec_round(v, mode):
    """
    Round the elements of a vector using different methods

    Parameters
    ----------
    v: list[]
        A vector of scalar values
    mode: {"nearest", "up", "down"}
        The rounding mode, where "nearest" is the standard method,
        "up" is the ceiling method and "down" is the flooring method.

    Returns
    -------
    list[]
        A vector with the rounded elements

    """
    if vec_is_void(v):
        return v
    if mode not in {"nearest", "up", "down"}:
        raise ValueError("Invalid rounding mode: %s" % mode)

    fround = _utl.get_round_f(v, mode)

    return [*map(fround, v)]


# ---------------------------------------------------------
#                     LINEAR OPERATIONS
# ---------------------------------------------------------


def conv2d(h, x):
    """
    Standard implementation of 2D convolution

    Parameters
    ----------
    h: list[list]
        A matrix of scalar values representing the filter
    x: list[list]
        A matrix of scalar values representing the signal

    Returns
    -------
    list[list]
        A matrix the same size as the input representing the
        (partial) convolution y = h * x

    """
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


def conv2d_mat(h, x, warning=True):
    """
    Performs a 2D convolution in matrix form

    Parameters
    ----------
    h: list[list]
        A matrix of scalar values representing the filter
    x: list[list]
        A matrix of scalar values representing the signal
    warning: bool

    Returns
    -------
    list[list]
        A matrix representing the (full) convolution y = h * x

    """
    # TODO: implement this with generators
    if warning:
        raise RuntimeWarning("\n\nWARNING: This method is currently not"
                             "optimized and for demonstration "
                             "purposes only. It can choke your computer "
                             "if used on big arrays (hundreds of columns "
                             "and/or rows). You can remove this warning "
                             "message by passing 'warning=False' and go "
                             "ahead (at your own risk).\n\n")
    Nh, Mh = mat_dim(h)
    Nx, Mx = mat_dim(x)
    Ny, My = Nh + Nx - 1, Mh + Mx - 1
    toep = mat_toeplitz_2d(h, x)
    xvec = mat_flatten(x, (1, True))
    yvec = mat_product(toep, xvec)
    y = mat_unflatten(yvec, (Ny, My), (1, True))
    return y

