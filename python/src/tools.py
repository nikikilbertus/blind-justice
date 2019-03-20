"""
A collection of helpful functions for working with fixed point arithmetic.
"""

import collections
from functools import reduce

import numpy as np

from spfpm.FixedPoint import FXnum


# -----------------------------------------------------------------------------
# Conversions between floats and fixed point
# -----------------------------------------------------------------------------


def to_fixed(x, family):
    """Convert x to the format specified in family.

        If family is `None`, x is returned unchanged.

        Args:

            x: Typically a numpy.ndarray, could also be a numpy.generic,
                float, or int
            family: either `None` or of type `FXfamily`

        Returns:

            simply x (without any changes) if family is None

            either a primitive number of type family or a list (of lists)
            thereof, depending on the dimensions of x
    """
    if family is None:
        return x
    xtype = _type_of(x)
    # Scalar
    if xtype == 'scal':
        return FXnum(x, family=family)
    # Vector
    if xtype == 'vec':
        return [FXnum(a, family=family) for a in x]
    # Matrix
    else:
        return [[FXnum(a, family=family) for a in x_row] for x_row in x]


def to_float(x):
    """Convert x to a scalar, or numpy.ndarray."""
    if _np_instance(x):
        return x
    xtype = _type_of(x)
    if xtype == 'scal':
        return float(x)
    if xtype == 'vec':
        return np.array([float(a) for a in x], dtype=float)
    else:
        return np.array([[float(a) for a in x_row] for x_row in x],
                        dtype=float)


def stack_fp(x, y):
    """Stack two columns vectors together vertically."""
    if _np_instance(x, y):
        return np.hstack((x, y))

    optype = _operation_type(x, y)

    assert optype == 'vec_vec', "Cannot stack {} with {}".format(dimensions(x),
                                                                 dimensions(y))
    return x + y


# -----------------------------------------------------------------------------
# Piecewise linear function approximations (only for scalars)
# -----------------------------------------------------------------------------


def sigmoid_pw(x):
    """A piecewise linear version of sigmoid."""

    # Make use of sigmoid(-x) = 1 - sigmoid(x)
    positive = True
    if x < 0.:
        positive = False
        x = -x

    if x < 1.:
        y = 0.23105 * x + 0.50346
    elif x < 2.:
        y = 0.14973 * x + 0.58714
    elif x < 3.:
        y = 0.07177 * x + 0.74097
    elif x < 4.:
        y = 0.02943 * x + 0.86595
    elif x < 5.:
        y = 0.01129 * x + 0.93751
    else:
        y = 1.0

    return y if positive else 1. - y


def sigmoid_secureml(x):
    """A crude piecewise lienar approximation of sigmoid."""
    if x < -0.5:
        return 0.0
    elif x > 0.5:
        return 1.0
    else:
        return x + 0.5


def d_sigmoid_pw(x):
    """A piecewise linear version of the derivative of sigmoid."""

    # Make use of sigmoid'(x) = sigmoid'(-x)
    if x < 0.:
        return d_sigmoid_pw(-x)
    elif x < 0.5:
        return -0.02999 * x + 0.25175
    elif x < 1.:
        return -0.07678 * x + 0.27442
    elif x < 2.:
        return -0.09161 * x + 0.28729
    elif x < 3.:
        return -0.05981 * x + 0.22213
    elif x < 4.:
        return -0.02751 * x + 0.12623
    elif x < 5.:
        return -0.01101 * x + 0.06108
    else:
        return -0.00001 * x + 0.00608


# -----------------------------------------------------------------------------
# Vectorized versions of important functions
# -----------------------------------------------------------------------------


def _apply_vectorized(x, func):
    """Apply a function in a vectorized way."""
    xtype = _type_of(x)
    if xtype == 'scal':
        return func(x)
    if xtype == 'vec':
        return [func(a) for a in x]
    else:
        return [[func(a) for a in x_row] for x_row in x]


def inv_fp(x):
    """Element wise invert (i.e. 1 ./ x)."""
    if _np_instance(x):
        return 1 / x
    return _apply_vectorized(x, lambda z: 1/z)


def neg_fp(x):
    """Element wise negation."""
    if _np_instance(x):
        return -x
    return _apply_vectorized(x, lambda z: -z)


def div_pow2_fp(x, powof2):
    """Divide by a power of two."""
    if _np_instance(x):
        return x / np.power(2, powof2)
    return _apply_vectorized(x, lambda z: z >> powof2)


def mul_pow2_fp(x, powof2):
    """Multiply by a power of two."""
    if _np_instance(x):
        return x * np.power(2, powof2)
    return _apply_vectorized(x, lambda z: z << powof2)


def abs_fp(x):
    """Element wise absolute value with fallback to numpy.abs."""
    if _np_instance(x):
        return np.abs(x)
    return _apply_vectorized(x, lambda z: (z if z > 0. else -z))


def exp_fp(x):
    """Element wise exponential function with fallback to numpy.exp."""
    if _np_instance(x):
        return np.exp(x)
    return _apply_vectorized(x, lambda z: z.exp())


def log_fp(x):
    """Element wise natural logarithm with fallback to numpy.log."""
    if _np_instance(x):
        return np.log(x)
    return _apply_vectorized(x, lambda z: z.log())


def sigmoid(x, approx=None):
    """Element wise sigmoid (logistic function) with fallback to numpy."""
    is_numpy = False
    if _np_instance(x):
        if approx is None or approx in ['none', 'None']:
            return 1. / (1. + np.exp(-x))
        is_numpy = True

    if approx is None or approx in ['none', 'None']:
        def approx(y):
            return 1. / (1. + (-y).exp())
    elif isinstance(approx, str):
        if approx in ['pw', 'piecewise', 'lin', 'linear']:
            approx = sigmoid_pw
        elif approx in ['secureml', 'secure_ml', 'sml']:
            approx = sigmoid_secureml
    if not callable(approx):
        raise UserWarning('No callable found for {}'.format(approx))

    res = _apply_vectorized(x, approx)
    if is_numpy and not isinstance(res, (float, int)):
        return np.array(res)
    else:
        return res


def _apply_elem_wise_fp(x, y, operation):
    """Element wise apply operation to scalars, vectors or matrices.

    Args:
        x: first argument
        y: second argument

    Returns:
        the elemnt wise min of the two inputs

    """
    if _np_instance(x, y):
        if operation == 'min':
            return np.minimum(x, y)
        elif operation == 'max':
            return np.maximum(x, y)
        elif operation == '>':
            return (x > y).astype(np.float64)
        elif operation == '>=':
            return (x >= y).astype(np.float64)
        elif operation == '<':
            return (x < y).astype(np.float64)
        elif operation == '<=':
            return (x <= y).astype(np.float64)
        elif operation == '==':
            return (x == y).astype(np.float64)
        else:
            raise ValueError("Unknown operation {}".format(operation))

    # Get the operation type
    optype = _operation_type(x, y)

    # Define the operation to perform
    if operation == 'min':
        def func(a, b): return a if a < b else b
    elif operation == 'max':
        def func(a, b): return a if a > b else b
    elif operation == '>':
        def func(a, b): return float(a > b)
    elif operation == '>=':
        def func(a, b): return float(a >= b)
    elif operation == '<':
        def func(a, b): return float(a < b)
    elif operation == '<=':
        def func(a, b): return float(a <= b)
    elif operation == '==':
        def func(a, b): return float(a == b)
    else:
        raise ValueError("Unknown operation {}".format(operation))

    if optype == 'scal_scal':
        return func(x, y)
    elif optype in ['vec_scal', 'mat_scal']:
        return _apply_vectorized(x, lambda z: func(z, y))
    elif optype in ['scal_vec', 'scal_mat']:
        return _apply_vectorized(y, lambda z: func(x, z))
    elif optype == 'vec_vec':
        return [func(x[i], y[i]) for i in range(len(x))]
    elif optype in ['mat_mat_equal', 'mat_mat_all']:
        xr, xc = dimensions(x)
        return [[func(x[i][j], y[i][j]) for j in range(xc)]
                for i in range(xr)]
    else:
        raise ValueError("Cannot take min of {} and {}".format(dimensions(x),
                                                               dimensions(y)))


def minimum_fp(x, y):
    """Element wise minimum of scalars, vectors or matrices.

    Args:
        x: first argument
        y: second argument

    Returns:
        the elemen wise minimum
    """
    return _apply_elem_wise_fp(x, y, 'min')


def maximum_fp(x, y):
    """Element wise maximum of scalars, vectors or matrices.

    Args:
        x: first argument
        y: second argument

    Returns:
        the elemen wise maximum
    """
    return _apply_elem_wise_fp(x, y, 'max')


def grt_fp(x, y):
    """Element wise greater than of scalars, vectors or matrices.

    Args:
        x: first argument
        y: second argument

    Returns:
        an 0-1 array indicating where the comparison is true
    """
    return _apply_elem_wise_fp(x, y, '>')


def gre_fp(x, y):
    """Element wise greater or equal of scalars, vectors or matrices.

    Args:
        x: first argument
        y: second argument

    Returns:
        an 0-1 array indicating where the comparison is true
    """
    return _apply_elem_wise_fp(x, y, '>=')


def let_fp(x, y):
    """Element wise less than of scalars, vectors or matrices.

    Args:
        x: first argument
        y: second argument

    Returns:
        an 0-1 array indicating where the comparison is true
    """
    return _apply_elem_wise_fp(x, y, '<')


def lee_fp(x, y):
    """Element wise less or equal of scalars, vectors or matrices.

    Args:
        x: first argument
        y: second argument

    Returns:
        an 0-1 array indicating where the comparison is true
    """
    return _apply_elem_wise_fp(x, y, '<=')


def eq_fp(x, y):
    """Element wise equality check of scalars, vectors or matrices.

    Args:
        x: first argument
        y: second argument

    Returns:
        an 0-1 array indicating where the comparison is true
    """
    return _apply_elem_wise_fp(x, y, '==')

# -----------------------------------------------------------------------------
# High Level Vector and Matrix Functions (for use externally)
# -----------------------------------------------------------------------------


def invert_fp(x):
    """Matrix inversion.

    WARNING: In fixed point only works for matrices smaller or equal 2x2.
    WARNING: Only used in projected gradient.

    """
    if _np_instance(x):
        if isinstance(x, (float, int)):
            return 1 / x
        return np.linalg.inv(x)

    xtype = _type_of(x)
    if xtype == 'scal':
        return 1. / x

    assert xtype == 'mat', "Cannot invert {}".format(xtype)
    xr, xc = dimensions(x)
    assert xr == xc, "Cannot invert matrix with dimensions {}x{}".format(xr, xc)

    if xr == 1:
        return [[1./x[0][0]]]
    if xr == 2:
        det = x[0][0] * x[1][1] - x[0][1] * x[1][0]
        return dot_fp(1. / det, [[x[1][1], -x[1][0]], [-x[0][1], x[0][0]]])
    else:
        # TODO: invert matrix with LDL^T appraoch
        raise RuntimeError("Still need to implement inversion for n > 2")


def sum_fp(x, axis=0):
    """Sum along a specific axis.

    Args:
        x: Vector or matrix
        axis: The axis along which to sum

    Returns:
        scalar (for vector input) or vector (for matrix input)

    """
    if _np_instance(x):
        return np.sum(x, axis=axis)

    xtype = _type_of(x)
    if xtype == 'scal':
        return x
    elif xtype == 'vec':
        assert axis == 0, "Can't sum over axis={} for a vector".format(axis)
        return sum(x)
    else:
        assert axis in [0, 1], "Can't sum over axis={} for matrix".format(axis)
        if axis == 1:
            return [sum(xrow) for xrow in x]
        else:
            xr, xc = dimensions(x)
            return [sum([xrow[i] for xrow in x]) for i in range(xc)]


def add_fp(x, y):
    """Addition for consistent combintation of scalars, vectors and matrices.

    Warning: No broadcasting! There are special functions for that!

    Args:
        x: scalar, vector or matrix (fixed or float)
        y: scalar, vector or matrix (fixed or float)
    """
    if _np_instance(x, y):
        return x + y

    optype = _operation_type(x, y)

    if optype == 'scal_scal':
        return x + y
    elif optype == 'vec_vec':
        return _vec_vec_add_fp(x, y)
    elif optype in ['mat_mat_equal', 'mat_mat_all']:
        return _mat_mat_add_fp(x, y)
    elif optype in ['scal_vec', 'scal_mat']:
        return _scal_add_fp(y, x)
    elif optype in ['vec_scal', 'mat_scal']:
        return _scal_add_fp(x, y)
    else:
        raise ValueError("Addition not possible for {}".format(optype))


def sub_fp(x, y):
    """Subtraction for consistent combintation of scalars, vectors and matrices.

    Warning: No broadcasting! There are special functions for that!

    Args:
        x: scalar, vector or matrix (float or fixed), minuend
        y: scalar, vector or matrix (fixed point or float), subtrahend
    """
    if _np_instance(x, y):
        return x - y

    optype = _operation_type(x, y)

    if optype == 'scal_scal':
        return x - y
    elif optype == 'vec_vec':
        return _vec_vec_sub_fp(x, y)
    elif optype in ['mat_mat_equal', 'mat_mat_all']:
        return _mat_mat_sub_fp(x, y)
    elif optype in ['vec_scal', 'mat_scal']:
        return _scal_sub_fp(x, y)
    elif optype in ['scal_vec', 'scal_mat']:
        return _scal_matvec_sub_fp(x, y)
    else:
        raise ValueError("Subtraction not possible for {}".format(optype))


def dot_fp(x, y):
    """Dot products for consistent scalars, vectors, and matrices.

    Possible combinations for x, y:
        scal, scal
        scal, vec
        scal, mat
        vec, scal
        mat, scal
        vec, vec (same length)
        mat, vec (n_column of mat == length of vec)

    Warning: No broadcasting! There are special functions for that!

    Args:
        x: scalar, vector, or matrix (fixed point or float)
        y: scalar, vector or matrix (fixed point or float)
    """
    # If both inputs are np.ndarray we can use np.dot
    if _np_instance(x, y):
        return np.dot(x, y)

    optype = _operation_type(x, y)

    if optype == 'scal_scal':
        return x * y
    elif optype == 'vec_vec':
        return _vec_vec_dot_fp(x, y)
    elif optype in ['mat_mat_dot', 'mat_mat_all']:
        return _mat_mat_dot_fp(x, y)
    elif optype in ['mat_vec_dot', 'mat_vec_all']:
        return _mat_vec_dot_fp(x, y)
    elif optype in ['vec_scal', 'mat_scal']:
        return _scal_dot_fp(x, y)
    elif optype in ['scal_vec', 'scal_mat']:
        return _scal_dot_fp(y, x)
    else:
        raise ValueError("Dot not possible for {}".format(optype))


def elem_mul_fp(x, y, axis=0):
    """Compute the element wise product for scalars, vectors and matrices.

    Args:
        x: scalar, vector or matrix (fixed or float)
        y: scalar, vector or matrix (fixed or float)
        axis: if broadcasting is needed, along which axis to one broadcast

    Returns:
        element wise product
    """
    if _np_instance(x, y):
        if axis == 0:
            return x * y
        elif axis == 1:
            return x * y[:, np.newaxis]
        else:
            raise ValueError("Cannot interpret axis {}".format(axis))

    optype = _operation_type(x, y)

    if optype == 'scal_scal':
        return x * y
    elif optype == 'vec_vec':
        return _vec_vec_elem_mul_fp(x, y)
    elif optype in ['mat_mat_equal', 'mat_mat_all']:
        return _mat_mat_elem_mul_fp(x, y)
    elif optype == 'mat_vec_equal':
        if axis == 1:
            return _mat_vec_elem_mul_fp(x, y, axis)
        else:
            raise ValueError("Multiplication not possible for {} along axis {}"
                             .format(optype, axis))
    elif optype in ['mat_vec_dot']:
        if axis == 0:
            return _mat_vec_elem_mul_fp(x, y, axis)
        else:
            raise ValueError("Multiplication not possible for {} along axis {}"
                             .format(optype, axis))
    elif optype in ['vec_scal', 'mat_scal']:
        return _scal_dot_fp(x, y)
    elif optype in ['scal_vec', 'scal_mat']:
        return _scal_dot_fp(y, x)
    else:
        raise ValueError("Multiplication not possible for {}".format(optype))


def elem_div_fp(x, y, axis=0):
    """Compute the element wise division for scalars, vectors and matrices.

    Args:
        x: scalar, vector or matrix (fixed or float)
        y: scalar, vector or matrix (fixed or float)
        axis: if broadcasting is needed, along which axis to one broadcast

    Returns:
        element wise division
    """
    if _np_instance(x, y):
        if axis == 0:
            return x / y
        elif axis == 1:
            return x / y[:, np.newaxis]
        else:
            raise ValueError("Cannot interpret axis {}".format(axis))

    optype = _operation_type(x, y)

    if optype == 'scal_scal':
        return x / y
    elif optype == 'vec_vec':
        return _vec_vec_elem_div_fp(x, y)
    elif optype in ['mat_mat_equal', 'mat_mat_all']:
        return _mat_mat_elem_div_fp(x, y)
    elif optype == 'mat_vec_equal':
        if axis == 1:
            return _mat_vec_elem_div_fp(x, y, axis)
        else:
            raise ValueError("Division not possible for {} along axis {}"
                             .format(optype, axis))
    elif optype in ['mat_vec_dot']:
        if axis == 0:
            return _mat_vec_elem_div_fp(x, y, axis)
        else:
            raise ValueError("Division not possible for {} along axis {}"
                             .format(optype, axis))
    elif optype in ['vec_scal', 'mat_scal']:
        return _scal_div_fp(x, y)
    elif optype in ['scal_vec', 'scal_mat']:
        return _scal_div_fp(y, x)
    else:
        raise ValueError("Division not possible for {}".format(optype))


def mean_fp(x, powof2):
    """Compute the mean of a vector or matrix (along the 0 dimension).

    Use blocking with blocksize 2**powof2 in computing the mean to avoid
    overflow problems in fixed point arithmetic.

    Args:
        x: vector or matrix, the size of the dimension along which the mean is
           computed needs to be a power of 2
        powof2: the power of 2 to use in the blocksize

    Returns:
        the mean of x along axis 0

    """
    if _np_instance(x):
        return np.mean(x, 0)

    nblocks = 2**powof2
    xr, xc = dimensions(x)
    n = xr
    if n <= nblocks:
        return dot_fp(1. / n, reduce(add_fp, x))
    block_size = int(n / nblocks)
    assert block_size * nblocks == n, "Need equal blocks"
    return div_pow2_fp(reduce(add_fp,
                              map(lambda i: mean_fp(x[i:i + block_size],
                                                    powof2),
                                  range(0, n, block_size))), powof2)


def get_part_of(x, idx, axis=0, type='numbers'):
    """Return a new matrix/vector containing only the rows/columns of the input.

    Args:
        x: the input matrix/vector
        idx: list of indices which rows/columns to select
        axis: for matrix only, along which axis to select, i.e. rows or columns
        type: 'numbers' or 'active' whether idx is a list of actual index
            numbers or a binary array of the same size

    Returns:
        the subpart of the vector/matrix

    """
    if _np_instance(x):
        if axis == 0:
            return x[np.array(idx, dtype=int)]
        elif axis == 1:
            assert x.ndim == 2, "Cannot select along axis {} for ndims {}"\
                .format(axis, x.ndims)
            return x[:, np.array(idx, dtype=int)]
        else:
            raise ValueError("Axis parameter must be 0 or 1.")

    xtpye = _type_of(x)
    if xtpye == 'mat':
        if axis == 0:
            if type == 'numbers':
                return [xrow for i, xrow in enumerate(x) if i in idx]
            else:
                return [xrow for i, xrow in enumerate(x) if idx[i] == 1]
        elif axis == 1:
            if type == 'numbers':
                return [[a for i, a in enumerate(xrow) if i in idx]
                        for xrow in x]
            else:
                return [[a for i, a in enumerate(xrow) if idx[i] == 1]
                        for xrow in x]
        else:
            raise ValueError("Axis parameter must be 0 or 1.")
    elif xtpye == 'vec':
        if type == 'numbers':
            return [a for i, a in enumerate(x) if i in idx]
        else:
            return [a for i, a in enumerate(x) if idx[i] == 1]
    else:
        raise ValueError("Can only select subpart of vectors and matrices.")


def project_to(x, A, active):
    """Compute the projection of x on the active subspace of A.

    Args:
        x: the vector to be projected
        A: the matrix that determines the subspace on which to project
        active: a list of the active rows of A

    Returns:
        the projection of x onto the active parts of A
    """
    if _np_instance(A, x):
        family = None
    else:
        if isinstance(A[0][0], (float, int)):
            family = None
        else:
            family = A[0][0].family

    Ar, Ac = dimensions(A)
    I = to_fixed(np.eye(Ac), family=family)
    if active is None:
        Ahat = A
    else:
        Ahat = get_part_of(A, active, axis=0)
    AhatT = transpose_fp(Ahat)
    proj = dot_fp(AhatT, dot_fp(invert_fp(dot_fp(Ahat, AhatT)), Ahat))
    return dot_fp(sub_fp(I, proj), x)


def any_fp(x):
    """Any nonzero elements."""
    if len(x) == 0:
        return False

    if _np_instance(x):
        return np.any(x.ravel())

    xtype = _type_of(x)
    if xtype == 'scal':
        return x != 0
    elif xtype == 'vec':
        for a in x:
            if a != 0:
                return True
    elif xtype == 'mat':
        for xrow in x:
            for a in xrow:
                if a != 0:
                    return True
    return False


# -----------------------------------------------------------------------------
# Low Level Vector and Matrix Functions (for internal use only)
# -----------------------------------------------------------------------------


def _operation_type(x, y):
    xr, xc = dimensions(x)
    yr, yc = dimensions(y)
    xtype = _type_of(x)
    ytype = _type_of(y)

    if xtype == 'mat' and ytype == 'mat':
        if xc == yr:
            if xr == yr and xc == yc:
                return 'mat_mat_all'
            return 'mat_mat_dot'
        if xr == yr and xc == yc:
            return 'mat_mat_equal'
        else:
            return 'badop'

    if xtype == 'scal' and ytype == 'scal':
        return 'scal_scal'

    if xtype == 'vec' and ytype == 'vec':
        if xr == yr:
            return 'vec_vec'
        else:
            return 'badop'

    if ytype == 'scal':
        if xtype == 'vec':
            return 'vec_scal'
        if xtype == 'mat':
            return 'mat_scal'

    if xtype == 'scal':
        if ytype == 'vec':
            return 'scal_vec'
        if ytype == 'mat':
            return 'scal_mat'

    if xtype == 'mat' and ytype == 'vec':
        if xc == yr:
            if xr == yr:
                return 'mat_vec_all'
            return 'mat_vec_dot'
        if xr == yr:
            return 'mat_vec_equal'

    return 'badop'


def _type_of(x):
    xr, xc = dimensions(x)
    if xr == 0 and xc == 0:
        return 'scal'
    elif xr > 0 and xc == 0:
        return 'vec'
    else:
        return 'mat'


# -----------------------------------------------------------------------------
# Addition + Subtraction
# -----------------------------------------------------------------------------


def _scal_add_fp(x, scal):
    """Add a scalar scal to a vector or matrix x."""
    if _type_of(x) == 'vec':
        return [scal + a for a in x]
    else:
        return [[scal + a for a in x_row] for x_row in x]


def _scal_sub_fp(x, scal):
    """Subtract a scalar scal from a vector or matrix x."""
    if _type_of(x) == 'vec':
        return [a - scal for a in x]
    else:
        return [[a - scal for a in x_row] for x_row in x]


def _scal_matvec_sub_fp(scal, x):
    """Subtract a vector/matrix from a scalar (with broadcasting."""
    if _type_of(x) == 'vec':
        return [scal - a for a in x]
    else:
        return [[scal - a for a in x_row] for x_row in x]


def _vec_vec_add_fp(x, y):
    """Add two vectors."""
    return [a + b for a, b in zip(x, y)]


def _vec_vec_sub_fp(x, y):
    """Subtract two vectors."""
    return [a - b for a, b in zip(x, y)]


def _mat_mat_add_fp(x, y):
    """Add to matrices."""
    return [[a + b for a, b in zip(x_row, y_row)]
            for x_row, y_row in zip(x, y)]


def _mat_mat_sub_fp(x, y):
    """Subtract to matrices."""
    return [[a - b for a, b in zip(x_row, y_row)]
            for x_row, y_row in zip(x, y)]


# -----------------------------------------------------------------------------
# Dot products
# -----------------------------------------------------------------------------


def _vec_vec_dot_fp(x, y):
    """Inner product of two vectors (lists)."""
    return sum(a * b for a, b in zip(x, y))


def _mat_vec_dot_fp(x, y):
    """Matrix (list of list) times vector (list)."""
    return [sum(a * b for a, b in zip(row_x, y)) for row_x in x]


def blocked_mat_vec_dot_avg_fp(a, b):
    """Blocked matrix vector multiplication scaled by inverse of dimension.

    This function just computes a @ b / n where n is the number of columns in a
    (= number of elements in b) and @ is the normal matrix multiplication.
    It splits the multiplication in blocks, where the blocksize is determined
    by the precision used in the fixed point representation of a and b and n.

    Args:
        a: matrix (small_dim, large_dim)
        b: matrix (large_dim)

    Returns:
        a @ b / large_dim
    """
    if _np_instance(a, b):
        return a @ b / a.shape[1]

    optype = _operation_type(a, b)
    assert optype in ['mat_vec_dot', 'mat_vec_all'], \
        "Can't multiply {} and {}".format(dimensions(a), dimensions(b))
    n, m = dimensions(a)

    n_int = a[0][0].family.integer_bits
    n_frac = a[0][0].family.fraction_bits
    n_bits = min(n_int, n_frac)
    assert n_bits >= 4, "Can't work with less than 4 bits"

    # Check that inner dimension is power of 2
    assert is_pow2(m), "Need inner dimensions powof2, received {}".format(m)
    mpow2 = get_pow2(m)

    # Set block size roughly as sqrt(m)
    bpow2 = mpow2 >> 1

    # Check whether we have enough bits for block and sample size
    assert bpow2 < n_bits, "Not enough bits for block {}".format(2**bpow2)
    assert mpow2 - bpow2 < n_bits, "Need more bits for inner: {}, block {}" \
        .format(m, 2**bpow2)

    return [_blocked_vec_vec_dot_fp(arow, b, 2**bpow2, bpow2) >> (mpow2 - bpow2)
            for arow in a]


def _blocked_vec_vec_dot_fp(x, y, block, blockpow2):
    """Blocked dot product between two vectors with block averaging."""
    n = len(x)
    blocks = [_vec_vec_dot_fp(x[i:min(i + block, n)], y[i:min(i + block, n)])
              for i in range(0, n, block)]
    return sum_fp(div_pow2_fp(blocks, blockpow2))


def _mat_mat_dot_fp(x, y):
    """Matrix (list of lists) times matrix (list of lists)."""
    zip_y = list(zip(*y))
    return [[sum(a * b for a, b in zip(row_x, col_y))
             for col_y in zip_y] for row_x in x]


def blocked_mat_mat_dot_avg_fp(a, b):
    """Blocked matrix multiplication scaled by inverse of internal dimension.

    This function just computes a @ b / n where n is the number of columns in a
    (= number of rows in b) and @ is the normal matrix multiplication.
    It splits the multiplication in blocks, where the blocksize is determined
    by the precision used in the fixed point representation of a and b and n.

    Args:
        a: first matrix (small_dim, large_dim)
        b: second matrix (large_dim, small_dim)

    Returns:
        a * b / large_dim
    """
    if _np_instance(a, b):
        return a @ b / a.shape[1]

    optype = _operation_type(a, b)
    assert optype in ['mat_mat_all', 'mat_mat_dot'],\
        "Can't multiply {} and {}".format(dimensions(a), dimensions(b))
    n, m = dimensions(a)
    _, l = dimensions(b)

    n_int = a[0][0].family.integer_bits
    n_frac = a[0][0].family.fraction_bits
    n_bits = min(n_int, n_frac)
    assert n_bits >= 4, "Can't work with less than 4 bits"

    # Check that inner dimension is power of 2
    assert is_pow2(m), "Need inner dimensions powof2, received {}".format(m)
    mpow2 = get_pow2(m)

    # Set block size roughly as sqrt(m)
    bpow2 = mpow2 >> 1
    block = 2**bpow2

    # Check whether we have enough bits for block and sample size
    assert bpow2 < n_bits, "Not enough bits for block {}".format(2**bpow2)
    assert mpow2 - bpow2 < n_bits, "Need more bits for inner: {}, block {}" \
        .format(m, 2**bpow2)

    c = to_fixed(np.zeros((n, l)), a[0][0].family)
    for i0 in range(0, n, block):
        for j0 in range(0, l, block):
            for k0 in range(0, m, block):
                imax = min(i0 + block, n)
                jmax = min(j0 + block, l)

                # Matrix multiplication of a block
                for i in range(i0, imax):
                    for j in range(j0, jmax):
                        tmp = 0
                        for k in range(k0, k0 + block):
                            tmp += a[i][k] * b[k][j]
                        c[i][j] += (tmp >> bpow2)

    return div_pow2_fp(c, mpow2 - bpow2)


# -----------------------------------------------------------------------------
# Element wise operations
# -----------------------------------------------------------------------------


def _scal_dot_fp(x, scal):
    """Multiply a vector or matrix x by a scalar scal."""
    if _type_of(x) == 'vec':
        return [a * scal for a in x]
    else:
        return [[a * scal for a in x_row] for x_row in x]


def _scal_div_fp(x, scal):
    """Divide a vector or matrix x by a scalar scal."""
    if _type_of(x) == 'vec':
        return [a / scal for a in x]
    else:
        return [[a / scal for a in x_row] for x_row in x]


def _vec_vec_elem_mul_fp(x, y):
    """Multiply two vectors element wise."""
    return [a * b for a, b in zip(x, y)]


def _vec_vec_elem_div_fp(x, y):
    """Divide two vectors element wise."""
    return [a / b for a, b in zip(x, y)]


def _mat_mat_elem_mul_fp(x, y):
    """Multiply to matrices element wise."""
    return [[a * b for a, b in zip(x_row, y_row)]
            for x_row, y_row in zip(x, y)]


def _mat_mat_elem_div_fp(x, y):
    """Divide to matrices element wise."""
    return [[a / b for a, b in zip(x_row, y_row)]
            for x_row, y_row in zip(x, y)]


def _mat_vec_elem_mul_fp(x, y, axis):
    """Broadcast vector y to dimensions of matrix x for element wise mult."""
    if axis == 0:
        return [_vec_vec_elem_mul_fp(x_row, y) for x_row in x]
    elif axis == 1:
        return [_scal_dot_fp(x_row, y[i]) for i, x_row in enumerate(x)]
    else:
        raise ValueError("Unknown axis {}".format(axis))


def _mat_vec_elem_div_fp(x, y, axis):
    """Broadcast vector y to dimensions of matrix x for element wise divide."""
    if axis == 0:
        return [_vec_vec_elem_div_fp(x_row, y) for x_row in x]
    elif axis == 1:
        return [_scal_div_fp(x_row, y[i]) for i, x_row in enumerate(x)]
    else:
        raise ValueError("Unknown axis {}".format(axis))


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def is_pow2(x):
    """Check if input is a power of 2."""
    return not (x & (x - 1)) and x != 0


def get_pow2(x):
    """Hacky version to get the greatest power of two smaller than x."""
    return len(bin(x)) - 3


def get_iterable(x):
    """If not already iterable, turn into iterable."""
    if isinstance(x, collections.Iterable):
        return x
    else:
        return (x,)


def _np_instance(*args):
    """Check whether numpy functions can be used."""
    try:
        res = True
        for arg in args:
            res = res and isinstance(arg, (np.ndarray, np.generic, float, int))
        return res
    except NameError:
        return False


def dimensions(x):
    """Get the number of rows and columns of a (list of) list.

        Warning:
        This only works either for lists of primitive types or a list of equal
        length lists of primitive types and fails silently otherwise.

    Args:
        x: Input scalar/vector/matrix
    """
    if isinstance(x, (float, int)):
        return 0, 0
    try:
        if isinstance(x, np.ndarray):
            if x.ndim >= 2:
                return x.shape
            elif x.ndim == 1:
                return x.shape[0], 0
            elif x.ndim == 0:  # scalar numpy
                return 0, 0
    except NameError:
        pass

    if isinstance(x, list):
        xr = len(x)
        if isinstance(x[0], list):
            xc = len(x[0])
        else:
            xc = 0
    else:
        xr = 0
        xc = 0
    return xr, xc


def transpose_fp(x):
    """Transpose vector or matrix."""
    try:
        if isinstance(x, np.ndarray):
            return x.T
    except NameError:
        pass

    xtype = _type_of(x)
    if xtype == 'vec':
        return list(map(list, zip(x)))
    else:
        return list(map(list, zip(*x)))


def index_of_last_non_zero(x):
    """Index of last non-zero item."""
    for i, a in reversed(list(enumerate(x))):
        if a != 0:
            return i
    return -1


# -----------------------------------------------------------------------------
# Some primitive tests of the matrix tools above
# -----------------------------------------------------------------------------


def test_all_fp(verbose=False):
    """Thest the `dot_fp`, `add_fp` and `elem_mult_fp` functions."""
    check = np.testing.assert_array_equal

    scal = 2
    vec = [1, 2, 3]
    vec2 = [3, 4]
    mat = [[1, 2, 1], [1, 2, 2]]
    npscal = np.array(scal)
    npvec = np.array(vec)
    npvec2 = np.array(vec2)
    npmat = np.array(mat)

    r1 = dot_fp(scal, scal)
    r2 = dot_fp(npscal, npscal)
    check(r1, r2)

    r1 = dot_fp(scal, vec)
    r2 = dot_fp(npscal, npvec)
    check(r1, r2)

    r1 = dot_fp(vec, scal)
    r2 = dot_fp(npvec, npscal)
    check(r1, r2)

    r1 = dot_fp(scal, mat)
    r2 = dot_fp(npscal, npmat)
    check(r1, r2)

    r1 = dot_fp(mat, scal)
    r2 = dot_fp(npmat, npscal)
    check(r1, r2)

    r1 = dot_fp(vec, vec)
    r2 = dot_fp(npvec, npvec)
    check(r1, r2)

    r1 = dot_fp(mat, vec)
    r2 = dot_fp(npmat, npvec)
    check(r1, r2)

    r1 = dot_fp(mat, transpose_fp(mat))
    r2 = dot_fp(npmat, npmat.T)
    check(r1, r2)

    r1 = dot_fp(transpose_fp(mat), mat)
    r2 = dot_fp(npmat.T, npmat)
    check(r1, r2)

    r1 = elem_mul_fp(scal, vec)
    r2 = elem_mul_fp(npscal, npvec)
    check(r1, r2)

    r1 = elem_mul_fp(vec, scal)
    r2 = elem_mul_fp(npvec, npscal)
    check(r1, r2)

    r1 = elem_mul_fp(scal, mat)
    r2 = elem_mul_fp(npscal, npmat)
    check(r1, r2)

    r1 = elem_mul_fp(mat, scal)
    r2 = elem_mul_fp(npmat, npscal)
    check(r1, r2)

    r1 = elem_mul_fp(vec, vec)
    r2 = elem_mul_fp(npvec, npvec)
    check(r1, r2)

    r1 = elem_mul_fp(mat, mat)
    r2 = elem_mul_fp(npmat, npmat)
    check(r1, r2)

    r1 = elem_mul_fp(mat, vec, axis=0)
    r2 = elem_mul_fp(npmat, npvec, axis=0)
    check(r1, r2)

    r1 = elem_mul_fp(mat, vec2, axis=1)
    r2 = elem_mul_fp(npmat, npvec2, axis=1)
    check(r1, r2)

    r1 = elem_div_fp(vec, scal)
    r2 = elem_div_fp(npvec, npscal)
    check(r1, r2)

    r1 = elem_div_fp(mat, scal)
    r2 = elem_div_fp(npmat, npscal)
    check(r1, r2)

    r1 = elem_div_fp(vec, vec)
    r2 = elem_div_fp(npvec, npvec)
    check(r1, r2)

    r1 = elem_div_fp(mat, mat)
    r2 = elem_div_fp(npmat, npmat)
    check(r1, r2)

    r1 = elem_div_fp(mat, vec, axis=0)
    r2 = elem_div_fp(npmat, npvec, axis=0)
    check(r1, r2)

    r1 = elem_div_fp(mat, vec2, axis=1)
    r2 = elem_div_fp(npmat, npvec2, axis=1)
    check(r1, r2)

    print("All multiplication and division tests passed!")

    if verbose:
        print("FP: scal x vec")
        print(dot_fp(scal, vec))
        print("numpy: scal x vec")
        print(dot_fp(npscal, npvec), end='\n\n')

        print("FP: scal x mat")
        print(dot_fp(scal, mat))
        print("numpy: scal x mat")
        print(dot_fp(np.array(scal), np.array(mat)), end='\n\n')

        print("FP: vec x vec")
        print(dot_fp(vec, vec))
        print("numpy: vec x vec")
        print(dot_fp(np.array(vec), np.array(vec)), end='\n\n')

        print("FP: mat x vec")
        print(dot_fp(mat, vec))
        print("numpy: mat x vec")
        print(dot_fp(np.array(mat), np.array(vec)), end='\n\n')

        print("FP: mat x mat.T")
        print(dot_fp(mat, transpose_fp(mat)))
        print("numpy: mat x mat.T")
        print(dot_fp(np.array(mat), np.array(mat).T), end='\n\n')

        print("FP: mat.T x mat")
        print(dot_fp(transpose_fp(mat), mat))
        print("numpy: mat.T x mat")
        print(dot_fp(np.array(mat).T, np.array(mat)))

    r1 = add_fp(scal, scal)
    r2 = add_fp(npscal, npscal)
    check(r1, r2)

    r1 = add_fp(vec, scal)
    r2 = add_fp(npvec, npscal)
    check(r1, r2)

    r1 = add_fp(scal, vec)
    r2 = add_fp(npscal, npvec)
    check(r1, r2)

    r1 = add_fp(mat, scal)
    r2 = add_fp(npmat, npscal)
    check(r1, r2)

    r1 = add_fp(scal, mat)
    r2 = add_fp(npscal, npmat)
    check(r1, r2)

    r1 = add_fp(vec, vec)
    r2 = add_fp(npvec, npvec)
    check(r1, r2)

    r1 = add_fp(mat, mat)
    r2 = add_fp(npmat, npmat)
    check(r1, r2)

    r1 = sub_fp(scal, scal)
    r2 = sub_fp(npscal, npscal)
    check(r1, r2)

    r1 = sub_fp(vec, scal)
    r2 = sub_fp(npvec, npscal)
    check(r1, r2)

    r1 = sub_fp(mat, scal)
    r2 = sub_fp(npmat, npscal)
    check(r1, r2)

    r1 = sub_fp(vec, vec)
    r2 = sub_fp(npvec, npvec)
    check(r1, r2)

    r1 = sub_fp(mat, mat)
    r2 = sub_fp(npmat, npmat)
    check(r1, r2)

    print("All addition and subtraction tests passed!")

    if verbose:
        print("FP: scal + vec")
        print(add_fp(scal, vec))
        print("numpy: scal + vec")
        print(add_fp(np.array(scal), np.array(vec)), end='\n\n')

        print("FP: scal + mat")
        print(add_fp(scal, mat))
        print("numpy: scal + mat")
        print(add_fp(np.array(scal), np.array(mat)), end='\n\n')

        print("FP: vec + vec")
        print(add_fp(vec, vec))
        print("numpy: vec + vec")
        print(add_fp(np.array(vec), np.array(vec)), end='\n\n')

        print("FP: mat + mat")
        print(add_fp(mat, mat))
        print("numpy: mat + mat")
        print(add_fp(np.array(mat), np.array(mat)))
