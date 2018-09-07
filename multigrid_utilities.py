import low_level_tools as llt
import numpy as np


def defects_2d(current, coeff, rhs):
    u, v = current
    N, M = v.shape
    coeff1, coeff2 = coeff
    a11, b11, c11, d11, e11, f11, a12, b12, c12, d12, e12, f12 = coeff1
    a21, b21, c21, d21, e21, f21, a22, b22, c22, d22, e22, f22 = coeff2
    r1, r2 = rhs
    defect1 = llt.PDE(u, N, [a11, b11, c11, d11, e11, f11]) + llt.PDE(v, N, [a12, b12, c12, d12, e12, f12]) - r1
    defect2 = llt.PDE(u, N, [a21, b21, c21, d21, e21, f21]) + llt.PDE(v, N, [a22, b22, c22, d22, e22, f22]) - r2
    d1 = np.linalg.norm(defect1[1:-1, 1:-1].reshape((-1,)), ord=np.inf)
    d2 = np.linalg.norm(defect2[1:-1, 1:-1].reshape((-1,)), ord=np.inf)
    return np.array([d1, d2])


def residuals_2d(current, coeff, rhs):
    u, v = current
    N, M = v.shape
    coeff1, coeff2 = coeff
    a11, b11, c11, d11, e11, f11, a12, b12, c12, d12, e12, f12 = coeff1
    a21, b21, c21, d21, e21, f21, a22, b22, c22, d22, e22, f22 = coeff2
    r1, r2 = rhs
    defect1 = r1 - llt.PDE(u, N, [a11, b11, c11, d11, e11, f11]) - llt.PDE(v, N, [a12, b12, c12, d12, e12, f12])
    defect2 = r2 - llt.PDE(u, N, [a21, b21, c21, d21, e21, f21]) - llt.PDE(v, N, [a22, b22, c22, d22, e22, f22])
    return np.array([defect1, defect2])


def errors_2d(current, exact):
    u, v = current
    u_exact, v_exact = exact
    error1 = u - u_exact
    error2 = v - v_exact
    e1 = np.linalg.norm(error1[1:-1, 1:-1].reshape((-1,)), ord=np.inf)
    e2 = np.linalg.norm(error2[1:-1, 1:-1].reshape((-1,)), ord=np.inf)
    return np.array([e1, e2])


def operator_2d(current, coeff):
    u, v = current
    N, M = v.shape
    coeff1, coeff2 = coeff
    a11, b11, c11, d11, e11, f11, a12, b12, c12, d12, e12, f12 = coeff1
    a21, b21, c21, d21, e21, f21, a22, b22, c22, d22, e22, f22 = coeff2
    defect1 = llt.PDE(u, N, [a11, b11, c11, d11, e11, f11]) + llt.PDE(v, N, [a12, b12, c12, d12, e12, f12])
    defect2 = llt.PDE(u, N, [a21, b21, c21, d21, e21, f21]) + llt.PDE(v, N, [a22, b22, c22, d22, e22, f22])
    return np.array([defect1, defect2])


def restrict_2d(current):
    u, v = current
    N, M = v.shape
    restricted_u = llt.bilinear_restriction(u, N)
    restricted_v = llt.bilinear_restriction(v, N)
    return np.array([restricted_u, restricted_v])


def injection_2d(current):
    u, v = current
    restricted_u = u[::2, ::2]
    restricted_v = v[::2, ::2]
    return np.array([restricted_u, restricted_v])


def inject_coeff_2d(coeff):
    coeff1, coeff2 = coeff
    a11, b11, c11, d11, e11, f11, a12, b12, c12, d12, e12, f12 = coeff1
    a21, b21, c21, d21, e21, f21, a22, b22, c22, d22, e22, f22 = coeff2
    cf1 = [a11[::2, ::2], b11[::2, ::2], c11[::2, ::2], d11[::2, ::2],
           e11[::2, ::2], f11[::2, ::2], a12[::2, ::2], b12[::2, ::2],
           c12[::2, ::2], d12[::2, ::2], e12[::2, ::2], f12[::2, ::2]]
    cf2 = [a21[::2, ::2], b21[::2, ::2], c21[::2, ::2], d21[::2, ::2], e21[::2, ::2], f21[::2, ::2], a22[::2, ::2], b22[::2, ::2],
           c22[::2, ::2], d22[::2, ::2], e22[::2, ::2], f22[::2, ::2]]
    return np.array([cf1, cf2])


def interpolate_2d(current, up_with_bc):
    u, v = current
    u0, v0 = up_with_bc
    u1, v1 = np.copy(u0), np.copy(v0)
    u1[1:-1, 1:-1] = 0
    v1[1:-1, 1:-1] = 0
    N, M = u.shape
    llt.bilinear_interpolation(u, u1, N)
    llt.bilinear_interpolation(v, v1, N)
    return np.array([u1, v1])


def cubic_interpolate_2d(current, up_with_bc):
    u, v = current
    u0, v0 = up_with_bc
    u1, v1 = np.copy(u0), np.copy(v0)
    u1[1:-1, 1:-1] = 0
    v1[1:-1, 1:-1] = 0
    N, M = u.shape
    llt.c_interpolation(u, u1, N)
    llt.c_interpolation(v, v1, N)
    return np.array([u1, v1])


def defects_1d(current, coeff, rhs):
    u = current[0]
    N, M = u.shape
    coeff1 = coeff[0]
    a11, b11, c11, d11, e11, f11 = coeff1
    r1 = rhs[0]
    defect1 = llt.PDE(u, N, [a11, b11, c11, d11, e11, f11]) - r1
    d1 = np.linalg.norm(defect1[1:-1, 1:-1].reshape((-1,)), ord=np.inf)
    return np.array([d1])


def residuals_1d(current, coeff, rhs):
    u = current[0]
    N, M = u.shape
    coeff1 = coeff[0]
    a11, b11, c11, d11, e11, f11 = coeff1
    r1 = rhs[0]
    defect1 = - llt.PDE(u, N, [a11, b11, c11, d11, e11, f11]) + r1
    return np.array([defect1])


def errors_1d(current, exact):
    u = current[0]
    u_exact = exact[0]
    error1 = u - u_exact
    e1 = np.linalg.norm(error1[1:-1, 1:-1].reshape((-1,)), ord=np.inf)
    return np.array([e1])


def operator_1d(current, coeff):
    u = current[0]
    N, M = u.shape
    coeff1 = coeff[0]
    a11, b11, c11, d11, e11, f11 = coeff1
    defect1 = llt.PDE(u, N, [a11, b11, c11, d11, e11, f11])
    return np.array([defect1])


def restrict_1d(current):
    u = current[0]
    N, M = u.shape
    restricted_u = llt.bilinear_restriction(u, N)
    return np.array([restricted_u])


def injection_1d(current):
    u = current[0]
    restricted_u = u[::2, ::2]
    return np.array([restricted_u])


def inject_coeff_1d(coeff):
    coeff1 = coeff[0]
    a11, b11, c11, d11, e11, f11 = coeff1
    cf1 = [a11[::2, ::2], b11[::2, ::2], c11[::2, ::2], d11[::2, ::2], e11[::2, ::2], f11[::2, ::2]]
    return np.array([cf1])


def interpolate_1d(current, up_with_bc):
    u = current[0]
    u0 = up_with_bc[0]
    u1 = np.copy(u0)
    u1[1:-1, 1:-1] = 0
    N, M = u.shape
    llt.bilinear_interpolation(u, u1, N)
    return np.array([u1])


def cubic_interpolate_1d(current, up_with_bc):
    u = current[0]
    u0 = up_with_bc[0]
    u1 = np.copy(u0)
    u1[1:-1, 1:-1] = 0
    N, M = u.shape
    llt.c_interpolation(u, u1, N)
    return np.array([u1])


def defects(current, coeff, rhs):
    """
    This function compute |b - Az| in the uniform norm for the linear equation.
    Scalar equation

    a u_{yy} + b u_{xx} + c u_{xy} + d u_{x} + e u_{y} + f u = r

    or the system of two equations

    a11 u_{yy} + b11 u_{xx} + c11 u_{xy} + d11 u_{x} + e11 u_{y} + f11 u +
    a12 v_{yy} + b12 v_{xx} + c12 v_{xy} + d12 v_{x} + e12 v_{y} + f12 v = r1

    a21 u_{yy} + b21 u_{xx} + c21 u_{xy} + d21 u_{x} + e21 u_{y} + f21 u +
    a22 v_{yy} + b22 v_{xx} + c22 v_{xy} + d22 v_{x} + e22 v_{y} + f22 v = r2

    Parameters
    ----------
    current: array_like
        Current value of z. Should have shape (1, N, N) for the scalar equation
        or (2, N, N) in case of the system of two equations. Second index is the
        coordinate in x direction and the third - in y direction.
    coeff: array_like
        For the scalar case [[a, b, c, d, e, f], ] or for the system case
        [[a11, ... ,f11, a12, ... ,f12], [a21, ... ,f21, a22, ... ,f22]]
    rhs: array_like
        Right hand side of the equation [r] or [r1, r2].
        Should have the same shape as `current`.

    Returns
    -------
    res: ndarray
        Uniform norm of the residual. Array of shape (1,) or (2,)

    """
    M, N, K = current.shape
    if M == 1:
        res = defects_1d(current, coeff, rhs)
    if M == 2:
        res = defects_2d(current, coeff, rhs)
    return res


def residuals(current, coeff, rhs):
    """
    This function compute r - Au for the linear equation. Scalar equation

    a u_{yy} + b u_{xx} + c u_{xy} + d u_{x} + e u_{y} + f u = r

    or the system of two equations

    a11 u_{yy} + b11 u_{xx} + c11 u_{xy} + d11 u_{x} + e11 u_{y} + f11 u +
    a12 v_{yy} + b12 v_{xx} + c12 v_{xy} + d12 v_{x} + e12 v_{y} + f12 v = r1

    a21 u_{yy} + b21 u_{xx} + c21 u_{xy} + d21 u_{x} + e21 u_{y} + f21 u +
    a22 v_{yy} + b22 v_{xx} + c22 v_{xy} + d22 v_{x} + e22 v_{y} + f22 v = r2

    Parameters
    ----------
    current: array_like
        Current value of z. Should have shape (1, N, N) for the scalar equation
        or (2, N, N) in case of the system of two equations. Second index is the
        coordinate in x direction and the third - in y direction.
    coeff: array_like
        For the scalar case [[a, b, c, d, e, f],] or for the system case
        [[a11, ... ,f11, a12, ... ,f12], [a21, ... ,f21, a22, ... ,f22]]
    rhs: array_like
        Right hand side of the equation. Should have the same shape as `current`.

    Returns
    -------
    res: ndarray
        Residual. Array of shape (1, N, N) or (2, N, N).

    """
    M, N, K = current.shape
    if M == 1:
        res = residuals_1d(current, coeff, rhs)
    if M == 2:
        res = residuals_2d(current, coeff, rhs)
    return res


def errors(current, exact):
    """
    This function compute |u - v| in the uniform norm for the linear equation.

    Parameters
    ----------
    current: array_like
        Current value of the solution. Should have shape (1, N, N) for the scalar
        equation or (2, N, N) in case of the system of two equations. Second index
        is the coordinate in x direction and the third - in y direction.
    exact: array_like
        Exact solution. Should have shape (1, N, N) for the scalar equation
        or (2, N, N) in case of the system of two equations. Second index is the
        coordinate in x direction and the third - in y direction.

    Returns
    -------
    res: ndarray
        Uniform norm of `current` - `exact`. Array of shape (1, ) or (2, ).

    """
    M, N, K = current.shape
    if M == 1:
        res = errors_1d(current, exact)
    if M == 2:
        res = errors_2d(current, exact)
    return res


def operator(current, coeff):
    """
    This function compute Az for the linear equation. Scalar equation

    a u_{yy} + b u_{xx} + c u_{xy} + d u_{x} + e u_{y} + f u = r

    or the system of two equations

    a11 u_{yy} + b11 u_{xx} + c11 u_{xy} + d11 u_{x} + e11 u_{y} + f11 u +
    a12 v_{yy} + b12 v_{xx} + c12 v_{xy} + d12 v_{x} + e12 v_{y} + f12 v = r1

    a21 u_{yy} + b21 u_{xx} + c21 u_{xy} + d21 u_{x} + e21 u_{y} + f21 u +
    a22 v_{yy} + b22 v_{xx} + c22 v_{xy} + d22 v_{x} + e22 v_{y} + f22 v = r2

    Parameters
    ----------
    current: array_like
        Current value of z. Should have shape (1, N, N) for the scalar equation
        or (2, N, N) in case of the system of two equations. Second index is the
        coordinate in x direction and the third - in y direction.
    coeff: array_like
        For the scalar case [[a, b, c, d, e, f],] or for the system case
        [[a11, ... ,f11, a12, ... ,f12], [a21, ... ,f21, a22, ... ,f22]]

    Returns
    -------
    res: ndarray
        Result of operator action on the given vector. Array of shape (1, N, N)
        or (2, N, N) in case of the system of two equations.

    """
    M, N, K = current.shape
    if M == 1:
        res = operator_1d(current, coeff)
    if M == 2:
        res = operator_2d(current, coeff)
    return res


def linear_restriction(current):
    '''
    This function compute full-weighted restriction - a transpose to the linear
    interpolation operator. The stencil along the x axis is [1/2, 1, 1/2] and
    for 2D the stencil is the tensor product of two 1D stencils. In order not to
    move the boundary only arrays of the shape `(2**J+1, 2**J+1)` should be used.

    Parameters
    ----------
    current: array_like
        Values for restriction. Should have shape (1, N, N) for the scalar equation
        or (2, N, N) in case of the system of two equations. Second index is the
        coordinate in x direction and the third - in y direction.

    Returns
    -------
    res: array_like
        Result of the bilinear restriction (1, (N+1)/2, (N+1)/2) or (2, (N+1)/2, (N+1)/2).
    '''

    M = current.shape
    if M[0] == 1:
        res = restrict_1d(current)
    if M[0] == 2:
        res = restrict_2d(current)
    return res


def injection(current):
    '''
    This function compute injection. The stencil along the x axis is [0, 1, 0] and
    for 2D the stencil is the tensor product of two 1D stencils. In order not to
    move the boundary only arrays of the shape `(2**J+1, 2**J+1)` should be used.

    Parameters
    ----------
    current: array_like
        Values for injection. Should have shape (1, N, N) for the scalar equation
        or (2, N, N) in case of the system of two equations. Second index is the
        coordinate in x direction and the third - in y direction.

    Returns
    -------
    res: array_like
        Result of the injection (1, (N+1)/2, (N+1)/2) or (2, (N+1)/2, (N+1)/2).
    '''
    M, N, K = current.shape
    if M == 1:
        res = injection_1d(current)
    if M == 2:
        res = injection_2d(current)
    return res


def inject_coeff(coeff):
    '''
    This function compute injection. The stencil along the x axis is [0, 1, 0] and
    for 2D the stencil is the tensor product of two 1D stencils. In order not to
    move the boundary only arrays of the shape `(2**J+1, 2**J+1)` should be used.

    Parameters
    ----------
    coeff: array_like
        The shape is supposed to be (1, 6, N, N) for the scalar equation
        or (2, 24, N, N) in case of the system of two equations. In case of
        the scalar equation the secon index corresponds to coefficients
        a u_{yy} + b u_{xx} + c u_{xy} + d u_{x} + e u_{y} + f u so
        `coeff` = [[a, b, c, d, e, f]]. In case of the system one has 12 coefficients
        for the first equation
        a11 u_{yy} + b11 u_{xx} + c11 u_{xy} + d11 u_{x} + e11 u_{y} + f11 u +
        a12 v_{yy} + b12 v_{xx} + c12 v_{xy} + d12 v_{x} + e12 v_{y} + f12 v
        and 12 coefficients for the second equation
        a21 u_{yy} + b21 u_{xx} + c21 u_{xy} + d21 u_{x} + e21 u_{y} + f21 u +
        a22 v_{yy} + b22 v_{xx} + c22 v_{xy} + d22 v_{x} + e22 v_{y} + f22 v
        and `coeff`= [[a11, ... ,f11, a12, ... ,f12], [a21, ... ,f21, a22, ... ,f22]

    Returns
    -------
    res: array_like
        Result of the injection (1, 6, (N+1)/2, (N+1)/2) or (2, 24, (N+1)/2, (N+1)/2).
    '''
    K = coeff.shape
    if K[0] == 1:
        res = inject_coeff_1d(coeff)
    if K[0] == 2:
        res = inject_coeff_2d(coeff)
    return res


def linear_interpolation(current, up_with_bc):
    '''
    This function compute bilinear interpolation. The stencil along the x axis is
    ]1/2, 1, 1/2[ and for 2D the stencil is the tensor product of two 1D stencils.
    In order not to move the boundary only arrays of the shape `(2**J+1, 2**J+1)`
    should be used.

    Parameters
    ----------
    current: array_like
        Values for interpolation. Should have shape (1, N, N) for the scalar equation
        or (2, N, N) in case of the system of two equations. Second index is the
        coordinate in x direction and the third - in y direction.
    up_with_bc: array_like
        Array of the same shape (1, 2N-1, 2N-1) or (2, 2N-1, 2N-1) but with correct
        boundary elements stored in the `up_with_bc[0, 0, :]`, `up_with_bc[0, -1, :]`,
        `up_with_bc[0, :, 0]`, `up_with_bc[0, :, -1]` and the same in the case of
        the system of equations.

    Returns
    -------
    res: array_like
        Result of the bilinear interpolation. Shape is the same as for `up_with_bc`.
    '''
    M, N, K = current.shape
    if M == 1:
        res = interpolate_1d(current, up_with_bc)
    if M == 2:
        res = interpolate_2d(current, up_with_bc)
    return res


def cubic_interpolation(current, up_with_bc):
    '''
    This function compute cubic interpolation. The stencil along the x axis is
    ]-1/16, 1/9, 1, 1/9, -1/16[ and for 2D the stencil is the tensor product of
    two 1D stencils. Near the right boundary the stencil is ]1/16, -5/15, 15/16, 1, 5/16[.
    In order not to move the boundary only arrays of the shape `(2**J+1, 2**J+1)`
    should be used.

    Parameters
    ----------
    current: array_like
        Values for interpolation. Should have shape (1, N, N) for the scalar equation
        or (2, N, N) in case of the system of two equations. Second index is the
        coordinate in x direction and the third - in y direction.
    up_with_bc: array_like
        Array of the same shape (1, 2N-1, 2N-1) or (2, 2N-1, 2N-1) but with correct
        boundary elements stored in the `up_with_bc[0, 0, :]`, `up_with_bc[0, -1, :]`,
        `up_with_bc[0, :, 0]`, `up_with_bc[0, :, -1]` and the same in the case of
        the system of equations.

    Returns
    -------
    res: array_like
        Result of the cubic interpolation. Shape is the same as for `up_with_bc`.
    '''
    M, N, K = current.shape
    if M == 1:
        res = cubic_interpolate_1d(current, up_with_bc)
    if M == 2:
        res = cubic_interpolate_2d(current, up_with_bc)
    return res


def zero_bc(current):
    current[:, :, 0] *= 0
    current[:, :, -1] *= 0
    current[:, 0, :] *= 0
    current[:, -1, :] *= 0
    return current


def trivial_bc(current):
    u, v = current
    N, M = u.shape
    z = np.linspace(0, 1, N)
    current[:, :, 0] = z
    current[:, :, -1] = z
    current[:, 0, :] = z
    current[:, -1, :] = z
    return current
