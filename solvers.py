import low_level_tools as llt
import numpy as np
import multigrid_utilities as mu
from scipy.linalg import solve_banded
import copy

def solve_pentadiagonal(coeff1, coeff2, bc_left, bc_right, rhs):
    """
    This function solve two boundary value problems in 1D
    a1 u_{xx} + b1 u_{x} + c1 u + d1 v_{xx} + e1 v_{x} + f1 v = r1
    d2 u_{xx} + e2 u_{x} + f2 u + a2 v_{xx} + b2 v_{x} + c2 v = r2
    u(0) = alpha1, v(0) = alpha2
    u(1) = beta1, v(1) = beta2
    using the standard scipy solver `solve_banded`.

    Parameters
    ----------
    coeff1: ndarray
        [a1, b1, c1, a2, b2, c2]
    coeff2: ndarray
        [d1, e1, f1, d2, e2, f2]
    bc_left: ndarray
        [alpha1, alpha2]
    bc_right: ndarray
        [beta1, beta2]
    rhs: ndarray
        [r1, r2]

    Returns
    -------
    res: ndarray
        Solution of the boundary value problem [u, v]. Shape (2, N).
    """
    r1, r2 = rhs
    N = len(r1)
    h = 1/(N+1)
    R = np.zeros(2*N)
    R[::2] = r1
    R[1::2] = r2
    a1, b1, c1, a2, b2, c2 = coeff1
    d1, e1, f1, d2, e2, f2 = coeff2
    M_left = np.array([[-b1[0]/(2*h) + a1[0]/(h**2), -e1[0]/(2*h) + d1[0]/(h**2)],
                       [-e2[0]/(2*h) + d2[0]/(h**2), -b2[0]/(2*h) + a2[0]/(h**2)]])
    M_right = np.array([[b1[-1]/(2*h) + a1[-1]/(h**2), e1[-1]/(2*h) + d1[-1]/(h**2)],
                        [e2[-1]/(2*h) + d2[-1]/(h**2), b2[-1]/(2*h) + a2[-1]/(h**2)]])
    bc_left_correct = M_left @ bc_left
    bc_right_correct = M_right @ bc_right
    R[:2] -= bc_left_correct
    R[-2:] -= bc_right_correct
    ###
    diag = np.zeros(2*N)
    diag[::2] = -2*a1/(h**2) + c1
    diag[1::2] = -2*a2/(h**2) + c2
    ###
    dp1 = np.zeros(2*N)
    dp1[::2] = -2*d1/(h**2) + f1
    dp1[1::2] = e2/(2*h) + d2/(h**2)
    dp1 = np.roll(dp1, 1)
    dp1[:1] = 0
    ###
    dp2 = np.zeros(2*N)
    dp2[::2] = b1/(2*h) + a1/(h**2)
    dp2[1::2] = b2/(2*h) + a2/(h**2)
    dp2 = np.roll(dp2, 2)
    dp2[:2] = 0
    ###
    dp3 = np.zeros(2*N)
    dp3[::2] = e1/(2*h) + d1/(h**2)
    dp3[1::2] = 0
    dp3 = np.roll(dp3, 3)
    dp3[:3] = 0
    ###
    dn1 = np.zeros(2*N)
    dn1[::2] = -e1/(2*h) + d1/(h**2)
    dn1[1::2] = -2*d2/(h**2) + f2
    dn1 = np.roll(dn1, -1)
    dn1[-1:] = 0
    ###
    dn2 = np.zeros(2*N)
    dn2[::2] = -b1/(2*h) + a1/(h**2)
    dn2[1::2] = -b2/(2*h) + a2/(h**2)
    dn2 = np.roll(dn2, -2)
    dn2[-2:] = 0
    ###
    dn3 = np.zeros(2*N)
    dn3[::2] = 0
    dn3[1::2] = -e2/(2*h) + d2/(h**2)
    dn3 = np.roll(dn3, -3)
    dn3[-3:] = 0
    ###
    A = np.vstack((dp3, dp2, dp1, diag, dn1, dn2, dn3))
    solution = solve_banded((3, 3), A, R, overwrite_ab=True, overwrite_b=True, check_finite=False)
    return solution[::2], solution[1::2]

def solve_tridiagonal(coeff1, bc_left, bc_right, rhs):
    """
    This function solve the single boundary value problems in 1D
    a1 u_{xx} + b1 u_{x} + c1 u = r1
    u(0) = alpha1
    u(1) = beta1
    using the standard scipy solver `solve_banded`.

    Parameters
    ----------
    coeff1: ndarray
        [a1, b1, c1]
    bc_left: double
        alpha1
    bc_right: double
        beta1
    rhs: ndarray
        r1

    Returns
    -------
    res: ndarray
        Solution of the boundary value problem u. Shape (N,).
    """
    R = np.copy(rhs)
    N = len(R)
    h = llt.get_h(N, '+')
    a1, b1, c1 = coeff1
    M_left = -b1[0]/(2*h) + a1[0]/(h**2)
    M_right = b1[-1]/(2*h) + a1[-1]/(h**2)
    bc_left_correct = M_left*bc_left
    bc_right_correct = M_right*bc_right
    R[:1] -= bc_left_correct
    R[-1:] -= bc_right_correct
    ###
    diag = np.zeros(N)
    diag = -2*a1/(h**2) + c1
    ###
    dp1 = np.zeros(N)
    dp1 = b1/(2*h) + a1/(h**2)
    dp1 = np.roll(dp1, 1)
    dp1[:1] = 0
    ###
    dn1 = np.zeros(N)
    dn1 = -b1/(2*h) + a1/(h**2)
    dn1 = np.roll(dn1, -1)
    dn1[-1:] = 0
    ###
    A = np.vstack((dp1, diag, dn1))
    solution = solve_banded((1, 1), A, R, overwrite_ab=True, overwrite_b=True, check_finite=False)
    return solution

def xZGS_2d_coupled(current, coeff, rhs):
    '''
    This is the coupled smoother that collectively update variables along x. Coupled
    means that the pentadiagonal system is solved along each line. Equations to be
    solved have the form

    a11 u_{yy} + b11 u_{xx} + c11 u_{xy} + d11 u_{x} + e11 u_{y} + f11 u +
    a12 v_{yy} + b12 v_{xx} + c12 v_{xy} + d12 v_{x} + e12 v_{y} + f12 v = r1

    a21 u_{yy} + b21 u_{xx} + c21 u_{xy} + d21 u_{x} + e21 u_{y} + f21 u +
    a22 v_{yy} + b22 v_{xx} + c22 v_{xy} + d22 v_{x} + e22 v_{y} + f22 v = r2

    Parameters
    ----------
    current: ndarray
        Current value of the solution. Should have shape (2, N, N) Second index is
        the coordinate in x direction and the third - in y direction. `current`
        should contain proper boundary values.
    coeff: array_like
        [[a11, ... ,f11, a12, ... ,f12], [a21, ... ,f21, a22, ... ,f22]
    rhs: ndarray
        [r1, r2]

    Returns
    -------
    next_current: ndarray
        The solution after one iteration. The shape is the same as `current`.

    '''
    r1, r2 = rhs
    coeff1, coeff2 = coeff
    a11, b11, c11, d11, e11, f11, a12, b12, c12, d12, e12, f12 = coeff1
    a21, b21, c21, d21, e21, f21, a22, b22, c22, d22, e22, f22 = coeff2
    u, v = np.copy(current)
    N, M = u.shape
    h = 1/(N-1)
    u_mod = llt.xRed_correction_2d(u, v, N, [a11, c11, e11, a12, c12, e12])
    v_mod = llt.xRed_correction_2d(v, u, N, [a22, c22, e22, a21, c21, e21])
    for i in range(int((N-1)/2)):
        i = 2*i + 1
        co1 = [b11[1:-1, i], d11[1:-1, i], f11[1:-1, i] - 2*a11[1:-1, i]/h**2, b22[1:-1, i], d22[1:-1, i], f22[1:-1, i] - 2*a22[1:-1, i]/h**2]
        co2 = [b12[1:-1, i], d12[1:-1, i], f12[1:-1, i] - 2*a12[1:-1, i]/h**2, b21[1:-1, i], d21[1:-1, i], f21[1:-1, i] - 2*a21[1:-1, i]/h**2]
        RHS = [r1[1:-1, i] - u_mod[1:-1, i], r2[1:-1, i] - v_mod[1:-1, i]]
        bc_left = [u[0, i], v[0, i]]
        bc_right = [u[-1, i], v[-1, i]]
        u[1:-1, i], v[1:-1, i] = solve_pentadiagonal(co1, co2, bc_left, bc_right, RHS)
    u_mod = llt.xBlack_correction_2d(u, v, N, [a11, c11, e11, a12, c12, e12])
    v_mod = llt.xBlack_correction_2d(v, u, N, [a22, c22, e22, a21, c21, e21])
    for i in range(int((N-1)/2)-1):
        i = 2*i + 2
        co1 = [b11[1:-1, i], d11[1:-1, i], f11[1:-1, i] - 2*a11[1:-1, i]/h**2, b22[1:-1, i], d22[1:-1, i], f22[1:-1, i] - 2*a22[1:-1, i]/h**2]
        co2 = [b12[1:-1, i], d12[1:-1, i], f12[1:-1, i] - 2*a12[1:-1, i]/h**2, b21[1:-1, i], d21[1:-1, i], f21[1:-1, i] - 2*a21[1:-1, i]/h**2]
        RHS = [r1[1:-1, i] - u_mod[1:-1, i], r2[1:-1, i] - v_mod[1:-1, i]]
        bc_left = [u[0, i], v[0, i]]
        bc_right = [u[-1, i], v[-1, i]]
        u[1:-1, i], v[1:-1, i] = solve_pentadiagonal(co1, co2, bc_left, bc_right, RHS)
    return np.array([u, v])

def yZGS_2d_coupled(current, coeff, rhs):
    '''
    This is the coupled smoother that collectively update variables along y. Coupled
    means that the pentadiagonal system is solved along each line. Equations to be
    solved have the form

    a11 u_{yy} + b11 u_{xx} + c11 u_{xy} + d11 u_{x} + e11 u_{y} + f11 u +
    a12 v_{yy} + b12 v_{xx} + c12 v_{xy} + d12 v_{x} + e12 v_{y} + f12 v = r1

    a21 u_{yy} + b21 u_{xx} + c21 u_{xy} + d21 u_{x} + e21 u_{y} + f21 u +
    a22 v_{yy} + b22 v_{xx} + c22 v_{xy} + d22 v_{x} + e22 v_{y} + f22 v = r2

    Parameters
    ----------
    current: ndarray
        Current value of the solution. Should have shape (2, N, N) Second index is
        the coordinate in x direction and the third - in y direction. `current`
        should contain proper boundary values.
    coeff: array_like
        [[a11, ... ,f11, a12, ... ,f12], [a21, ... ,f21, a22, ... ,f22]]
    rhs: ndarray
        [r1, r2]

    Returns
    -------
    current+1: ndarray
        The solution after one iteration. The shape is the same as `current`.

    '''
    r1, r2 = rhs
    coeff1, coeff2 = coeff
    a11, b11, c11, d11, e11, f11, a12, b12, c12, d12, e12, f12 = coeff1
    a21, b21, c21, d21, e21, f21, a22, b22, c22, d22, e22, f22 = coeff2
    u, v = np.copy(current)
    N, M = u.shape
    h = 1/(N-1)
    u_mod = llt.yRed_correction_2d(u, v, N, [b11, c11, d11, b12, c12, d12])
    v_mod = llt.yRed_correction_2d(v, u, N, [b22, c22, d22, b21, c21, d21])
    for i in range(int((N-1)/2)):
        i = 2*i + 1
        co1 = [a11[i, 1:-1], e11[i, 1:-1], f11[i, 1:-1] - 2*b11[i, 1:-1]/h**2, a22[i, 1:-1], e22[i, 1:-1], f22[i, 1:-1] - 2*b22[i, 1:-1]/h**2]
        co2 = [a12[i, 1:-1], e12[i, 1:-1], f12[i, 1:-1] - 2*b12[i, 1:-1]/h**2, a21[i, 1:-1], e21[i, 1:-1], f21[i, 1:-1] - 2*b21[i, 1:-1]/h**2]
        RHS = [r1[i, 1:-1] - u_mod[i, 1:-1], r2[i, 1:-1] - v_mod[i, 1:-1]]
        bc_left = [u[i, 0], v[i, 0]]
        bc_right = [u[i, -1], v[i, -1]]
        u[i, 1:-1], v[i, 1:-1] = solve_pentadiagonal(co1, co2, bc_left, bc_right, RHS)
    u_mod = llt.yBlack_correction_2d(u, v, N, [b11, c11, d11, b12, c12, d12])
    v_mod = llt.yBlack_correction_2d(v, u, N, [b22, c22, d22, b21, c21, d21])
    for i in range(int((N-1)/2)-1):
        i = 2*i + 2
        co1 = [a11[i, 1:-1], e11[i, 1:-1], f11[i, 1:-1] - 2*b11[i, 1:-1]/h**2, a22[i, 1:-1], e22[i, 1:-1], f22[i, 1:-1] - 2*b22[i, 1:-1]/h**2]
        co2 = [a12[i, 1:-1], e12[i, 1:-1], f12[i, 1:-1] - 2*b12[i, 1:-1]/h**2, a21[i, 1:-1], e21[i, 1:-1], f21[i, 1:-1] - 2*b21[i, 1:-1]/h**2]
        RHS = [r1[i, 1:-1] - u_mod[i, 1:-1], r2[i, 1:-1] - v_mod[i, 1:-1]]
        bc_left = [u[i, 0], v[i, 0]]
        bc_right = [u[i, -1], v[i, -1]]
        u[i, 1:-1], v[i, 1:-1] = solve_pentadiagonal(co1, co2, bc_left, bc_right, RHS)
    return np.array([u, v])

def xZGS_2d_coupled_boundary_correction(current, coeff, rhs):
    r1, r2 = rhs
    coeff1, coeff2 = coeff
    a11, b11, c11, d11, e11, f11, a12, b12, c12, d12, e12, f12 = coeff1
    a21, b21, c21, d21, e21, f21, a22, b22, c22, d22, e22, f22 = coeff2
    u, v = np.copy(current)
    N, M = u.shape
    h = 1/(N-1)
    u_mod = llt.xRed_boundary_correction_2d(u, v, N, [a11, c11, e11, a12, c12, e12])
    v_mod = llt.xRed_boundary_correction_2d(v, u, N, [a22, c22, e22, a21, c21, e21])
    i = 1
    co1 = [b11[1:-1, i], d11[1:-1, i], f11[1:-1, i] - 2*a11[1:-1, i]/h**2, b22[1:-1, i], d22[1:-1, i], f22[1:-1, i] - 2*a22[1:-1, i]/h**2]
    co2 = [b12[1:-1, i], d12[1:-1, i], f12[1:-1, i] - 2*a12[1:-1, i]/h**2, b21[1:-1, i], d21[1:-1, i], f21[1:-1, i] - 2*a21[1:-1, i]/h**2]
    RHS = [r1[1:-1, i] - u_mod[1:-1, i], r2[1:-1, i] - v_mod[1:-1, i]]
    bc_left = [u[0, i], v[0, i]]
    bc_right = [u[-1, i], v[-1, i]]
    u[1:-1, i], v[1:-1, i] = solve_pentadiagonal(co1, co2, bc_left, bc_right, RHS)
    i = -2
    co1 = [b11[1:-1, i], d11[1:-1, i], f11[1:-1, i] - 2*a11[1:-1, i]/h**2, b22[1:-1, i], d22[1:-1, i], f22[1:-1, i] - 2*a22[1:-1, i]/h**2]
    co2 = [b12[1:-1, i], d12[1:-1, i], f12[1:-1, i] - 2*a12[1:-1, i]/h**2, b21[1:-1, i], d21[1:-1, i], f21[1:-1, i] - 2*a21[1:-1, i]/h**2]
    RHS = [r1[1:-1, i] - u_mod[1:-1, i], r2[1:-1, i] - v_mod[1:-1, i]]
    bc_left = [u[0, i], v[0, i]]
    bc_right = [u[-1, i], v[-1, i]]
    u[1:-1, i], v[1:-1, i] = solve_pentadiagonal(co1, co2, bc_left, bc_right, RHS)
    return np.array([u, v])

def yZGS_2d_coupled_boundary_correction(current, coeff, rhs):
    '''
    This is the coupled smoother that collectively update variables along y. Coupled
    means that the pentadiagonal system is solved along each line. Equations to be
    solved have the form

    a11 u_{yy} + b11 u_{xx} + c11 u_{xy} + d11 u_{x} + e11 u_{y} + f11 u +
    a12 v_{yy} + b12 v_{xx} + c12 v_{xy} + d12 v_{x} + e12 v_{y} + f12 v = r1

    a21 u_{yy} + b21 u_{xx} + c21 u_{xy} + d21 u_{x} + e21 u_{y} + f21 u +
    a22 v_{yy} + b22 v_{xx} + c22 v_{xy} + d22 v_{x} + e22 v_{y} + f22 v = r2

    Parameters
    ----------
    current: ndarray
        Current value of the solution. Should have shape (2, N, N) Second index is
        the coordinate in x direction and the third - in y direction. `current`
        should contain proper boundary values.
    coeff: array_like
        [[a11, ... ,f11, a12, ... ,f12], [a21, ... ,f21, a22, ... ,f22]]
    rhs: ndarray
        [r1, r2]

    Returns
    -------
    current+1: ndarray
        The solution after one iteration. The shape is the same as `current`.

    '''
    r1, r2 = rhs
    coeff1, coeff2 = coeff
    a11, b11, c11, d11, e11, f11, a12, b12, c12, d12, e12, f12 = coeff1
    a21, b21, c21, d21, e21, f21, a22, b22, c22, d22, e22, f22 = coeff2
    u, v = np.copy(current)
    N, M = u.shape
    h = 1/(N-1)
    u_mod = llt.yRed_boundary_correction_2d(u, v, N, [b11, c11, d11, b12, c12, d12])
    v_mod = llt.yRed_boundary_correction_2d(v, u, N, [b22, c22, d22, b21, c21, d21])
    i = 1
    co1 = [a11[i, 1:-1], e11[i, 1:-1], f11[i, 1:-1] - 2*b11[i, 1:-1]/h**2, a22[i, 1:-1], e22[i, 1:-1], f22[i, 1:-1] - 2*b22[i, 1:-1]/h**2]
    co2 = [a12[i, 1:-1], e12[i, 1:-1], f12[i, 1:-1] - 2*b12[i, 1:-1]/h**2, a21[i, 1:-1], e21[i, 1:-1], f21[i, 1:-1] - 2*b21[i, 1:-1]/h**2]
    RHS = [r1[i, 1:-1] - u_mod[i, 1:-1], r2[i, 1:-1] - v_mod[i, 1:-1]]
    bc_left = [u[i, 0], v[i, 0]]
    bc_right = [u[i, -1], v[i, -1]]
    u[i, 1:-1], v[i, 1:-1] = solve_pentadiagonal(co1, co2, bc_left, bc_right, RHS)
    i = -2
    co1 = [a11[i, 1:-1], e11[i, 1:-1], f11[i, 1:-1] - 2*b11[i, 1:-1]/h**2, a22[i, 1:-1], e22[i, 1:-1], f22[i, 1:-1] - 2*b22[i, 1:-1]/h**2]
    co2 = [a12[i, 1:-1], e12[i, 1:-1], f12[i, 1:-1] - 2*b12[i, 1:-1]/h**2, a21[i, 1:-1], e21[i, 1:-1], f21[i, 1:-1] - 2*b21[i, 1:-1]/h**2]
    RHS = [r1[i, 1:-1] - u_mod[i, 1:-1], r2[i, 1:-1] - v_mod[i, 1:-1]]
    bc_left = [u[i, 0], v[i, 0]]
    bc_right = [u[i, -1], v[i, -1]]
    u[i, 1:-1], v[i, 1:-1] = solve_pentadiagonal(co1, co2, bc_left, bc_right, RHS)
    return np.array([u, v])

def aZGS_2d_coupled_boundary_correction(current, coeff, rhs):
    current = yZGS_2d_coupled(current, coeff, rhs)
    current = xZGS_2d_coupled(current, coeff, rhs)
    for i in range(2):
        current = yZGS_2d_coupled_boundary_correction(current, coeff, rhs)
        current = xZGS_2d_coupled_boundary_correction(current, coeff, rhs)
    return current

def aZGS_2d_coupled(current, coeff, rhs):
    '''
    This is the coupled smoother that collectively update variables along x and then
    along y. Coupled means that the pentadiagonal system is solved along each line.
    Equations to be solved have the form

    a11 u_{yy} + b11 u_{xx} + c11 u_{xy} + d11 u_{x} + e11 u_{y} + f11 u +
    a12 v_{yy} + b12 v_{xx} + c12 v_{xy} + d12 v_{x} + e12 v_{y} + f12 v = r1

    a21 u_{yy} + b21 u_{xx} + c21 u_{xy} + d21 u_{x} + e21 u_{y} + f21 u +
    a22 v_{yy} + b22 v_{xx} + c22 v_{xy} + d22 v_{x} + e22 v_{y} + f22 v = r2

    Parameters
    ----------
    current: ndarray
        Current value of the solution. Should have shape (2, N, N) Second index is
        the coordinate in x direction and the third - in y direction. `current`
        should contain proper boundary values.
    coeff: ndarray
        [[a11, ... ,f11, a12, ... ,f12], [a21, ... ,f21, a22, ... ,f22]]
    rhs: ndarray
        [r1, r2]

    Returns
    -------
    next_current: ndarray
        The solution after one iteration. The shape is the same as `current`.

    '''
    current = yZGS_2d_coupled(current, coeff, rhs)
    current = xZGS_2d_coupled(current, coeff, rhs)
    return current

def yZGS_1d(current, coeff, rhs):
    '''
    This is the GS smoother that collectively update variables along y.
    Tridiagonal system is solved each iteration.

    a11 u_{yy} + b11 u_{xx} + c11 u_{xy} + d11 u_{x} + e11 u_{y} + f11 u = r1

    Parameters
    ----------
    current: ndarray
        Current value of the solution. Should have shape (1, N, N) Second index is
        the coordinate in x direction and the third - in y direction. `current`
        should contain proper boundary values.
    coeff: ndarray
        [[a11, b11, c11, d11, e11, f11], ]
    rhs: ndarray
        [r1, ]

    Returns
    -------
    next_current: ndarray
        The solution after one iteration. The shape is the same as `current`.

    '''
    a, b, c, d, e, f = coeff[0]
    u = np.copy(current[0])
    r1 = rhs[0]
    N, M = u.shape
    h = 1/(N-1)
    u_mod = llt.yRed_correction_1d(u, N, [b, c, d])
    for i in range(int((N-1)/2)):
        i = 2*i + 1
        co1 = [a[i, 1:-1], e[i, 1:-1], f[i, 1:-1] - 2*b[i, 1:-1]/h**2]
        RHS = r1[i, 1:-1] - u_mod[i, 1:-1]
        bc_left = u[i, 0]
        bc_right = u[i, -1]
        u[i, 1:-1] = solve_tridiagonal(co1, bc_left, bc_right, RHS)
    u_mod = llt.yBlack_correction_1d(u, N, [b, c, d])
    for i in range(int((N-1)/2)-1):
        i = 2*i + 2
        co1 = [a[i, 1:-1], e[i, 1:-1], f[i, 1:-1] - 2*b[i, 1:-1]/h**2]
        RHS = r1[i, 1:-1] - u_mod[i, 1:-1]
        bc_left = u[i, 0]
        bc_right = u[i, -1]
        u[i, 1:-1] = solve_tridiagonal(co1, bc_left, bc_right, RHS)
    return np.array([u])

def xZGS_1d(current, coeff, rhs):
    '''
    This is the GS smoother that collectively update variables along x.
    Tridiagonal system is solved each iteration.

    a11 u_{yy} + b11 u_{xx} + c11 u_{xy} + d11 u_{x} + e11 u_{y} + f11 u = r1

    Parameters
    ----------
    current: ndarray
        Current value of the solution. Should have shape (1, N, N) Second index is
        the coordinate in x direction and the third - in y direction. `current`
        should contain proper boundary values.
    coeff: ndarray
        [[a11, b11, c11, d11, e11, f11], ]
    rhs: ndarray
        [r1, ]

    Returns
    -------
    next_current: ndarray
        The solution after one iteration. The shape is the same as `current`.

    '''
    a, b, c, d, e, f = coeff[0]
    u = np.copy(current[0])
    r1 = rhs[0]
    N, M = u.shape
    h = 1/(N-1)
    u_mod = llt.xRed_correction_1d(u, N, [a, c, e])
    for i in range(int((N-1)/2)):
        i = 2*i + 1
        co1 = [b[1:-1, i], d[1:-1, i], f[1:-1, i] - 2*a[1:-1, i]/h**2]
        RHS = r1[1:-1, i] - u_mod[1:-1, i]
        bc_left = u[0, i]
        bc_right = u[-1, i]
        u[1:-1, i] = solve_tridiagonal(co1, bc_left, bc_right, RHS)
    u_mod = llt.xBlack_correction_1d(u, N, [a, c, e])
    for i in range(int((N-1)/2)-1):
        i = 2*i + 2
        co1 = [b[1:-1, i], d[1:-1, i], f[1:-1, i] - 2*a[1:-1, i]/h**2]
        RHS = r1[1:-1, i] - u_mod[1:-1, i]
        bc_left = u[0, i]
        bc_right = u[-1, i]
        u[1:-1, i] = solve_tridiagonal(co1, bc_left, bc_right, RHS)
    return np.array([u])

def xGS_1d(current, coeff, rhs):
    a, b, c, d, e, f = coeff[0]
    u = np.copy(current[0])
    r1 = rhs[0]
    N, M = u.shape
    h = llt.get_h(N)
    for i in range(1, N-1):
        u_mod = llt.xGS_correction(u, N, [a, c, e], i)
        co1 = [b[1:-1, i], d[1:-1, i], f[1:-1, i] - 2*a[1:-1, i]/h**2]
        RHS = r1[1:-1, i] - u_mod[1:-1]
        bc_left = u[0, i]
        bc_right = u[-1, i]
        u[1:-1, i] = solve_tridiagonal(co1, bc_left, bc_right, RHS)
    # Back sweep
    for i in range(N-2, 0, -1):
        u_mod = llt.xGS_correction(u, N, [a, c, e], i)
        co1 = [b[1:-1, i], d[1:-1, i], f[1:-1, i] - 2*a[1:-1, i]/h**2]
        RHS = r1[1:-1, i] - u_mod[1:-1]
        bc_left = u[0, i]
        bc_right = u[-1, i]
        u[1:-1, i] = solve_tridiagonal(co1, bc_left, bc_right, RHS)
    return np.array([u])

def yGS_1d(current, coeff, rhs):
    a, b, c, d, e, f = coeff[0]
    u = np.copy(current[0])
    r1 = rhs[0]
    N, M = u.shape
    h = llt.get_h(N)
    for i in range(1, N-1):
        u_mod = llt.yGS_correction(u, N, [b, c, d], i)
        co1 = [a[i, 1:-1], e[i, 1:-1], f[i, 1:-1] - 2*b[i, 1:-1]/h**2]
        RHS = r1[i, 1:-1] - u_mod[1:-1]
        bc_left = u[i, 0]
        bc_right = u[i, -1]
        u[i, 1:-1] = solve_tridiagonal(co1, bc_left, bc_right, RHS)
    # Back sweep
    for i in range(N-2, 0, -1):
        u_mod = llt.yGS_correction(u, N, [b, c, d], i)
        co1 = [a[i, 1:-1], e[i, 1:-1], f[i, 1:-1] - 2*b[i, 1:-1]/h**2]
        RHS = r1[i, 1:-1] - u_mod[1:-1]
        bc_left = u[i, 0]
        bc_right = u[i, -1]
        u[i, 1:-1] = solve_tridiagonal(co1, bc_left, bc_right, RHS)
    return np.array([u])

def aGS_1d(current, coeff, rhs):
    current = xGS_1d(current, coeff, rhs)
    current = yGS_1d(current, coeff, rhs)
    return current

def decoupled_xGS_2d(current, coeff, rhs):
    r1, r2 = rhs
    coeff1, coeff2 = coeff
    co11 = coeff1[:6]
    co12 = coeff1[6:]
    co22 = coeff2[6:]
    co21 = coeff2[:6]
    u, v = current
    ###
    mod_r1 = r1 - mu.operator(np.array([v]), np.array([co12]))[0]
    u = xGS_1d([u], [co11], [mod_r1])[0]
    ###
    mod_r2 = r2 - mu.operator(np.array([u]), np.array([co21]))[0]
    v = xGS_1d([v], [co22], [mod_r2])[0]
    ###
    return np.array([u, v])

def decoupled_yGS_2d(current, coeff, rhs):
    r1, r2 = rhs
    coeff1, coeff2 = coeff
    co11 = coeff1[:6]
    co12 = coeff1[6:]
    co22 = coeff2[6:]
    co21 = coeff2[:6]
    u, v = current
    ###
    mod_r1 = r1 - mu.operator(np.array([v]), np.array([co12]))[0]
    u = yGS_1d([u], [co11], [mod_r1])[0]
    ###
    mod_r2 = r2 - mu.operator(np.array([u]), np.array([co21]))[0]
    v = yGS_1d([v], [co22], [mod_r2])[0]
    ###
    return np.array([u, v])

def decoupled_aGS_2d(current, coeff, rhs):
    current = decoupled_xGS_2d(current, coeff, rhs)
    current = decoupled_yGS_2d(current, coeff, rhs)
    return current

def aGS(current, coeff, rhs):
    M, N, K = current.shape
    if M == 1:
        current = aGS_1d(current, coeff, rhs)
    if M == 2:
        current = decoupled_aGS_2d(current, coeff, rhs)
    return current

def GS_1d(current, coeff, rhs):
    N, M = rhs[0].shape
    u = llt.xGS(current[0], rhs[0], N,  coeff[0])
    return np.array([u])

def decoupled_GS_2d(current, coeff, rhs):
    r1, r2 = rhs
    coeff1, coeff2 = coeff
    co11 = coeff1[:6]
    co12 = coeff1[6:]
    co22 = coeff2[6:]
    co21 = coeff2[:6]
    u, v = current
    ###
    mod_r1 = r1 - mu.operator(np.array([v]), np.array([co12]))[0]
    u = GS_1d([u], [co11], [mod_r1])[0]
    ###
    mod_r2 = r2 - mu.operator(np.array([u]), np.array([co21]))[0]
    v = GS_1d([v], [co22], [mod_r2])[0]
    ###
    return np.array([u, v])

def GS(current, coeff, rhs):
    M, N, K = current.shape
    if M == 1:
        current = GS_1d(current, coeff, rhs)
    if M == 2:
        current = decoupled_GS_2d(current, coeff, rhs)
    return current

def aZGS_1d(current, coeff, rhs):
    '''
    This is the GS smoother that collectively update variables along x and
    then along y. Tridiagonal system is solved each iteration.

    a11 u_{yy} + b11 u_{xx} + c11 u_{xy} + d11 u_{x} + e11 u_{y} + f11 u = r1

    Parameters
    ----------
    current: ndarray
        Current value of the solution. Should have shape (1, N, N) Second index is
        the coordinate in x direction and the third - in y direction. `current`
        should contain proper boundary values.
    coeff: ndarray
        [[a11, b11, c11, d11, e11, f11], ]
    rhs: ndarray
        [r1, ]

    Returns
    -------
    next_current: ndarray
        The solution after one iteration. The shape is the same as `current`.

    '''
    current = xZGS_1d(current, coeff, rhs)
    current = yZGS_1d(current, coeff, rhs)
    return current

def aZGS(current, coeff, rhs):
    '''
    This function either solve the system of equations

    a11 u_{yy} + b11 u_{xx} + c11 u_{xy} + d11 u_{x} + e11 u_{y} + f11 u +
    a12 v_{yy} + b12 v_{xx} + c12 v_{xy} + d12 v_{x} + e12 v_{y} + f12 v = r1

    a21 u_{yy} + b21 u_{xx} + c21 u_{xy} + d21 u_{x} + e21 u_{y} + f21 u +
    a22 v_{yy} + b22 v_{xx} + c22 v_{xy} + d22 v_{x} + e22 v_{y} + f22 v = r2

    using coupled smoother that collectively update variables along x and then
    along y. Coupled means that the pentadiagonal system is solved along each line.
    Equations to be solved have the form

    or the scalar equation

    a11 u_{yy} + b11 u_{xx} + c11 u_{xy} + d11 u_{x} + e11 u_{y} + f11 u = r1

    usin GS smoother that collectively update variables along x and
    then along y. Tridiagonal system is solved each iteration.

    Parameters
    ----------
    current: array_like
        Current value of the solution. Should have shape (2, N, N) or (1, N, N).
    coeff: array_like
        [[a11, ... ,f11, a12, ... ,f12], [a21, ... ,f21, a22, ... ,f22]] or
        [[a11, b11, c11, d11, e11, f11],]
    rhs: array_like
        [r1, r2] or [r1, ]

    Returns
    -------
    next_current: array_like
        The solution after one iteration. The shape is the same as `current`.

    '''
    M, N, K = current.shape
    if M == 1:
        current = aZGS_1d(current, coeff, rhs)
    if M == 2:
        current = aZGS_2d_coupled(current, coeff, rhs)
    return current

def aZGS_2d_decoupled(current, coeff, rhs):
    '''
    This is the decoupled smoother that collectively update variables along x and then
    along y. Deoupled means that the standard GS splitting is used on the level
    of equation variables i.e. for the first equation the second variable is fixed
    and for the second the first one is fixed.

    a11 u_{yy} + b11 u_{xx} + c11 u_{xy} + d11 u_{x} + e11 u_{y} + f11 u +
    a12 v_{yy} + b12 v_{xx} + c12 v_{xy} + d12 v_{x} + e12 v_{y} + f12 v = r1

    a21 u_{yy} + b21 u_{xx} + c21 u_{xy} + d21 u_{x} + e21 u_{y} + f21 u +
    a22 v_{yy} + b22 v_{xx} + c22 v_{xy} + d22 v_{x} + e22 v_{y} + f22 v = r2

    Parameters
    ----------
    current: ndarray
        Current value of the solution. Should have shape (2, N, N) Second index is
        the coordinate in x direction and the third - in y direction. `current`
        should contain proper boundary values.
    coeff: ndarray
        [[a11, ... ,f11, a12, ... ,f12], [a21, ... ,f21, a22, ... ,f22]]
    rhs: ndarray
        [r1, r2]

    Returns
    -------
    next_current: ndarray
        The solution after one iteration. The shape is the same as `current`.

    '''

    r1, r2 = rhs
    coeff1, coeff2 = coeff
    co11 = coeff1[:6]
    co12 = coeff1[6:]
    co22 = coeff2[6:]
    co21 = coeff2[:6]
    u, v = current
    ###
    mod_r1 = r1 - mu.operator(np.array([v]), np.array([co12]))[0]
    u = aZGS_1d([u], [co11], [mod_r1])[0]
    ###
    mod_r2 = r2 - mu.operator(np.array([u]), np.array([co21]))[0]
    v = aZGS_1d([v], [co22], [mod_r2])[0]
    ###
    return np.array([u, v])

class solver:
    def __init__(self, smoother, tol=None, verbose=False):
        self.smoother = smoother
        self.tol = tol
        self.verbose = verbose

    def sweep(self, equation, current, rhs, **kwargs):
        current = self.smoother(equation, current, rhs, **kwargs)
        return current

    # current, rhs,
    def detailed_solve(self, equation, current, rhs, **kwargs):
        if self.tol == None:
            J = int(np.log2(len(current[0])-1))
            tol = 2**(-2*J)
        else:
            tol = self.tol
        d = equation.rhs_defects(current, rhs)
        i = 0
        N = equation.n_equations
        if self.verbose:
            print('Tolerance is {:1.2}'.format(tol))
            print('Iteration', i)
            if N == 2:
                print('          Defect 1 = {:1.2}, Defect 2 = {:1.2}'.format(d[0], d[1]))
            if N == 1:
                print('          Defect = {:1.2}'.format(d[0]))
        if N == 1:
            while d[0]>tol:
                current = self.sweep(equation, current, rhs, **kwargs)
                i+=1
                d = equation.rhs_defects(current, rhs)
                if self.verbose:
                    print('Iteration', i)
                    print('          Defect = {:1.2}'.format(d[0]))
        if N == 2:
            while d[0]>tol or d[1]>tol:
                current = self.sweep(equation, current, rhs, **kwargs)
                i+=1
                d = equation.rhs_defects(current, rhs)
                if self.verbose:
                    print('Iteration', i)
                    print('          Defect 1 = {:1.2}, Defect 2 = {:1.2}'.format(d[0], d[1]))
        return current

    def solve(self, equation, J, **kwargs):
        current = equation.produce_initial_guess(J)
        rhs = equation.rhs(current)
        if self.tol == None:
            J = int(np.log2(len(current[0])-1))
            tol = 2**(-2*J)
        else:
            tol = self.tol
        d = equation.defects(current)
        i = 0
        N = equation.n_equations
        if self.verbose:
            print('Tolerance is {:1.2}'.format(tol))
            print('Iteration', i)
            if N == 2:
                print('          Defect 1 = {:1.2}, Defect 2 = {:1.2}'.format(d[0], d[1]))
            if N == 1:
                print('          Defect = {:1.2}'.format(d[0]))
        if N == 1:
            while d[0]>tol:
                current = self.sweep(equation, current, rhs, **kwargs)
                i+=1
                d = equation.defects(current)
                if self.verbose:
                    print('Iteration', i)
                    print('          Defect = {:1.2}'.format(d[0]))
        if N == 2:
            while d[0]>tol or d[1]>tol:
                current = self.sweep(equation, current, rhs, **kwargs)
                i+=1
                d = equation.defects(current)
                if self.verbose:
                    print('Iteration', i)
                    print('          Defect 1 = {:1.2}, Defect 2 = {:1.2}'.format(d[0], d[1]))
        return current

def FAS(equation, current, rhs, pre_smoother, pre_n, post_smoother, post_n, restriction, interpolation, coarse_solver, J_min):
    # Pre-smoothing
    for i in range(pre_n):
        current = pre_smoother(equation, current, rhs)
    # Extract a defect
    fine_defects = equation.rhs_residuals(current, rhs)
    # Restrict the defect
    coarse_defects = restriction(fine_defects)
    # Restrict the solution
    coarse_current = mu.injection(current)
    # Modify the defect
    coarse_defects = coarse_defects + equation.operator(coarse_current)
    # Solve error equation
    N = coarse_current.shape
    if N[1] == 2**J_min + 1:
        coarse_errors = coarse_solver.detailed_solve(equation, coarse_current, coarse_defects)
    else:
        args = (pre_smoother, pre_n, post_smoother, post_n, restriction, interpolation, coarse_solver, J_min)
        coarse_errors = FAS(equation, coarse_current, coarse_defects, *args)
        coarse_errors = FAS(equation, coarse_errors, coarse_defects, *args)
    # Modify an errror
    coarse_errors = coarse_errors - coarse_current
    # Interpolate an error
    fine_errors = interpolation(coarse_errors, 0*current)
    # Correct
    current = current + fine_errors
    # Post-smoothing
    for i in range(post_n):
        current = post_smoother(equation, current, rhs)
    return current

def exact_global_nonlinear_smoother(solver, equation, current, rhs):
    rhs1 = rhs - equation.operator(current)
    equation1 = copy.deepcopy(equation)
    equation1.coefficients = equation1.linear_coefficients
    equation1.bc = mu.zero_bc
    current = current + solver.detailed_solve(equation1, current*0, rhs1)
    return current

def global_nonlinear_smoother(smoother, equation, current, rhs):
    coeff = equation.linear_coefficients(current)
    rhs1 = rhs - equation.operator(current)
    current = current + smoother(0*current, coeff, rhs1)
    return current

def linear_smoother(smoother, equation, current, rhs):
    coeff = equation.coefficients(current)
    current = smoother(current, coeff, rhs)
    return current

def exact_linear_smoother(equation, current, rhs):
    D, M, N = current.shape
    M = N - 2
    J = int(np.log2(N-1))
    A, R = equation.dense_representation(J, bc='Special', rhs='Special', current=current, Rh=rhs)
    solution = np.linalg.inv(A) @ R
    if D == 2:
        u, v = np.copy(current)
        u[1:-1, 1:-1] = solution[:M**2].reshape((M, M)).T
        v[1:-1, 1:-1] = solution[M**2:].reshape((M, M)).T
        current = np.array([u, v])
    if D == 1:
        u = np.copy(current[0])
        u[1:-1, 1:-1] = solution.reshape((M, M)).T
        current = np.array([u,])
    return current

def nonlinear_aZGS(equation, current, rhs):
    return global_nonlinear_smoother(aZGS, equation, current, rhs)

def nonlinear_GS(equation, current, rhs):
    return global_nonlinear_smoother(GS, equation, current, rhs)

def linear_GS(equation, current, rhs):
    return linear_smoother(GS, equation, current, rhs)

def nonlinear_aGS(equation, current, rhs):
    return global_nonlinear_smoother(aGS, equation, current, rhs)

def linear_aGS(equation, current, rhs):
    return linear_smoother(aGS, equation, current, rhs)

def nonlinear_aZGS_with_bc_corrections(equation, current, rhs):
    return global_nonlinear_smoother(aZGS_2d_coupled_boundary_correction, equation, current, rhs)

def decoupled_nonlinear_aZGS(equation, current, rhs):
    return global_nonlinear_smoother(aZGS_2d_decoupled, equation, current, rhs)

def linear_aZGS(equation, current, rhs):
    return linear_smoother(aZGS, equation, current, rhs)

def decoupled_linear_aZGS(equation, current, rhs):
    return linear_smoother(aZGS_2d_decoupled, equation, current, rhs)

class FAS_solver(solver):
    def __init__(self, type, pre_smoother=None, pre_n=1, post_smoother=None, post_n=1,\
                        restriction=mu.linear_restriction, interpolation=mu.linear_interpolation,\
                                                        coarse_solver=None, J_min=2, tol=None, verbose=False):
        if type == 'nonlinear':
            if pre_smoother == None:
                self.pre_smoother = nonlinear_aZGS
            else:
                self.pre_smoother = pre_smoother
            if post_smoother == None:
                self.post_smoother = nonlinear_aZGS
            else:
                self.post_smoother = post_smoother
            if coarse_solver == None:
                self.coarse_solver = solver(nonlinear_aZGS, tol=1e-10)
            else:
                self.coarse_solver = coarse_solver

        if type == 'linear':
            if pre_smoother == None:
                self.pre_smoother = linear_aZGS
            else:
                self.pre_smoother = pre_smoother
            if post_smoother == None:
                self.post_smoother = linear_aZGS
            else:
                self.post_smoother = post_smoother
            if coarse_solver == None:
                self.coarse_solver = solver(linear_aZGS, tol=1e-10)
            else:
                self.coarse_solver = coarse_solver
        self.J_min = J_min
        self.pre_n = pre_n
        self.post_n = post_n
        self.restriction = restriction
        self.interpolation = interpolation
        self.tol = tol
        self.verbose = verbose
        params = (self.pre_smoother, self.pre_n, self.post_smoother, self.post_n, \
                            self.restriction, self.interpolation, self.coarse_solver, self.J_min)
        self.smoother = lambda equation, current, rhs: FAS(equation, current, rhs, *params)

    def refresh_parameters(self):
        params = (self.pre_smoother, self.pre_n, self.post_smoother, self.post_n, \
                            self.restriction, self.interpolation, self.coarse_solver, self.J_min)
        self.smoother = lambda equation, current, rhs: FAS(equation, current, rhs, *params)
