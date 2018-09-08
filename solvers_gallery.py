import copy
import numpy as np
import equation_tools as et
import solver_tools as st
from FAS import FAS
from solver import solver

def exact_global_nonlinear_smoother(solver, equation, current, rhs):
    rhs1 = rhs - equation.operator(current)
    equation1 = copy.deepcopy(equation)
    equation1.coefficients = equation1.linear_coefficients
    equation1.bc = et.zero_bc
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
    return global_nonlinear_smoother(st.aZGS, equation, current, rhs)

def nonlinear_GS(equation, current, rhs):
    return global_nonlinear_smoother(st.GS, equation, current, rhs)

def linear_GS(equation, current, rhs):
    return linear_smoother(st.GS, equation, current, rhs)

def nonlinear_aGS(equation, current, rhs):
    return global_nonlinear_smoother(st.aGS, equation, current, rhs)

def linear_aGS(equation, current, rhs):
    return linear_smoother(st.aGS, equation, current, rhs)

def nonlinear_aZGS_with_bc_corrections(equation, current, rhs):
    return global_nonlinear_smoother(st.aZGS_2d_coupled_boundary_correction, equation, current, rhs)

def decoupled_nonlinear_aZGS(equation, current, rhs):
    return global_nonlinear_smoother(st.aZGS_2d_decoupled, equation, current, rhs)

def linear_aZGS(equation, current, rhs):
    return linear_smoother(st.aZGS, equation, current, rhs)

def decoupled_linear_aZGS(equation, current, rhs):
    return linear_smoother(st.aZGS_2d_decoupled, equation, current, rhs)

def get_FAS(type):
    global FAS_solver
    if type == 'linear':
        pre_smoother = linear_aZGS
        pre_n = post_n = 1
        post_smoother = linear_aZGS
        restriction = et.linear_restriction
        interpolation = et.linear_interpolation
        coarse_solver = solver(linear_aZGS, tol=1e-10)
        J_min = 2
        verbose = True
        tol = None
        FAS_solver = FAS(pre_smoother, pre_n, post_smoother, post_n,
                    restriction, interpolation, coarse_solver, J_min, tol, verbose)
    if type == 'nonlinear':
        pre_smoother = nonlinear_aGS
        pre_n = post_n = 1
        post_smoother = nonlinear_aGS
        restriction = et.linear_restriction
        interpolation = et.linear_interpolation
        coarse_solver = solver(nonlinear_aGS, tol=1e-10)
        J_min = 2
        verbose = True
        tol = None
        FAS_solver = FAS(pre_smoother, pre_n, post_smoother, post_n,
                    restriction, interpolation, coarse_solver, J_min, tol, verbose)
    return FAS_solver
