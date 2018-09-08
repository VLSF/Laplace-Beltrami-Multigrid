import numpy as np
from harmonic_equation import harmonic_equation
from equation import equation
import low_level_tools as llt

################################################################################

def eq_11_bc(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_exact = np.exp(x * y)
    current[0][0, :] = u_exact[0, :]
    current[0][-1, :] = u_exact[-1, :]
    current[0][:, 0] = u_exact[:, 0]
    current[0][:, -1] = u_exact[:, -1]
    return current


def eq_11_exact(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_exact = np.exp(x * y)
    return np.array([u_exact])


def eq_11_rhs(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_rhs = ((1 + np.exp(x * y)) * x ** 2 + (2 + np.cos(np.pi * x)) * y ** 2 + \
             (1 + x * y) * np.exp(-x * y) + y * np.exp(x) + x * np.exp(y) + np.sin(np.pi * x * y)) * np.exp(x * y)
    u_rhs[0, :] = 0
    u_rhs[N - 1, :] = 0
    u_rhs[:, N - 1] = 0
    u_rhs[:, 0] = 0
    rhs = np.array([u_rhs])
    return rhs


def eq_11_coeff(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    ###
    a11 = 1 + np.exp(x * y)
    b11 = 2 + np.cos(np.pi * x)
    c11 = np.exp(-x * y)
    d11 = np.exp(x)
    e11 = np.exp(y)
    f11 = np.sin(np.pi * x * y)
    ###
    coeff1 = [a11, b11, c11, d11, e11, f11]
    coeff = np.array([coeff1])
    return coeff


################################################################################

def eq_red_fox_bc(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_exact = np.exp(x * y)
    current[0][0, :] = u_exact[0, :]
    current[0][-1, :] = u_exact[-1, :]
    current[0][:, 0] = u_exact[:, 0]
    current[0][:, -1] = u_exact[:, -1]
    return current


def eq_red_fox_exact(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_exact = np.exp(x * y)
    return np.array([u_exact])


def eq_red_fox_rhs(current, a=1):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_rhs = (x**2 + y**2 + a*y)*np.exp(x*y)
    u_rhs[0, :] = 0
    u_rhs[N - 1, :] = 0
    u_rhs[:, N - 1] = 0
    u_rhs[:, 0] = 0
    rhs = np.array([u_rhs])
    return rhs


def eq_red_fox_coeff(current, a=1):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    ###
    a11 = np.ones((N, N))
    b11 = np.ones((N, N))
    c11 = np.zeros((N, N))
    d11 = a*np.ones((N, N))
    e11 = np.zeros((N, N))
    f11 = np.zeros((N, N))
    ###
    coeff1 = [a11, b11, c11, d11, e11, f11]
    coeff = np.array([coeff1])
    return coeff

################################################################################


def eq_00_bc(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_exact = np.sin(np.pi * x) * np.sin(np.pi * y) / 2
    current[0][0, :] = u_exact[0, :]
    current[0][-1, :] = u_exact[-1, :]
    current[0][:, 0] = u_exact[:, 0]
    current[0][:, -1] = u_exact[:, -1]
    return current


def eq_00_exact(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_exact = np.sin(np.pi * x) * np.sin(np.pi * y) / 2
    return np.array([u_exact])


def eq_00_rhs(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_rhs = -np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y) * (
            4 + y * np.cos(x * np.pi) + 4 + x * np.exp(-x * y)) / 2 + \
            np.pi ** 2 * np.cos(np.pi * x) * np.cos(np.pi * y) * np.exp(y * x) / 2 + \
            np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) * x * y ** 3 / 2 + \
            np.pi * np.sin(np.pi * x) * np.cos(np.pi * y) * (y + x ** 2 + 0.2) / 2 + \
            np.sinh(x + 3 * y) * np.sin(np.pi * x) * np.sin(np.pi * y) / 2
    u_rhs[0, :] = 0;
    u_rhs[N - 1, :] = 0;
    u_rhs[:, N - 1] = 0;
    u_rhs[:, 0] = 0
    rhs = np.array([u_rhs])
    return rhs


def eq_00_coeff(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    ###
    a11 = 4 + y * np.cos(x * np.pi)
    b11 = 4 + x * np.exp(-x * y)
    c11 = np.exp(y * x)
    d11 = x * y ** 3
    e11 = y + x ** 2 + 0.2
    f11 = np.sinh(x + 3 * y)
    ###
    coeff1 = [a11, b11, c11, d11, e11, f11]
    coeff = np.array([coeff1])
    return coeff


################################################################################

def eq_12_bc(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_exact = np.exp(x + y)
    current[0][0, :] = u_exact[0, :]
    current[0][-1, :] = u_exact[-1, :]
    current[0][:, 0] = u_exact[:, 0]
    current[0][:, -1] = u_exact[:, -1]
    return current


def eq_12_exact(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_exact = np.exp(x + y)
    return np.array([u_exact])


def eq_12_rhs(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_rhs = (4 + np.cos(2 * np.pi * x * y) + 2 + np.sin(np.pi * x * y) + np.exp(-x * y) \
             + np.exp(x) + np.exp(y) + np.sin(np.pi * x * y) + 2) * np.exp(x + y)
    u_rhs[0, :] = 0;
    u_rhs[N - 1, :] = 0;
    u_rhs[:, N - 1] = 0;
    u_rhs[:, 0] = 0
    rhs = np.array([u_rhs])
    return rhs


def eq_12_coeff(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    ###
    a11 = 4 + np.cos(2 * np.pi * x * y)
    b11 = 2 + np.sin(np.pi * x * y)
    c11 = np.exp(-x * y)
    d11 = np.exp(x)
    e11 = np.exp(y)
    f11 = np.sin(np.pi * x * y) + 2
    ###
    coeff1 = [a11, b11, c11, d11, e11, f11]
    coeff = np.array([coeff1])
    return coeff


################################################################################

def eq_13_bc(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_exact = y * np.exp(x)
    current[0][0, :] = u_exact[0, :]
    current[0][-1, :] = u_exact[-1, :]
    current[0][:, 0] = u_exact[:, 0]
    current[0][:, -1] = u_exact[:, -1]
    return current


def eq_13_exact(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_exact = y * np.exp(x)
    return np.array([u_exact])


def eq_13_rhs(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_rhs = (2 + x * np.exp(x * y) + 6 + np.sin(np.pi * x * y)) * y * np.exp(x) + \
            x * np.exp(-x * y) * np.exp(x) + y ** 2 * np.exp(2 * x) + x * y ** 2 * np.exp(x) * np.exp(y)
    u_rhs[0, :] = 0;
    u_rhs[N - 1, :] = 0;
    u_rhs[:, N - 1] = 0;
    u_rhs[:, 0] = 0
    rhs = np.array([u_rhs])
    return rhs


def eq_13_coeff(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    ###
    a11 = 4 + y * np.exp(-x * y)
    b11 = 2 + x * np.exp(x * y)
    c11 = x * np.exp(-x * y)
    d11 = y * np.exp(x)
    e11 = x * y ** 2 * np.exp(y)
    f11 = 6 + np.sin(np.pi * x * y)
    ###
    coeff1 = [a11, b11, c11, d11, e11, f11]
    coeff = np.array([coeff1])
    return coeff


################################################################################

def eq_14_bc(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_exact = np.exp(x + y)
    current[0][0, :] = u_exact[0, :]
    current[0][-1, :] = u_exact[-1, :]
    current[0][:, 0] = u_exact[:, 0]
    current[0][:, -1] = u_exact[:, -1]
    return current


def eq_14_exact(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_exact = np.exp(x + y)
    return np.array([u_exact])


def eq_14_rhs(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    b = 4
    a = 3
    u_rhs = (b + np.exp(x * y) + a + np.exp(-x * y) +
            np.cos(np.pi*(x + 2*y)) + np.sin(np.pi*(y + 2*x)))*np.exp(x + y)
    u_rhs[0, :] = 0;
    u_rhs[N - 1, :] = 0;
    u_rhs[:, N - 1] = 0;
    u_rhs[:, 0] = 0
    rhs = np.array([u_rhs])
    return rhs


def eq_14_coeff(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    ###
    b = 4
    a = 3
    a11 = b + np.exp(x * y)
    b11 = a + np.exp(-x * y)
    c11 = np.zeros((N, N))
    d11 = np.cos(np.pi*(x + 2*y))
    e11 = np.sin(np.pi*(y + 2*x))
    f11 = np.zeros((N, N))
    ###
    coeff1 = [a11, b11, c11, d11, e11, f11]
    coeff = np.array([coeff1])
    return coeff

################################################################################

def eq_21_bc(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_exact = np.exp(x * y)
    v_exact = np.exp(2 * x * y)
    ###
    current[0][0, :] = u_exact[0, :]
    current[0][-1, :] = u_exact[-1, :]
    current[0][:, 0] = u_exact[:, 0]
    current[0][:, -1] = u_exact[:, -1]
    ###
    current[1][0, :] = v_exact[0, :]
    current[1][-1, :] = v_exact[-1, :]
    current[1][:, 0] = v_exact[:, 0]
    current[1][:, -1] = v_exact[:, -1]
    ###
    return current


def eq_21_exact(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_exact = np.exp(x * y)
    v_exact = np.exp(2 * x * y)
    exact = np.array([u_exact, v_exact])
    return exact


def eq_21_rhs(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_rhs = 20 * np.exp(2 * x * y) * x ** 2 - x - np.exp(-x * y) * y
    v_rhs = np.exp(x * y) + 4 * (7 + (np.sin(np.pi * x * y)) ** 2) * np.exp(2 * x * y) * y ** 2 + 16 * np.exp(
        3 * x * y) * x ** 2 - \
            2 * x * np.exp(2 * x * y - x) - 2 * y * np.exp(2 * x * y - y) + (2 + 4 * x * y) * np.sin(
        np.pi * x * y) * np.exp(2 * x * y)
    v_rhs[0, :] = 0
    v_rhs[N - 1, :] = 0
    v_rhs[:, N - 1] = 0
    v_rhs[:, 0] = 0
    u_rhs[0, :] = 0
    u_rhs[N - 1, :] = 0
    u_rhs[:, N - 1] = 0
    u_rhs[:, 0] = 0
    rhs = np.array([u_rhs, v_rhs])
    return rhs


def eq_21_coeff(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    ###
    a11 = 20 * np.exp(x * y)
    b11 = 7 + (np.cos(np.pi * x * y)) ** 2
    c11 = np.cos(np.pi * x * y)
    d11 = -np.exp(-2 * x * y)
    e11 = -np.exp(-x * y)
    f11 = np.zeros((N, N))
    ###
    a12 = np.zeros((N, N))
    b12 = np.zeros((N, N))
    c12 = np.zeros((N, N))
    d12 = np.zeros((N, N))
    e12 = np.zeros((N, N))
    f12 = -((7 + (np.cos(np.pi * x * y)) ** 2) * y ** 2 + np.cos(np.pi * x * y) * (1 + x * y)) * np.exp(-x * y)
    ###
    a22 = 4 * np.exp(x * y)
    b22 = 7 + (np.sin(np.pi * x * y)) ** 2
    c22 = np.sin(np.pi * x * y)
    d22 = -np.exp(-y)
    e22 = -np.exp(-x)
    f22 = np.zeros((N, N))
    ###
    a21 = np.zeros((N, N))
    b21 = np.zeros((N, N))
    c21 = np.zeros((N, N))
    d21 = np.zeros((N, N))
    e21 = np.zeros((N, N))
    f21 = np.ones((N, N))
    ###
    coeff1 = [a11, b11, c11, d11, e11, f11, a12, b12, c12, d12, e12, f12]
    coeff2 = [a21, b21, c21, d21, e21, f21, a22, b22, c22, d22, e22, f22]
    coeff = np.array([coeff1, coeff2])
    return coeff


################################################################################

def eq_22_bc(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_exact = np.exp(x * y)
    v_exact = np.exp(x + y)
    ###
    current[0][0, :] = u_exact[0, :]
    current[0][-1, :] = u_exact[-1, :]
    current[0][:, 0] = u_exact[:, 0]
    current[0][:, -1] = u_exact[:, -1]
    ###
    current[1][0, :] = v_exact[0, :]
    current[1][-1, :] = v_exact[-1, :]
    current[1][:, 0] = v_exact[:, 0]
    current[1][:, -1] = v_exact[:, -1]
    ###
    return current


def eq_22_exact(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_exact = np.exp(x * y)
    v_exact = np.exp(x + y)
    exact = np.array([u_exact, v_exact])
    return exact


def eq_22_rhs(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_rhs = ((1 + np.exp(x * y)) * x ** 2 + (2 + np.cos(np.pi * x)) * y ** 2 +
             (1 + x * y) * np.exp(-x * y) + y * np.exp(x) + x * np.exp(y) + np.sin(np.pi * x * y)) * np.exp(x * y) + \
            (4 + np.cos(2 * np.pi * x * y) + 2 + np.sin(np.pi * x * y) + np.exp(-x * y)
             + np.exp(x) + np.exp(y) + np.sin(np.pi * x * y) + 2) * np.exp(x + y)
    v_rhs = (2 + np.log(1 + x) + 4 + np.exp(2 * x * y + 3) / 200 + np.log(1 + x * y) +
             (1 + np.cos(4 * np.pi * x * y)) / 3 + 16 * np.ones((N, N))) * np.exp(x + y) + \
            (20 * np.exp(x * y) * x ** 2 + (7 + (np.cos(np.pi * x * y)) ** 2) * y ** 2 +
             np.cos(np.pi * x * y) * (x * y + 1) - y * np.exp(-2 * x * y) - x * np.exp(-x * y)) * np.exp(x * y)
    v_rhs[0, :] = 0
    v_rhs[N - 1, :] = 0
    v_rhs[:, N - 1] = 0
    v_rhs[:, 0] = 0
    u_rhs[0, :] = 0
    u_rhs[N - 1, :] = 0
    u_rhs[:, N - 1] = 0
    u_rhs[:, 0] = 0
    rhs = np.array([u_rhs, v_rhs])
    return rhs


def eq_22_coeff(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    ###
    a11 = 1 + np.exp(x * y)
    b11 = 2 + np.cos(np.pi * x)
    c11 = np.exp(-x * y)
    d11 = np.exp(x)
    e11 = np.exp(y)
    f11 = np.sin(np.pi * x * y)
    ###
    a12 = 4 + np.cos(2 * np.pi * x * y)
    b12 = 2 + np.sin(np.pi * x * y)
    c12 = np.exp(-x * y)
    d12 = np.exp(x)
    e12 = np.exp(y)
    f12 = np.sin(np.pi * x * y) + 2
    ###
    a22 = 2 + np.log(1 + x)
    b22 = 4 * np.ones((N, N))
    c22 = np.exp(2 * x * y + 3) / 200
    d22 = np.log(1 + x * y)
    e22 = (1 + np.cos(4 * np.pi * x * y)) / 3
    f22 = 16 * np.ones((N, N))
    ###
    a21 = 20 * np.exp(x * y)
    b21 = 7 + (np.cos(np.pi * x * y)) ** 2
    c21 = np.cos(np.pi * x * y)
    d21 = -np.exp(-2 * x * y)
    e21 = -np.exp(-x * y)
    f21 = np.zeros((N, N))
    ###
    coeff1 = [a11, b11, c11, d11, e11, f11, a12, b12, c12, d12, e12, f12]
    coeff2 = [a21, b21, c21, d21, e21, f21, a22, b22, c22, d22, e22, f22]
    coeff = np.array([coeff1, coeff2])
    return coeff


################################################################################

def get_quasilinear(dim, number, a=1):
    """
    This function provide two 1d quasilinear equations and to 2d quasilinear
    equations.


    -------------------
    dim = 1, number = 1

    [(1 + exp(x*y))d2y + (2 + cos(pi*x))d2x + exp(-x*y)dxdy + exp(x)dx +
    exp(y)dy + sin(pi*x*y)]u = rhs

    u_exact = exp(x*y), bc and rhs are taken from the operator and exact solution.
    -------------------
    dim = 1, number = 2

    [(4 + cos(2*pi*x*y))d2y + (2 + sin(pi*x*y))d2x + exp(-x*y)dxdy + exp(x)dx +
    exp(y)dy + sin(pi*x*y) + 2]u = rhs

    u_exact = exp(x+y), bc and rhs are taken from the operator and exact solution.
    -------------------
    dim = 2, number = 1

    Add description later!

    Parameters
    ----------
    dim: int
    Dimensionality: 2 for to equations, 1 for the one equation.

    number: int
    The number of the equation: 1 or 2 in case of any dimensionality.
    """
    if dim == 1:
        if number == 0:
            quasilinear = equation(eq_00_coeff, eq_00_rhs, 1, eq_00_bc, eq_00_exact)
        if number == 1:
            quasilinear = equation(eq_11_coeff, eq_11_rhs, 1, eq_11_bc, eq_11_exact)
        if number == 2:
            quasilinear = equation(eq_12_coeff, eq_12_rhs, 1, eq_12_bc, eq_12_exact)
        if number == 3:
            quasilinear = equation(eq_13_coeff, eq_13_rhs, 1, eq_13_bc, eq_13_exact)
        if number == 4:
            quasilinear = equation(eq_14_coeff, eq_14_rhs, 1, eq_14_bc, eq_14_exact)
        if number == 'red fox':
            rhs = lambda x: eq_red_fox_rhs(x, a)
            coeff = lambda x: eq_red_fox_coeff(x, a)
            quasilinear = equation(coeff, rhs, 1, eq_red_fox_bc, eq_red_fox_exact)
    if dim == 2:
        if number == 1:
            quasilinear = equation(eq_21_coeff, eq_21_rhs, 2, eq_21_bc, eq_21_exact)
        if number == 2:
            quasilinear = equation(eq_22_coeff, eq_22_rhs, 2, eq_22_bc, eq_22_exact)
    return quasilinear

################################################################################

def nleq_21_bc(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_exact = np.exp(-x * y)
    v_exact = np.exp(-2 * x * y)
    ###
    current[0][0, :] = u_exact[0, :]
    current[0][-1, :] = u_exact[-1, :]
    current[0][:, 0] = u_exact[:, 0]
    current[0][:, -1] = u_exact[:, -1]
    ###
    current[1][0, :] = v_exact[0, :]
    current[1][-1, :] = v_exact[-1, :]
    current[1][:, 0] = v_exact[:, 0]
    current[1][:, -1] = v_exact[:, -1]
    ###
    return current


def nleq_21_exact(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_exact = np.exp(-x * y)
    v_exact = np.exp(-2 * x * y)
    exact = np.array([u_exact, v_exact])
    return exact


def nleq_21_rhs(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_rhs = y ** 2 + np.exp(x * y) * x ** 2 + np.exp(-5 * x * y)
    v_rhs = np.exp(-5 * x * y) - 2 * y * np.exp(-2 * x * y) / 7 + \
            4 * np.exp(-2 * x * y) * y ** 2 + 4 * (np.cos(np.pi * x) ** 2 + 1) * np.exp(-2 * x * y) * x ** 2
    v_rhs[0, :] = 0;
    v_rhs[-1, :] = 0;
    v_rhs[:, -1] = 0;
    v_rhs[:, 0] = 0
    u_rhs[0, :] = 0;
    u_rhs[-1, :] = 0;
    u_rhs[:, -1] = 0;
    u_rhs[:, 0] = 0
    rhs = np.array([u_rhs, v_rhs])
    return rhs


def nleq_21_coeff(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    u, v = current
    x, y = np.meshgrid(z, z, indexing='ij')
    ###
    a11 = np.exp(2 * x * y)
    b11 = np.exp(x * y)
    c11 = np.zeros((N, N))
    d11 = np.zeros((N, N))
    e11 = np.zeros((N, N))
    f11 = np.zeros((N, N))
    ###
    a12 = np.zeros((N, N))
    b12 = np.zeros((N, N))
    c12 = np.zeros((N, N))
    d12 = np.zeros((N, N))
    e12 = np.zeros((N, N))
    f12 = u * v
    ###
    a22 = np.cos(np.pi * x) ** 2 + 1
    b22 = np.ones((N, N))
    c22 = np.zeros((N, N))
    d22 = np.ones((N, N)) / 7
    e22 = np.zeros((N, N))
    f22 = np.zeros((N, N))
    ###
    a21 = np.zeros((N, N))
    b21 = np.zeros((N, N))
    c21 = np.zeros((N, N))
    d21 = np.zeros((N, N))
    e21 = np.zeros((N, N))
    f21 = np.exp(-x * y) * u * v
    ###
    coeff1 = [a11, b11, c11, d11, e11, f11, a12, b12, c12, d12, e12, f12]
    coeff2 = [a21, b21, c21, d21, e21, f21, a22, b22, c22, d22, e22, f22]
    coeff = np.array([coeff1, coeff2])
    return coeff


def nleq_21_lcoeff(current):
    N, M = current[0].shape
    u, v = current
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    ###
    a11 = np.exp(2 * x * y)
    b11 = np.exp(x * y)
    c11 = np.zeros((N, N))
    d11 = np.zeros((N, N))
    e11 = np.zeros((N, N))
    f11 = v ** 2
    ###
    a12 = np.zeros((N, N))
    b12 = np.zeros((N, N))
    c12 = np.zeros((N, N))
    d12 = np.zeros((N, N))
    e12 = np.zeros((N, N))
    f12 = 2 * u * v
    ###
    a22 = np.cos(np.pi * x) ** 2 + 1
    b22 = np.ones((N, N))
    c22 = np.zeros((N, N))
    d22 = np.ones((N, N)) / 7
    e22 = np.zeros((N, N))
    f22 = np.exp(-x * y) * u ** 2
    ###
    a21 = np.zeros((N, N))
    b21 = np.zeros((N, N))
    c21 = np.zeros((N, N))
    d21 = np.zeros((N, N))
    e21 = np.zeros((N, N))
    f21 = 2 * np.exp(-x * y) * u * v
    ###
    coeff1 = [a11, b11, c11, d11, e11, f11, a12, b12, c12, d12, e12, f12]
    coeff2 = [a21, b21, c21, d21, e21, f21, a22, b22, c22, d22, e22, f22]
    coeff = np.array([coeff1, coeff2])
    return coeff


################################################################################

def nleq1_bc(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_exact = np.exp(x * y)
    current[0][0, :] = u_exact[0, :]
    current[0][-1, :] = u_exact[-1, :]
    current[0][:, 0] = u_exact[:, 0]
    current[0][:, -1] = u_exact[:, -1]
    return current


def nleq1_exact(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_exact = np.exp(x * y)
    return np.array([u_exact])


def nleq1_rhs(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_rhs = np.exp(x * y) * ((4 + np.exp(x * y)) * y ** 2 + (4 + np.exp(-x * y)) * x ** 2 + \
                             (1 + x * y) * np.exp(-2 * x * y) + np.cos(np.pi * x * y) * y + np.sin(
                np.pi * x * y) * x + np.sinh(2 * x * y))
    u_rhs[0, :] = 0;
    u_rhs[N - 1, :] = 0;
    u_rhs[:, N - 1] = 0;
    u_rhs[:, 0] = 0
    rhs = np.array([u_rhs])
    return rhs


def nleq1_l_coeff(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    ###
    a11 = 4 + np.exp(-x * y)
    b11 = 4 + current[0]
    c11 = np.exp(-2 * x * y)
    d11 = np.cos(np.pi * x * y)
    e11 = np.sin(np.pi * x * y)
    f11 = np.sinh(2 * x * y) + llt.d2x(current[0], N)
    ###
    coeff1 = [a11, b11, c11, d11, e11, f11]
    coeff = np.array([coeff1])
    return coeff


def nleq1_coeff(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    ###
    a11 = 4 + np.exp(-x * y)
    b11 = 4 + current[0]
    c11 = np.exp(-2 * x * y)
    d11 = np.cos(np.pi * x * y)
    e11 = np.sin(np.pi * x * y)
    f11 = np.sinh(2 * x * y)
    ###
    coeff1 = [a11, b11, c11, d11, e11, f11]
    coeff = np.array([coeff1])
    return coeff


################################################################################

def get_nonlinear(dim):
    global nonlinear
    if dim == 2:
        nonlinear = equation(nleq_21_coeff, nleq_21_rhs, 2, nleq_21_bc, nleq_21_exact, nleq_21_lcoeff)
    if dim == 1:
        nonlinear = equation(nleq1_coeff, nleq1_rhs, 1, nleq1_bc, nleq1_exact, l_coeff=nleq1_l_coeff)
    return nonlinear


################################################################################


def trivial_harmonic_bc(current):
    N, M = current[0].shape
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_exact = x
    v_exact = y
    ###
    current[0][0, :] = u_exact[0, :]
    current[0][-1, :] = u_exact[-1, :]
    current[0][:, 0] = u_exact[:, 0]
    current[0][:, -1] = u_exact[:, -1]
    ###
    current[1][0, :] = v_exact[0, :]
    current[1][-1, :] = v_exact[-1, :]
    current[1][:, 0] = v_exact[:, 0]
    current[1][:, -1] = v_exact[:, -1]
    ###
    return current

def trivial_harmonic_rhs(current):
    return np.zeros_like(current)

def basic_harmonic_coeff(current):
    N, M = current[0].shape
    u, v = current
    u_x, v_x = llt.dx(u, N), llt.dx(v, N)
    u_y, v_y = llt.dy(u, N), llt.dy(v, N)
    ###
    a11 = u_x ** 2 + v_x ** 2
    b11 = u_y ** 2 + v_y ** 2
    c11 = -2 * (u_x * u_y + v_x * v_y)
    d11 = np.zeros((N, N))
    e11 = np.zeros((N, N))
    f11 = np.zeros((N, N))
    ###
    a12 = np.zeros((N, N))
    b12 = np.zeros((N, N))
    c12 = np.zeros((N, N))
    d12 = np.zeros((N, N))
    e12 = np.zeros((N, N))
    f12 = np.zeros((N, N))
    ###
    a22 = u_x ** 2 + v_x ** 2
    b22 = u_y ** 2 + v_y ** 2
    c22 = -2 * (u_x * u_y + v_x * v_y)
    d22 = np.zeros((N, N))
    e22 = np.zeros((N, N))
    f22 = np.zeros((N, N))
    ###
    a21 = np.zeros((N, N))
    b21 = np.zeros((N, N))
    c21 = np.zeros((N, N))
    d21 = np.zeros((N, N))
    e21 = np.zeros((N, N))
    f21 = np.zeros((N, N))
    ###
    coeff1 = [a11, b11, c11, d11, e11, f11, a12, b12, c12, d12, e12, f12]
    coeff2 = [a21, b21, c21, d21, e21, f21, a22, b22, c22, d22, e22, f22]
    coeff = np.array([coeff1, coeff2])
    return coeff

def diagonal_metrics(current):
    N, M = current[0].shape
    g = np.zeros((2, 2, N, N))
    g[1, 1], g[0, 0] = 1, 1
    return g

def harmonic_coeff(current, metrics=diagonal_metrics):
    N, M = current[0].shape
    u, v = current
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_x, v_x = llt.dx(u, N), llt.dx(v, N)
    u_y, v_y = llt.dy(u, N), llt.dy(v, N)
    g = metrics(current)
    det_g_x = g[0, 0]*g[1, 1] - g[0, 1]*g[1, 0]
    J_xi = u_x*v_y - u_y*v_x
    R = J_xi**2
    ###
    a11 = g[0, 0]*u_x**2 + g[1, 1]*v_x**2 + 2*g[0, 1]*u_x*v_x
    b11 = g[0, 0]*u_y**2 + g[1, 1]*v_y**2 + 2*g[0, 1]*u_y*v_y
    c11 = -2*(g[0, 0]*u_x*u_y + g[1, 1]*v_x*v_y + g[0, 1]*(v_x*u_y + u_x*v_y))
    d11 = R*llt.dy(g[0, 1]/np.sqrt(det_g_x), N)
    e11 = -R*llt.dx(g[0, 1]/np.sqrt(det_g_x), N)
    f11 = np.zeros((N, N))
    ###
    a12 = np.zeros((N, N))
    b12 = np.zeros((N, N))
    c12 = np.zeros((N, N))
    d12 = R*llt.dy(g[1, 1]/np.sqrt(det_g_x), N)
    e12 = -R*llt.dx(g[1, 1]/np.sqrt(det_g_x), N)
    f12 = np.zeros((N, N))
    ###
    a22 = a11
    b22 = b11
    c22 = c11
    d22 = -R*llt.dy(g[1, 0]/np.sqrt(det_g_x), N)
    e22 = R*llt.dx(g[1, 0]/np.sqrt(det_g_x), N)
    f22 = np.zeros((N, N))
    ###
    a21 = np.zeros((N, N))
    b21 = np.zeros((N, N))
    c21 = np.zeros((N, N))
    d21 = -R*llt.dy(g[0, 0]/np.sqrt(det_g_x), N)
    e21 = R*llt.dx(g[0, 0]/np.sqrt(det_g_x), N)
    f21 = np.zeros((N, N))
    ###
    coeff1 = [a11, b11, c11, d11, e11, f11, a12, b12, c12, d12, e12, f12]
    coeff2 = [a21, b21, c21, d21, e21, f21, a22, b22, c22, d22, e22, f22]
    coeff = np.array([coeff1, coeff2])
    return coeff

def winslow_coeff(current, metrics=diagonal_metrics):
    N, M = current[0].shape
    u, v = current
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_x, v_x = llt.dx(u, N), llt.dx(v, N)
    u_y, v_y = llt.dy(u, N), llt.dy(v, N)
    g = metrics(current)
    det_g_x = g[0, 0]*g[1, 1] - g[0, 1]*g[1, 0]
    J_xi = u_x*v_y - u_y*v_x
    R = np.sqrt(det_g_x)*J_xi**2
    ###
    a11 = g[0, 0]*u_x**2 + g[1, 1]*v_x**2 + 2*g[0, 1]*u_x*v_x
    b11 = g[0, 0]*u_y**2 + g[1, 1]*v_y**2 + 2*g[0, 1]*u_y*v_y
    c11 = -2*(g[0, 0]*u_x*u_y + g[1, 1]*v_x*v_y + g[0, 1]*(v_x*u_y + u_x*v_y))
    d11 = R*llt.dy(g[0, 1]/det_g_x, N)
    e11 = -R*llt.dx(g[0, 1]/det_g_x, N)
    f11 = np.zeros((N, N))
    ###
    a12 = np.zeros((N, N))
    b12 = np.zeros((N, N))
    c12 = np.zeros((N, N))
    d12 = R*llt.dy(g[1, 1]/det_g_x, N)
    e12 = -R*llt.dx(g[1, 1]/det_g_x, N)
    f12 = np.zeros((N, N))
    ###
    a22 = a11
    b22 = b11
    c22 = c11
    d22 = -R*llt.dy(g[1, 0]/det_g_x, N)
    e22 = R*llt.dx(g[1, 0]/det_g_x, N)
    f22 = np.zeros((N, N))
    ###
    a21 = np.zeros((N, N))
    b21 = np.zeros((N, N))
    c21 = np.zeros((N, N))
    d21 = -R*llt.dy(g[0, 0]/det_g_x, N)
    e21 = R*llt.dx(g[0, 0]/det_g_x, N)
    f21 = np.zeros((N, N))
    ###
    coeff1 = [a11, b11, c11, d11, e11, f11, a12, b12, c12, d12, e12, f12]
    coeff2 = [a21, b21, c21, d21, e21, f21, a22, b22, c22, d22, e22, f22]
    coeff = np.array([coeff1, coeff2])
    return coeff

def advection_harmonic_rhs(current, metrics=diagonal_metrics):
    rhs = np.zeros_like(current)
    N, M = current[0].shape
    u, v = current
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_x, v_x = llt.dx(u, N), llt.dx(v, N)
    u_y, v_y = llt.dy(u, N), llt.dy(v, N)
    g = metrics(N)
    det_g_x = g[0, 0]*g[1, 1] - g[0, 1]*g[1, 0]
    J_xi = u_x*v_y - u_y*v_x
    R = J_xi**2
    rhs[0] = -R*llt.dy(g[0, 1]/np.sqrt(det_g_x), N)*u_x + R*llt.dx(g[0, 1]/np.sqrt(det_g_x), N)*u_y
    - R*llt.dy(g[1, 1]/np.sqrt(det_g_x), N)*v_x + R*llt.dx(g[1, 1]/np.sqrt(det_g_x), N)*v_y
    rhs[1] = R*llt.dy(g[1, 0]/np.sqrt(det_g_x), N)*v_x - R*llt.dx(g[1, 0]/np.sqrt(det_g_x), N)*v_y
    + R*llt.dy(g[0, 0]/np.sqrt(det_g_x), N)*u_x - R*llt.dx(g[0, 0]/np.sqrt(det_g_x), N)*u_y
    return rhs

def advection_free_harmonic_coeff(current, metrics=diagonal_metrics):
    N, M = current[0].shape
    u, v = current
    z = np.linspace(0, 1, N)
    x, y = np.meshgrid(z, z, indexing='ij')
    u_x, v_x = llt.dx(u, N), llt.dx(v, N)
    u_y, v_y = llt.dy(u, N), llt.dy(v, N)
    g = metrics(N)
    det_g_x = g[0, 0]*g[1, 1] - g[0, 1]*g[1, 0]
    J_xi = u_x*v_y - u_y*v_x
    R = J_xi**2
    ###
    a11 = g[0, 0]*u_y**2 + g[1, 1]*v_y**2 + 2*g[0, 1]*u_y*v_y
    b11 = g[0, 0]*u_x**2 + g[1, 1]*v_x**2 + 2*g[0, 1]*u_x*v_x
    c11 = -2*(g[0, 0]*u_x*u_y + g[1, 1]*v_x*v_y + g[0, 1]*(v_x*u_y + u_x*v_y))
    d11 = np.zeros((N, N))
    e11 = np.zeros((N, N))
    f11 = np.zeros((N, N))
    ###
    a12 = np.zeros((N, N))
    b12 = np.zeros((N, N))
    c12 = np.zeros((N, N))
    d12 = np.zeros((N, N))
    e12 = np.zeros((N, N))
    f12 = np.zeros((N, N))
    ###
    a22 = a11
    b22 = b11
    c22 = c11
    d22 = np.zeros((N, N))
    e22 = np.zeros((N, N))
    f22 = np.zeros((N, N))
    ###
    a21 = np.zeros((N, N))
    b21 = np.zeros((N, N))
    c21 = np.zeros((N, N))
    d21 = np.zeros((N, N))
    e21 = np.zeros((N, N))
    f21 = np.zeros((N, N))
    ###
    coeff1 = [a11, b11, c11, d11, e11, f11, a12, b12, c12, d12, e12, f12]
    coeff2 = [a21, b21, c21, d21, e21, f21, a22, b22, c22, d22, e22, f22]
    coeff = np.array([coeff1, coeff2])
    return coeff

def basic_mixed_harmonic_coeff(current):
    N, M = current[0].shape
    u, v = current
    u_x, v_x = llt.dx(u, N), llt.dx(v, N)
    u_y, v_y = llt.dy(u, N), llt.dy(v, N)
    u_x_f, v_x_f = llt.dx_forward(u, N), llt.dx_forward(v, N)
    u_y_f, v_y_f = llt.dy_forward(u, N), llt.dy_forward(v, N)
    u_x_b, v_x_b = llt.dx_backward(u, N), llt.dx_backward(v, N)
    u_y_b, v_y_b = llt.dy_backward(u, N), llt.dy_backward(v, N)
    ###
    a11 = u_x_f * u_x_b + v_x_f * v_x_b
    b11 = u_y_f * u_y_b + v_y_f * v_y_b
    c11 = -2 * (u_x * u_y + v_x * v_y)
    d11 = np.zeros((N, N))
    e11 = np.zeros((N, N))
    f11 = np.zeros((N, N))
    ###
    a12 = np.zeros((N, N))
    b12 = np.zeros((N, N))
    c12 = np.zeros((N, N))
    d12 = np.zeros((N, N))
    e12 = np.zeros((N, N))
    f12 = np.zeros((N, N))
    ###
    a22 = u_x_f * u_x_b + v_x_f * v_x_b
    b22 = u_y_f * u_y_b + v_y_f * v_y_b
    c22 = -2 * (u_x * u_y + v_x * v_y)
    d22 = np.zeros((N, N))
    e22 = np.zeros((N, N))
    f22 = np.zeros((N, N))
    ###
    a21 = np.zeros((N, N))
    b21 = np.zeros((N, N))
    c21 = np.zeros((N, N))
    d21 = np.zeros((N, N))
    e21 = np.zeros((N, N))
    f21 = np.zeros((N, N))
    ###
    coeff1 = [a11, b11, c11, d11, e11, f11, a12, b12, c12, d12, e12, f12]
    coeff2 = [a21, b21, c21, d21, e21, f21, a22, b22, c22, d22, e22, f22]
    coeff = np.array([coeff1, coeff2])
    return coeff


def basic_fair_newton_harmonic_linear_coeff(current):
    N, M = current[0].shape
    u, v = current
    u_x, v_x = llt.dx(u, N), llt.dx(v, N)
    u_y, v_y = llt.dy(u, N), llt.dy(v, N)
    u_xx, v_xx = llt.d2x(u, N), llt.d2x(v, N)
    u_yy, v_yy = llt.d2y(u, N), llt.d2y(v, N)
    u_xy, v_xy = llt.dxdy(u, N), llt.dxdy(v, N)
    u0, v0 = np.zeros_like(u), np.zeros_like(v)
    ###
    a11 = u_x ** 2 + v_x ** 2
    b11 = u_y ** 2 + v_y ** 2
    c11 = -2 * (u_x * u_y + v_x * v_y)
    d11 = 2 * (u_yy * u_x - u_xy * u_y)
    e11 = 2 * (u_xx * u_y - u_xy * u_x)
    f11 = np.zeros((N, N))
    ###
    a12 = np.zeros((N, N))
    b12 = np.zeros((N, N))
    c12 = np.zeros((N, N))
    d12 = 2 * (u_yy * v_x - u_xy * v_y)
    e12 = 2 * (u_xx * v_y - u_xy * v_x)
    f12 = np.zeros((N, N))
    ###
    a22 = u_x ** 2 + v_x ** 2
    b22 = u_y ** 2 + v_y ** 2
    c22 = -2 * (u_x * u_y + v_x * v_y)
    d22 = 2 * (v_yy * v_x - v_xy * v_y)
    e22 = 2 * (v_xx * v_y - v_xy * v_x)
    f22 = np.zeros((N, N))
    ###
    a21 = np.zeros((N, N))
    b21 = np.zeros((N, N))
    c21 = np.zeros((N, N))
    d21 = 2 * (v_yy * u_x - v_xy * u_y)
    e21 = 2 * (v_xx * u_y - v_xy * u_x)
    f21 = np.zeros((N, N))
    ###
    coeff1 = [a11, b11, c11, d11, e11, f11, a12, b12, c12, d12, e12, f12]
    coeff2 = [a21, b21, c21, d21, e21, f21, a22, b22, c22, d22, e22, f22]
    coeff = np.array([coeff1, coeff2])
    return coeff

################################################################################

def get_harmonic(name):
    global res
    if name == 'Fair Newton':
        res = harmonic_equation(basic_harmonic_coeff, trivial_harmonic_rhs, 2, bc=trivial_harmonic_bc,
                                l_coeff=basic_fair_newton_harmonic_linear_coeff)
    if name == 'Frozen Metric':
        res = harmonic_equation(basic_harmonic_coeff, trivial_harmonic_rhs, 2, bc=trivial_harmonic_bc)
    if name == 'Harmonic Frozen Metric':
        res = harmonic_equation(harmonic_coeff, trivial_harmonic_rhs, 2, bc=trivial_harmonic_bc)
    if name == 'Winslow Frozen Metric':
        res = harmonic_equation(winslow_coeff, trivial_harmonic_rhs, 2, bc=trivial_harmonic_bc)
    if name == 'Harmonic with Frozen Metric':
        res = harmonic_equation(harmonic_coeff, trivial_harmonic_rhs, 2, bc=trivial_harmonic_bc)
    if name == 'Upwind + Downwind Frozen Metric':
        res = harmonic_equation(basic_mixed_harmonic_coeff, trivial_harmonic_rhs, 2, bc=trivial_harmonic_bc)
    return res
