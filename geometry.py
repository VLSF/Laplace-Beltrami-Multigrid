import numpy as np
import low_level_tools as llt

def transfinite_interpolation(boundary_values):
    right, up, left, down = boundary_values
    M = len(right)
    z = np.linspace(0, 1, M)
    x, y = np.meshgrid(z, z, indexing='ij')
    horizontal = left*(1-x) + right*x
    mod_down = down - horizontal[:, 0]
    mod_up = up - horizontal[:, -1]
    vertical = (mod_down*(1-y).T).T + (mod_up*y.T).T
    return vertical + horizontal

def TI(current):
    x, y = current
    y1 = transfinite_interpolation([y[-1, :], y[:, -1], y[0, :], y[:, 0]])
    x1 = transfinite_interpolation([x[-1, :], x[:, -1], x[0, :], x[:, 0]])
    return np.array([x1, y1])
    
def contravariant_metric(solution):
    u, v = solution
    N, M = u.shape
    u_x, v_x = llt.dx_with_boundary(u, N), llt.dx_with_boundary(v, N)
    u_y, v_y = llt.dy_with_boundary(u, N), llt.dy_with_boundary(v, N)
    g = np.array([[u_y**2 + v_y**2, -(u_x*u_y + v_x*v_y)], [-(u_x*u_y + v_x*v_y) , u_x**2 + v_x**2]])
    return g

def covariant_metric(solution):
    u, v = solution
    N, M = u.shape
    u_x, v_x = llt.dx_with_boundary(u, N), llt.dx_with_boundary(v, N)
    u_y, v_y = llt.dy_with_boundary(u, N), llt.dy_with_boundary(v, N)
    g = np.array([[u_x**2 + v_x**2, u_x*u_y + v_x*v_y], [u_x*u_y + v_x*v_y , u_y**2 + v_y**2]])
    return g

def SO(a, b, c):
    """
    For the symmetric matrices of the form

    | a b |
    | b c |

    this function returns orthogonal transformations that diagonalize matrices
    and matrices in the new coordinates of the form

    | l_1 0 |
    | 0 l_2 |

    where l_1 and l_2 are eigenvalues.

    Parameters
    ----------
    a: ndarray
    Shape of the a is (N, M). It represents the [1, 1] element of the matrix on the subset of points
    with coordinates x[i,j], y[i,j].

    b: ndarray
    Same shape, element [1, 2] and [2, 1].

    c: ndarray
    Same shape, element [2, 2].

    Returns
    -------
    O: ndarray
    Array containing orthogonal matrices such that `O @ A @ O.T` is the diagonal matrix, det(O) = 1.
    The shape is (N, M, 2, 2) so O[i, j, :, :] is the matrix correspond to the point [i, j].

    diag_M: ndarray
    Array containing diagonal form of the original matrices. The shape is (N, M, 2, 2).
    """
    N, M = a.shape
    l_plus = (a+c)/2 + np.sqrt(((a-c)/2)**2 + b**2)
    l_minus = (a+c)/2 - np.sqrt(((a-c)/2)**2 + b**2)
    v_plus = np.array([b, l_plus-a])/np.linalg.norm(np.array([b, l_plus-a]), axis=0)
    v_minus = np.array([b, l_minus-a])/np.linalg.norm(np.array([b, l_minus-a]), axis=0)
    O = np.dstack((v_plus.T, v_minus.T)).reshape((N, M, 2, 2))
    O[:, :, :, 0] *= np.dstack((np.linalg.det(O), np.linalg.det(O)))
    diag_M = np.zeros((N, M, 2, 2))
    diag_M[:, :, 0, 0] = l_plus
    diag_M[:, :, 1, 1] = l_minus
    return np.transpose(O, axes=[1, 0, 3, 2]), diag_M

def transform_field(O, field, up_or_down='up'):
    """
    You give a field for the transformation and the field of SO(2) transformations
    and specify the type (covariant or the contravariant) and we will do the rest!

    Parameters
    ----------
    O: ndarray
    Field of the SO(2) transformations on the plain.
    The shape is (N, M, 2, 2) first two dimensions are coordinates of the point, other two
    are colums and rows of the matrix.

    field: ndarray
    Field to be transformed of the same shape as `O` with exactly the same meaning.

    up_or_down: string
    'up' or 'down' in case of the 'up' (contravariant) at the each point initial vector v -> O v and
    in case of the 'down' (covariant) at the each point initial vector v -> O.T v.
    This rule is consistent with the output of the function `SO(a, b, c)`.
    """
    global res
    if up_or_down=='up':
        res = np.einsum('ijkl, lij -> kij', O, field)
    if up_or_down=='down':
        res = np.einsum('ijlk, lij -> kij', O, field)
    return res
