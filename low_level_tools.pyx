import numpy as np

def xRed_correction_2d(double [:,:] u, double [:,:] v, N, coeff):
    h = 1/(N-1)
    a1, c1, e1, a2, c2, e2 = coeff
    w = np.zeros_like(u)
    cdef long i, j
    for j in range(int((N-1)/2)):
        j = 2*j + 1
        for i in range(1, N-1):
            w[i, j] = c1[i, j]*(u[i+1, j+1] + u[i-1, j-1] - u[i-1, j+1] - u[i+1, j-1])/4 +\
                        (a1[i, j] + e1[i, j]*h/2)*u[i, j+1] + (a1[i, j] - e1[i, j]*h/2)*u[i, j-1] +\
                            c2[i, j]*(v[i+1, j+1] + v[i-1, j-1] - v[i-1, j+1] - v[i+1, j-1])/4 +\
                                (a2[i, j] + e2[i, j]*h/2)*v[i, j+1] + (a2[i, j] - e2[i, j]*h/2)*v[i, j-1]
            w[i,j] /= h**2
    return w

def xBlack_correction_2d(double [:,:] u, double [:,:] v, N, coeff):
    h = 1/(N-1)
    a1, c1, e1, a2, c2, e2 = coeff
    w = np.zeros_like(u)
    cdef long i, j
    for j in range(int((N-1)/2)-1):
        j = 2*j + 2
        for i in range(1, N-1):
            w[i, j] = c1[i, j]*(u[i+1, j+1] + u[i-1, j-1] - u[i-1, j+1] - u[i+1, j-1])/4 +\
                        (a1[i, j] + e1[i, j]*h/2)*u[i, j+1] + (a1[i, j] - e1[i, j]*h/2)*u[i, j-1] +\
                            c2[i, j]*(v[i+1, j+1] + v[i-1, j-1] - v[i-1, j+1] - v[i+1, j-1])/4 +\
                                (a2[i, j] + e2[i, j]*h/2)*v[i, j+1] + (a2[i, j] - e2[i, j]*h/2)*v[i, j-1]
            w[i,j] /= h**2
    return w

def xRed_boundary_correction_2d(double [:,:] u, double [:,:] v, N, coeff):
    h = 1/(N-1)
    a1, c1, e1, a2, c2, e2 = coeff
    w = np.zeros_like(u)
    cdef long i, j
    for i in range(1, N-1):
        j = 1
        w[i, j] = c1[i, j]*(u[i+1, j+1] + u[i-1, j-1] - u[i-1, j+1] - u[i+1, j-1])/4 +\
                    (a1[i, j] + e1[i, j]*h/2)*u[i, j+1] + (a1[i, j] - e1[i, j]*h/2)*u[i, j-1] +\
                        c2[i, j]*(v[i+1, j+1] + v[i-1, j-1] - v[i-1, j+1] - v[i+1, j-1])/4 +\
                            (a2[i, j] + e2[i, j]*h/2)*v[i, j+1] + (a2[i, j] - e2[i, j]*h/2)*v[i, j-1]
        w[i,j] /= h**2
    for i in range(1, N-1):
        j = N-2
        w[i, j] = c1[i, j]*(u[i+1, j+1] + u[i-1, j-1] - u[i-1, j+1] - u[i+1, j-1])/4 +\
                    (a1[i, j] + e1[i, j]*h/2)*u[i, j+1] + (a1[i, j] - e1[i, j]*h/2)*u[i, j-1] +\
                        c2[i, j]*(v[i+1, j+1] + v[i-1, j-1] - v[i-1, j+1] - v[i+1, j-1])/4 +\
                            (a2[i, j] + e2[i, j]*h/2)*v[i, j+1] + (a2[i, j] - e2[i, j]*h/2)*v[i, j-1]
        w[i,j] /= h**2
    return w

def yRed_correction_2d(double [:,:] u, double [:,:] v, N, coeff):
    h = 1/(N-1)
    b1, c1, d1, b2, c2, d2 = coeff
    w = np.zeros_like(u)
    cdef long i, j
    for i in range(int((N-1)/2)):
        i = 2*i + 1
        for j in range(1, N-1):
            w[i, j] = c1[i, j]*(u[i+1, j+1] + u[i-1, j-1] - u[i-1, j+1] - u[i+1, j-1])/4 +\
                        (b1[i, j] + d1[i, j]*h/2)*u[i+1, j] + (b1[i, j] - d1[i, j]*h/2)*u[i-1, j] +\
                            c2[i, j]*(v[i+1, j+1] + v[i-1, j-1] - v[i-1, j+1] - v[i+1, j-1])/4 +\
                                (b2[i, j] + d2[i, j]*h/2)*v[i+1, j] + (b2[i, j] - d2[i, j]*h/2)*v[i-1, j]
            w[i,j] /= h**2
    return w

def yRed_boundary_correction_2d(double [:,:] u, double [:,:] v, N, coeff):
    h = 1/(N-1)
    b1, c1, d1, b2, c2, d2 = coeff
    w = np.zeros_like(u)
    cdef long i, j
    i = 1
    for j in range(1, N-1):
        i = 1
        w[i, j] = c1[i, j]*(u[i+1, j+1] + u[i-1, j-1] - u[i-1, j+1] - u[i+1, j-1])/4 +\
                    (b1[i, j] + d1[i, j]*h/2)*u[i+1, j] + (b1[i, j] - d1[i, j]*h/2)*u[i-1, j] +\
                        c2[i, j]*(v[i+1, j+1] + v[i-1, j-1] - v[i-1, j+1] - v[i+1, j-1])/4 +\
                            (b2[i, j] + d2[i, j]*h/2)*v[i+1, j] + (b2[i, j] - d2[i, j]*h/2)*v[i-1, j]
        w[i,j] /= h**2
    for j in range(1, N-1):
        i = N-2
        w[i, j] = c1[i, j]*(u[i+1, j+1] + u[i-1, j-1] - u[i-1, j+1] - u[i+1, j-1])/4 +\
                    (b1[i, j] + d1[i, j]*h/2)*u[i+1, j] + (b1[i, j] - d1[i, j]*h/2)*u[i-1, j] +\
                        c2[i, j]*(v[i+1, j+1] + v[i-1, j-1] - v[i-1, j+1] - v[i+1, j-1])/4 +\
                            (b2[i, j] + d2[i, j]*h/2)*v[i+1, j] + (b2[i, j] - d2[i, j]*h/2)*v[i-1, j]
        w[i,j] /= h**2
    return w

def yBlack_correction_2d(double [:,:] u, double [:,:] v, N, coeff):
    h = 1/(N-1)
    b1, c1, d1, b2, c2, d2 = coeff
    w = np.zeros_like(u)
    cdef long i, j
    for i in range(int((N-1)/2)-1):
        i = 2*i + 2
        for j in range(1, N-1):
            w[i, j] = c1[i, j]*(u[i+1, j+1] + u[i-1, j-1] - u[i-1, j+1] - u[i+1, j-1])/4 +\
                        (b1[i, j] + d1[i, j]*h/2)*u[i+1, j] + (b1[i, j] - d1[i, j]*h/2)*u[i-1, j] +\
                            c2[i, j]*(v[i+1, j+1] + v[i-1, j-1] - v[i-1, j+1] - v[i+1, j-1])/4 +\
                                (b2[i, j] + d2[i, j]*h/2)*v[i+1, j] + (b2[i, j] - d2[i, j]*h/2)*v[i-1, j]
            w[i,j] /= h**2
    return w

def PDE(double [:,:] u, N,  coefficients):
    h = 1/(N-1)
    a, b, c, d, e, f = coefficients
    v = np.zeros((N, N))
    cdef long i, j
    for i in range(1, N-1):
        for j in range(1, N-1):
            v[i, j] = (-(2*a[i, j] + 2*b[i, j]) + f[i, j]*h**2)*u[i, j] + c[i, j]*(u[i+1, j+1] + u[i-1, j-1] - u[i-1, j+1] - u[i+1, j-1])/4 +\
            (b[i, j] + d[i, j]*h/2)*u[i+1, j] + (b[i, j] - d[i, j]*h/2)*u[i-1, j] + (a[i, j] + e[i, j]*h/2)*u[i, j+1] + (a[i, j] - e[i, j]*h/2)*u[i, j-1]
    return v/h**2

def modify_rhs(double [:,:] u, N, coefficients):
    h = 1/(N-1)
    a, b, c, d, e, f = coefficients
    v = np.zeros((N, N))
    cdef long i, j
    i = 1
    for j in range(1, N-1):
        v[i, j] = c[i, j]*(u[i-1, j-1] - u[i-1, j+1])/4 + (b[i, j] - d[i, j]*h/2)*u[i-1, j]
    i = N-2
    for j in range(1, N-1):
        v[i, j] = c[i, j]*(u[i+1, j+1] - u[i+1, j-1])/4 + (b[i, j] + d[i, j]*h/2)*u[i+1, j]
    j = 1
    for i in range(1, N-1):
        if i == 1:
            v[i, j] = c[i, j]*(u[i-1, j-1] - u[i-1, j+1] - u[i+1, j-1])/4 +(b[i, j] - d[i, j]*h/2)*u[i-1, j] + (a[i, j] - e[i, j]*h/2)*u[i, j-1]
        if i == N-2:
            v[i, j] = c[i, j]*(u[i+1, j+1] + u[i-1, j-1] - u[i+1, j-1])/4 + (b[i, j] + d[i, j]*h/2)*u[i+1, j] + (a[i, j] - e[i, j]*h/2)*u[i, j-1]
        if i != 1 and i != N-2:
            v[i, j] = c[i, j]*(u[i-1, j-1] - u[i+1, j-1])/4 + (a[i, j] - e[i, j]*h/2)*u[i, j-1]
    j = N-2
    for i in range(1, N-1):
        if i == 1:
            v[i, j] = c[i, j]*(u[i+1, j+1] + u[i-1, j-1] - u[i-1, j+1])/4 + (b[i, j] - d[i, j]*h/2)*u[i-1, j] + (a[i, j] + e[i, j]*h/2)*u[i, j+1]
        if i == N-2:
            v[i, j] = c[i, j]*(u[i+1, j+1] - u[i-1, j+1] - u[i+1, j-1])/4 + (b[i, j] + d[i, j]*h/2)*u[i+1, j] + (a[i, j] + e[i, j]*h/2)*u[i, j+1]
        if i != 1 and i != N-2:
            v[i, j] = c[i, j]*(u[i+1, j+1] - u[i-1, j+1])/4 + (a[i, j] + e[i, j]*h/2)*u[i, j+1]
    return v/h**2

def get_sparse_1d(n_row, n_col, N, coefficients):
    h = 1/(N+1)
    a, b, c, d, e, f = coefficients
    a = a.T[1:-1, 1:-1].reshape((-1,))
    b = b.T[1:-1, 1:-1].reshape((-1,))
    c = c.T[1:-1, 1:-1].reshape((-1,))
    d = d.T[1:-1, 1:-1].reshape((-1,))
    e = e.T[1:-1, 1:-1].reshape((-1,))
    f = f.T[1:-1, 1:-1].reshape((-1,))
    row = []
    col = []
    data = []
    index = [0, 1, -1, N, N+1, N-1, -N, -N+1, -N-1]
    coeff = np.array([f - 2*(a+b)/h**2, b/h**2 + d/(2*h), b/h**2 - d/(2*h), \
                a/h**2 + e/(2*h), c/(4*h**2), -c/(4*h**2), a/h**2 - e/(2*h), -c/(4*h**2), c/(4*h**2)])
    cdef long i
    for i in range(N**2):
        for j, c in zip(index, coeff[:, i]):
            if i+j >= 0 and i+j <= (N**2-1):
                if i%N == 0:
                    if j!=-1 and j!=N-1 and j!=-N-1:
                        row.append(n_row + i)
                        col.append(n_col + i + j)
                        data.append(c)
                if (i+1)%N == 0:
                    if j!=1 and j!=N+1 and j!=-N+1:
                        row.append(n_row + i)
                        col.append(n_col + i + j)
                        data.append(c)
                if i%N != 0 and (i+1)%N != 0:
                    row.append(n_row + i)
                    col.append(n_col + i + j)
                    data.append(c)
    return np.array(data), np.array(row), np.array(col)

def xGS(double [:,:] u, double [:,:] rhs, N,  coefficients):
    h = 1/(N-1)
    a, b, c, d, e, f = coefficients
    v = np.copy(u)
    cdef long i, j
    for j in range(1, N-1):
        for i in range(1, N-1):
            v[i, j] = rhs[i, j]*h**2 - c[i, j]*(v[i+1, j+1] + v[i-1, j-1] - v[i-1, j+1] - v[i+1, j-1])/4 -\
            (b[i, j] + d[i, j]*h/2)*v[i+1, j] - (b[i, j] - d[i, j]*h/2)*v[i-1, j] - (a[i, j] + e[i, j]*h/2)*v[i, j+1] - (a[i, j] - e[i, j]*h/2)*v[i, j-1]
            v[i, j] = v[i, j]/(-(2*a[i, j] + 2*b[i, j]) + f[i, j]*h**2)
    return v

def xRed_correction_1d(double [:,:] u, N, coeff):
    h = 1/(N-1)
    a1, c1, e1 = coeff
    w = np.zeros_like(u)
    cdef long i, j
    for j in range(int((N-1)/2)):
        j = 2*j + 1
        for i in range(1, N-1):
            w[i, j] = c1[i, j]*(u[i+1, j+1] + u[i-1, j-1] - u[i-1, j+1] - u[i+1, j-1])/4 +\
                        (a1[i, j] + e1[i, j]*h/2)*u[i, j+1] + (a1[i, j] - e1[i, j]*h/2)*u[i, j-1]
            w[i,j] /= h**2
    return w

def xGS_correction(double [:,:] u, N, coeff, long j):
    h = 1/(N-1)
    a1, c1, e1 = coeff
    w = np.zeros_like(u[:, j])
    cdef long i
    for i in range(1, N-1):
        w[i] = c1[i, j]*(u[i+1, j+1] + u[i-1, j-1] - u[i-1, j+1] - u[i+1, j-1])/4 +\
                    (a1[i, j] + e1[i, j]*h/2)*u[i, j+1] + (a1[i, j] - e1[i, j]*h/2)*u[i, j-1]
        w[i] /= h**2
    return w

def yGS_correction(double [:,:] u, N, coeff, long i):
    h = 1/(N-1)
    b1, c1, d1 = coeff
    w = np.zeros_like(u[i, :])
    cdef long j
    for j in range(1, N-1):
        w[j] = c1[i, j]*(u[i+1, j+1] + u[i-1, j-1] - u[i-1, j+1] - u[i+1, j-1])/4 +\
                    (b1[i, j] + d1[i, j]*h/2)*u[i+1, j] + (b1[i, j] - d1[i, j]*h/2)*u[i-1, j]
        w[j] /= h**2
    return w

def xBlack_correction_1d(double [:,:] u, N, coeff):
    h = 1/(N-1)
    a1, c1, e1 = coeff
    w = np.zeros_like(u)
    cdef long i, j
    for j in range(int((N-1)/2)-1):
        j = 2*j + 2
        for i in range(1, N-1):
            w[i, j] = c1[i, j]*(u[i+1, j+1] + u[i-1, j-1] - u[i-1, j+1] - u[i+1, j-1])/4 +\
                        (a1[i, j] + e1[i, j]*h/2)*u[i, j+1] + (a1[i, j] - e1[i, j]*h/2)*u[i, j-1]
            w[i,j] /= h**2
    return w

def yRed_correction_1d(double [:,:] u, N, coeff):
    h = 1/(N-1)
    b1, c1, d1 = coeff
    w = np.zeros_like(u)
    cdef long i, j
    for i in range(int((N-1)/2)):
        i = 2*i + 1
        for j in range(1, N-1):
            w[i, j] = c1[i, j]*(u[i+1, j+1] + u[i-1, j-1] - u[i-1, j+1] - u[i+1, j-1])/4 +\
                        (b1[i, j] + d1[i, j]*h/2)*u[i+1, j] + (b1[i, j] - d1[i, j]*h/2)*u[i-1, j]
            w[i,j] /= h**2
    return w

def yBlack_correction_1d(double [:,:] u, N, coeff):
    h = 1/(N-1)
    b1, c1, d1 = coeff
    w = np.zeros_like(u)
    cdef long i, j
    for i in range(int((N-1)/2)-1):
        i = 2*i + 2
        for j in range(1, N-1):
            w[i, j] = c1[i, j]*(u[i+1, j+1] + u[i-1, j-1] - u[i-1, j+1] - u[i+1, j-1])/4 +\
                        (b1[i, j] + d1[i, j]*h/2)*u[i+1, j] + (b1[i, j] - d1[i, j]*h/2)*u[i-1, j]
            w[i,j] /= h**2
    return w

def bilinear_restriction(double [:,:] u, N):
    M = int((N-1)/2) + 1
    v = np.zeros((M, M))
    cdef long i, j
    for i in range(M):
        v[i, 0] = u[2*i, 0]
        v[i, M-1] = u[2*i, N-1]
        v[0, i] = u[0, 2*i]
        v[M-1, i] = u[N-1, 2*i]
    for i in range(1, M-1):
        for j in range(1, M-1):
            v[i, j] = u[2*i, 2*j]/4 + (u[2*i+1, 2*j] + u[2*i-1, 2*j] + u[2*i, 2*j+1] + u[2*i, 2*j-1])/8 +\
            (u[2*i+1, 2*j+1] + u[2*i-1, 2*j-1] + u[2*i-1, 2*j+1] + u[2*i+1, 2*j-1])/16
    return v

def bilinear_interpolation(double [:,:] u, double [:,:] v, M):
    N = 2*M - 1
    cdef long i, j
    for j in range(1, M-1):
        for i in range(1, M-1):
            v[2*i, 2*j] = u[i, j]
            v[2*i-1, 2*j] = (u[i, j] + u[i-1, j])/2
            if i == M-2:
                v[2*i+1, 2*j] = (u[i, j] + u[i+1, j])/2
    for i in range(1, N-1):
        for j in range(int((N-1)/2)):
                v[i, 2*j+1] = (v[i, 2*j] + v[i, 2*j+2])/2

def c_interpolation(double [:,:] u, double [:,:] v, M):
    N = 2*M - 1
    cdef long i, j
    for j in range(1, M-1):
        for i in range(1, M-1):
            v[2*i, 2*j] = u[i, j]
            if i != M-2:
                v[2*i+1, 2*j] = (- u[i-1, j] + 9*u[i, j] + 9*u[i+1, j] - u[i+2, j])/16
            if i == 1:
                v[2*i-1, 2*j] = (5*u[i-1, j] + 15*u[i, j] - 5*u[i+1, j] + u[i+2, j])/16
            if i == M-2:
                v[2*i+1, 2*j] = (u[i-2, j] - 5*u[i-1, j] + 15*u[i, j] + 5*u[i+1, j])/16
    for i in range(1, N-1):
        for j in range(int((N-1)/2)):
            if j == 0:
                v[i, 2*j+1] = (5*v[i, 2*j] + 15*v[i, 2*j+2] - 5*v[i, 2*j+4] + v[i, 2*j+6])/16
            if j == (int((N-1)/2) - 1):
                v[i, 2*j+1] = (v[i, 2*j-4] - 5*v[i, 2*j-2] + 15*v[i, 2*j] + 5*v[i, 2*j+2])/16
            if j != 0 and j != (int((N-1)/2) - 1):
                v[i, 2*j + 1] = (- v[i, 2*j-2] + 9*v[i, 2*j] + 9*v[i, 2*j+2] - v[i, 2*j+4])/16

def four_interpolation(double [:,:] u, double [:,:] v, M):
    N = 2*M - 1
    cdef long i, j
    for j in range(1, M-1):
        for i in range(1, M-1):
            v[2*i, 2*j] = u[i, j]
            if i > 2 and i < M-3:
                v[2*i+1, 2*j] = (- u[i-1, j] + 9*u[i, j] + 9*u[i+1, j] - u[i+2, j])/16
            if i == 2:
                v[2*i-1, 2*j] = (5*u[i-1, j] + 15*u[i, j] - 5*u[i+1, j] + u[i+2, j])/16
            if i == M-3:
                v[2*i+1, 2*j] = (u[i-2, j] - 5*u[i-1, j] + 15*u[i, j] + 5*u[i+1, j])/16
    for i in range(1, N-1):
        for j in range(int((N-1)/2)):
            if j == 0:
                v[i, 2*j+1] = (5*v[i, 2*j] + 15*v[i, 2*j+2] - 5*v[i, 2*j+4] + v[i, 2*j+6])/16
            if j == (int((N-1)/2) - 1):
                v[i, 2*j+1] = (v[i, 2*j-4] - 5*v[i, 2*j-2] + 15*v[i, 2*j] + 5*v[i, 2*j+2])/16
            if j != 0 and j != (int((N-1)/2) - 1):
                v[i, 2*j + 1] = (- v[i, 2*j-2] + 9*v[i, 2*j] + 9*v[i, 2*j+2] - v[i, 2*j+4])/16

def dx_forward(double [:,:] u, N):
    h = 1/(N-1)
    v = np.zeros_like(u)
    cdef long i, j
    for j in range(1, N-1):
        for i in range(1, N-1):
            v[i, j] = (u[i+1, j] - u[i, j])/h
    return v

def dx_backward(double [:,:] u, N):
    h = 1/(N-1)
    v = np.zeros_like(u)
    cdef long i, j
    for j in range(1, N-1):
        for i in range(1, N-1):
            v[i, j] = (u[i, j] - u[i-1, j])/h
    return v

def dx(double [:,:] u, N):
    h = 1/(N-1)
    v = np.zeros_like(u)
    cdef long i, j
    for j in range(1, N-1):
        for i in range(1, N-1):
            v[i, j] = (u[i+1, j] - u[i-1, j])/(2*h)
    return v

def dy_forward(double [:,:] u, N):
    h = 1/(N-1)
    v = np.zeros_like(u)
    cdef long i, j
    for i in range(1, N-1):
        for j in range(1, N-1):
            v[i, j] = (u[i, j+1] - u[i, j])/h
    return v

def dy_backward(double [:,:] u, N):
    h = 1/(N-1)
    v = np.zeros_like(u)
    cdef long i, j
    for i in range(1, N-1):
        for j in range(1, N-1):
            v[i, j] = (u[i, j] - u[i, j-1])/h
    return v

def dy(double [:,:] u, N):
    h = 1/(N-1)
    v = np.zeros_like(u)
    cdef long i, j
    for i in range(1, N-1):
        for j in range(1, N-1):
            v[i, j] = (u[i, j+1] - u[i, j-1])/(2*h)
    return v

def d2x(double [:,:] u, N):
    h = 1/(N-1)
    v = np.zeros_like(u)
    cdef long i, j
    for j in range(1, N-1):
        for i in range(1, N-1):
            v[i, j] = (u[i+1, j] + u[i-1, j] - 2*u[i, j])/h**2
    return v

def d2y(double [:,:] u, N):
    h = 1/(N-1)
    v = np.zeros_like(u)
    cdef long i, j
    for i in range(1, N-1):
        for j in range(1, N-1):
            v[i, j] = (u[i, j+1] + u[i, j-1] - 2*u[i, j])/h**2
    return v

def dxdy(double [:,:] u, N):
    h = 1/(N-1)
    v = np.copy(u)
    cdef long i, j
    for j in range(1, N-1):
        for i in range(1, N-1):
            v[i, j] = (u[i+1, j+1] + u[i-1, j-1] - u[i-1, j+1] - u[i+1, j-1])/(4*h**2)
    return v

def dx_with_boundary(double [:,:] u, N):
    h = 1/(N-1)
    v = np.zeros_like(u)
    cdef long i, j
    for j in range(N):
        for i in range(N):
            if i == 0:
                v[i, j] = -3*u[i, j]/(2*h) + 2*u[i+1, j]/h - u[i+2, j]/(2*h)
            if i == N-1:
                v[i, j] = 3*u[i, j]/(2*h) - 2*u[i-1, j]/h + u[i-2, j]/(2*h)
            if i != 0 and i != N-1:
                v[i, j] = (u[i+1, j] - u[i-1, j])/(2*h)
    return v

def dy_with_boundary(double [:,:] u, N):
    h = 1/(N-1)
    v = np.zeros_like(u)
    cdef long i, j
    for i in range(N):
        for j in range(N):
            if j == 0:
                v[i, j] = -3*u[i, j]/(2*h) + 2*u[i, j+1]/h - u[i, j+2]/(2*h)
            if j == N-1:
                v[i, j] = 3*u[i, j]/(2*h) - 2*u[i, j-1]/h + u[i, j-2]/(2*h)
            if j != 0 and j != N-1:
                v[i, j] = (u[i, j+1] - u[i, j-1])/(2*h)
    return v
