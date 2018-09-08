import numpy as np
import low_level_tools as llt
import equation_tools as et
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

class equation:
    """
    This is the basic class for all linear and nonlinear equations.
    It contains methods that allow to compute defects and residuals
    (with and without rhs), errors (in case exact solution is specified),
    an action of the operator, and produce initial guess with the correct bc.
    """

    def __init__(self, coeff, rhs, n_equations, bc=None, exact=None, l_coeff=None, continuation=None):
        """
        You are kindly asked to specify either the scalar equation

        a u_{yy} + b u_{xx} + c u_{xy} + d u_{x} + e u_{y} + f u = r

        or the system of two equations

        a11 u_{yy} + b11 u_{xx} + c11 u_{xy} + d11 u_{x} + e11 u_{y} + f11 u +
        a12 v_{yy} + b12 v_{xx} + c12 v_{xy} + d12 v_{x} + e12 v_{y} + f12 v = r1

        a21 u_{yy} + b21 u_{xx} + c21 u_{xy} + d21 u_{x} + e21 u_{y} + f21 u +
        a22 v_{yy} + b22 v_{xx} + c22 v_{xy} + d22 v_{x} + e22 v_{y} + f22 v = r2

        Parameters
        ----------
        coeff: callable object
        This function should take [u,] or [u, v] and return [[a, b, c, d, e, f],]
        or [[a11, ... , f11, a12, ..., f12], [a21, ... , f21, a22, ..., f22]].

        rhs: callable object
        The function should take [u,] or [u, v] and return [r1,] or [r1, r2]

        n_equations: int
        1 in case of the scalar equation and 2 in case of the system of two equation.

        bc: (optional) callable object
        The function should take [u,] or [u, v] and return array_like of the same
        shape but with boundary values adjusted accordingly chosen bc.

        exact: (optional) callable object
        The function takes [u,] or [u,v] and returns array_like of the same shape
        containing exact solution.

        l_coeff: (optional) callable object
        This function should take [u,] or [u, v] and return [[a, b, c, d, e, f],]
        or [[a11, ... , f11, a12, ..., f12], [a21, ... , f21, a22, ..., f22]] obtained
        from the operator linearization (for Newton's solvers).
        """
        self.rhs = rhs
        self.coefficients = coeff
        self.n_equations = n_equations
        if bc == None:
            self.bc = lambda x: x
        else:
            self.bc = bc
        if l_coeff == None:
            self.linear_coefficients = coeff
        else:
            self.linear_coefficients = l_coeff
        if exact == None:
            self.exact = lambda x: x
        else:
            self.exact = exact
        if continuation == None:
            self.continuation = lambda x: x
        else:
            self.continuation = continuation

    def defects(self, current):
        '''
        The method compute residual |b - Ax|_inf where b is the vector 'native'
        for the equation.

        Parameters
        ----------
        current: array_like
        The current guess: [u,] or [u, v].

        Returns
        -------
        res: array_like
        Either [d1,] or [d1, d2] where each d is double.
        '''
        current = self.bc(current)
        COEFF = self.coefficients(current)
        RHS = self.rhs(current)
        return et.defects(current, COEFF, RHS)

    def rhs_defects(self, current, rhs):
        '''
        The method compute defects |b - Ax|_inf where b is the external vector.

        Parameters
        ----------
        current: array_like
        The current guess: [u,] or [u, v].

        rhs: array_like
        The rhs: [r1,] or [r1, r2].

        Returns
        -------
        res: array_like
        Either [d1,] or [d1, d2] where each d is double.
        '''
        current = self.bc(current)
        COEFF = self.coefficients(current)
        return et.defects(current, COEFF, rhs)

    def residuals(self, current):
        '''
        The method compute residual b - Ax where b is the vector 'native' for the
        equation.

        Parameters
        ----------
        current: array_like
        The current guess: [u,] or [u, v].

        Returns
        -------
        res: array_like
        Either [res1,] or [res1, res2] shapes coincide with the shape of current.
        '''
        current = self.bc(current)
        COEFF = self.coefficients(current)
        RHS = self.rhs(current)
        return et.residuals(current, COEFF, RHS)

    def rhs_residuals(self, current, rhs):
        '''
        The method compute residual b - Ax where b is the external vector.

        Parameters
        ----------
        current: array_like
        The current guess: [u,] or [u, v].

        Returns
        -------
        res: array_like
        Either [res1,] or [res1, res2] shapes coincide with the shape of current.
        '''
        current = self.bc(current)
        COEFF = self.coefficients(current)
        return et.residuals(current, COEFF, rhs)

    def errors(self, current):
        '''
        The method compute |exact - current|_inf. In case exact is not providen
        the result is meaningless.

        Parameters
        ----------
        current: array_like
        The current guess: [u,] or [u, v].

        Returns
        -------
        res: array_like
        Either [e1,] or [e1, e2] where each e is of the double type.
        '''
        current = self.bc(current)
        EXACT = self.exact(current)
        return et.errors(current, EXACT)

    def operator(self, current):
        '''
        The method compute residual Ax i.e. the result of the application of the
        operator to the given vector.

        Parameters
        ----------
        current: array_like
        The current guess: [u,] or [u, v].

        Returns
        -------
        res: ndarray
        Either [res1,] or [res1, res2] shapes coincide with the shape of current.
        '''
        current = self.bc(current)
        COEFF = self.coefficients(current)
        return et.operator(current, COEFF)

    def produce_initial_guess(self, J):
        '''
        The method produce the array of the shape `(1, 2**J+1, 2**J+1)` or
        `(2, 2**J+1, 2**J+1)` with zero everywhere but the boundary. The boundary
        is initialized with the chosen bcs.

        Parameters
        ----------
        J: int
        The level of the resolution.

        Returns
        -------
        res: array_like
        Either [res1,] or [res1, res2].
        '''
        N = 2 ** J + 1
        current = np.zeros((self.n_equations, N, N))
        current = self.bc(current)
        return current

    def defects_conv_plot(self):
        '''
        Method is solely for tests. In case an exact solution is given it produce
        a convergence plot in jupyter notebook with inline matplotlib. It should have
        the -2 slope = - (the order of discretization). If it is not the case you
        have a typo somewhere.
        '''
        D1 = []
        D2 = []
        H = []
        for J in [3, 4, 5, 6, 7]:
            N = 2 ** J + 1
            h = 2 ** -J
            current = np.zeros((self.n_equations, N, N))
            current = self.exact(current)
            d = self.defects(current)
            D1.append(d[0])
            if self.n_equations == 2:
                D2.append(d[1])
            H.append(h)

        plt.plot(np.log10(H), np.log10(D1), label='defect 1')
        if self.n_equations == 2:
            plt.plot(np.log10(H), np.log10(D2), label='defect 2')
        plt.xlabel('$\log_{10} h$')
        plt.ylabel('$\log_{10} |d|_{\infty}$')
        plt.legend()

    def sparse_representation(self, J, bc='Default', rhs='Default', current=None, Rh=None):
        '''
        Compute the (sparse) matrix representation of the equation.

        Parameters
        ----------
        J: int
        The level of the resolution.

        bc: string
        If `Default` bc is taken from the `self.bc`, if `special` then the parameter
        `current` should contain bc.

        rhs: string
        If `Default` bc is taken from the `self.rhs`, if `special` then the parameter
        `RH` should contain rhs.

        current: None or ndarray
        Explained above.

        Rh: None or ndarray
        Explained above.

        Returns
        -------
        A: csr_matrix
        Scipy `csr_matrix` of linear coefficients of the equation. It means that
        (linear A) (exact solution of the linearized equation) = R.

        R: ndarray
        Right hand side modified by Dirichlet boundary conditions.
        '''
        M = 2 ** J + 1
        N = 2 ** J - 1
        h = 2 ** -J
        if self.n_equations == 1:
            if bc == 'Default':
                trial = [np.zeros((M, M)), ]
                trial = self.bc(trial)[0]
            if bc == 'Special':
                trial = current
            if rhs == 'Default':
                uR = self.rhs(trial)[0]
            if rhs == 'Special':
                uR = Rh[0]
            b1 = trial[0]
            a, b, c, d, e, f = self.linear_coefficients([np.zeros((M, M)), ])[0]
            r_mod = llt.modify_rhs(b1, M, [a, b, c, d, e, f])
            # r = uR*h**2 - r_mod
            r = uR - r_mod
            R = r.T[1:-1, 1:-1].reshape((-1,))
            data, row, col = llt.get_sparse_1d(0, 0, N, [a, b, c, d, e, f])
            A = csr_matrix((data, (row, col)), shape=(N ** 2, N ** 2))
        if self.n_equations == 2:
            if bc == 'Default':
                trial = [np.zeros((M, M)), np.zeros((M, M))]
                trial = self.bc(trial)
            if bc == 'Special':
                trial = current
            if rhs == 'Default':
                uR1, uR2 = self.rhs(trial)
            if rhs == 'Special':
                uR1, uR2 = Rh
            bc = trial
            coeff = self.linear_coefficients(trial)
            r_mod_11 = llt.modify_rhs(bc[0], M, coeff[0][:6])
            r_mod_12 = llt.modify_rhs(bc[1], M, coeff[0][6:])
            r_mod_21 = llt.modify_rhs(bc[0], M, coeff[1][:6])
            r_mod_22 = llt.modify_rhs(bc[1], M, coeff[1][6:])
            # R1 = (uR1*h**2 - r_mod_11 - r_mod_12).T[1:-1, 1:-1].reshape((-1,))
            # R2 = (uR2*h**2 - r_mod_21 - r_mod_22).T[1:-1, 1:-1].reshape((-1,))
            R1 = (uR1 - r_mod_11 - r_mod_12).T[1:-1, 1:-1].reshape((-1,))
            R2 = (uR2 - r_mod_21 - r_mod_22).T[1:-1, 1:-1].reshape((-1,))
            R = np.hstack((R1, R2))
            data11, row11, col11 = llt.get_sparse_1d(0, 0, N, coeff[0][:6])
            data12, row12, col12 = llt.get_sparse_1d(0, N ** 2, N, coeff[0][6:])
            data21, row21, col21 = llt.get_sparse_1d(N ** 2, 0, N, coeff[1][:6])
            data22, row22, col22 = llt.get_sparse_1d(N ** 2, N ** 2, N, coeff[1][6:])
            data = np.hstack((data11, data12, data21, data22))
            row = np.hstack((row11, row12, row21, row22))
            col = np.hstack((col11, col12, col21, col22))
            A = csr_matrix((data, (row, col)), shape=(2 * N ** 2, 2 * N ** 2))
        return A, R

    def dense_representation(self, J, bc='Default', rhs='Default', current=None, Rh=None):
        '''
        Compute the (dense) matrix representation of the equation.

        Look at the description of the `sparse_representation`.
        '''
        A, R = self.sparse_representation(J, bc, rhs, current, Rh)
        return A.toarray(), R
