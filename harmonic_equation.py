from equation import equation
import numpy as np
from geometry import TI

class harmonic_equation(equation):
    """
    This is the basic class for all harmonic equations. See `class equation` for
    description of the constructor. The only difference that this class can not be
    used for the scalar equation.
    """
    def __init__(self, coeff, rhs, n_equations, bc=None, exact=None, l_coeff=None, continuation=None):
        equation.__init__(self, coeff, rhs, n_equations, bc, exact, l_coeff, continuation)

    def produce_initial_guess(self, J):
        '''
        The method produce the array of the shape `(2, 2**J+1, 2**J+1)`. The boundary
        is initialized with the chosen bcs and transfinite interpolation gives all
        inner values. Do not use in case of non-smooth boundary conditions!

        Parameters
        ----------
        J: int
        The level of the resolution.

        Returns
        -------
        res: ndarray
        [res1, res2]
        '''
        N = 2 ** J + 1
        current = np.zeros((2, N, N))
        current = self.bc(current)
        return TI(current)
