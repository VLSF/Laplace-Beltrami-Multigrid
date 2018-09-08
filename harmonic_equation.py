from equation import equation
import numpy as np
from geometry import TI

class harmonic_equation(equation):
    def __init__(self, coeff, rhs, n_equations, bc=None, exact=None, l_coeff=None, continuation=None):
        equation.__init__(self, coeff, rhs, n_equations, bc, exact, l_coeff, continuation)

    def produce_initial_guess(self, J):
        N = 2 ** J + 1
        current = np.zeros((2, N, N))
        current = self.bc(current)
        return TI(current)
