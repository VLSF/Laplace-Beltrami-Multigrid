import numpy as np

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
