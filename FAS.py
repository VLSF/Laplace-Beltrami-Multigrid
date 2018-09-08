from solver import solver
import solver_tools as st

class FAS(solver):
    def __init__(self, pre_smoother, pre_n, post_smoother, post_n,
                restriction, interpolation, coarse_solver, J_min, tol=None, verbose=False):
        self.pre_smoother = pre_smoother
        self.post_smoother = post_smoother
        self.coarse_solver = coarse_solver
        self.J_min = J_min
        self.pre_n = pre_n
        self.post_n = post_n
        self.restriction = restriction
        self.interpolation = interpolation
        self.tol = tol
        self.verbose = verbose
        params = (self.pre_smoother, self.pre_n, self.post_smoother, self.post_n,
                            self.restriction, self.interpolation, self.coarse_solver, self.J_min)
        self.smoother = lambda equation, current, rhs: st.FAS(equation, current, rhs, *params)

    def refresh_parameters(self):
        params = (self.pre_smoother, self.pre_n, self.post_smoother, self.post_n, \
                            self.restriction, self.interpolation, self.coarse_solver, self.J_min)
        self.smoother = lambda equation, current, rhs: st.FAS(equation, current, rhs, *params)
