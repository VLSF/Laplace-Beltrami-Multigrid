from solver import solver
import solver_tools as st

class FAS(solver):
    def __init__(self, pre_smoother, pre_n, post_smoother, post_n,
                restriction, interpolation, coarse_solver, J_min, tol=None, verbose=False):
        self._pre_smoother = pre_smoother
        self._post_smoother = post_smoother
        self._coarse_solver = coarse_solver
        self.J_min = J_min
        self.pre_n = pre_n
        self.post_n = post_n
        self._restriction = restriction
        self._interpolation = interpolation
        self.tol = tol
        self.verbose = verbose
        params = (self._pre_smoother, self.pre_n, self._post_smoother, self.post_n,
                            self._restriction, self._interpolation, self._coarse_solver, self.J_min)
        self.smoother = lambda equation, current, rhs: st.FAS(equation, current, rhs, *params)

    @property
    def pre_smoother(self):
        return self._pre_smoother

    @pre_smoother.setter
    def pre_smoother(self, smoother):
        self._pre_smoother = smoother
        self.refresh_parameters()

    @property
    def post_smoother(self):
        return self._post_smoother

    @post_smoother.setter
    def post_smoother(self, smoother):
        self._post_smoother = smoother
        self.refresh_parameters()

    @property
    def coarse_solver(self):
        return self._coarse_solver

    @coarse_solver.setter
    def coarse_solver(self, solver):
        self._coarse_solver = solver
        self.refresh_parameters()

    @property
    def restriction(self):
        return self._restriction

    @restriction.setter
    def restriction(self, frestriction):
        self._restriction = frestriction
        self.refresh_parameters()

    @property
    def interpolation(self):
        return self._interpolation

    @interpolation.setter
    def interpolation(self, finterpolation):
        self._interpolation = finterpolation
        self.refresh_parameters()

    def refresh_parameters(self):
        params = (self._pre_smoother, self.pre_n, self._post_smoother, self.post_n, \
                            self._restriction, self._interpolation, self._coarse_solver, self.J_min)
        self.smoother = lambda equation, current, rhs: st.FAS(equation, current, rhs, *params)
