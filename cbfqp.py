import numpy as np
import proxsuite


class CBFQP:
    def __init__(self, n):
        self.n = n
        self.n_eq = 0
        self.n_ieq = 1
        self.qp = proxsuite.proxqp.dense.QP(self.n, self.n_eq, self.n_ieq)
        self.initialized = False

    def solve(self, params):
        H, g, A, b, C, l, u = self.compute_params(params)

        if not self.initialized:
            self.qp.init(H, g, A, b, C, l, u)
            self.qp.settings.eps_abs = 1.0e-6
            self.initialized = True
        else:
            self.qp.update(H, g, A, b, C, l, u)

        self.qp.solve()

    def compute_params(self, params):
        H = np.eye(self.n)
        g = -2 * params["u_ref"]

        A = np.zeros((self.n_eq, self.n))
        b = np.zeros((self.n_eq,))

        C = params["∂h/∂x"] @ params["g(x)"]
        lb = -params["α"] * params["h"] - params["∂h/∂x"] @ params["f(x)"]
        ub = np.inf

        return H, g, A, b, C, lb, ub
