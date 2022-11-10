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
        self.H, self.g, self.C, self.lb, self.ub = self.compute_params(params)

        if not self.initialized:
            self.qp.init(H=self.H, g=self.g, C=self.C, l=self.lb, u=self.ub)
            self.qp.settings.eps_abs = 1.0e-6
            self.initialized = True
        else:
            self.qp.update(H=self.H, g=self.g, C=self.C, l=self.lb, u=self.ub)

        self.qp.solve()

        # bool for cbf constraint satisfaction
        self.cbf_constraint_satisfied = (
            self.lb <= self.C @ self.qp.results.x[:, np.newaxis]
        )[0, 0]

    def compute_params(self, params):
        H = 2 * np.eye(self.n)
        g = -2 * params["u_ref"][:, 0]

        # dt = 1 / 1000
        # epsilon = 0.0

        # C = np.vstack(
        #     (params["∂h/∂x"] @ params["g(x)"], params["∂h/∂x"] @ params["g(x)"] * dt)
        # )

        # lb = np.vstack(
        #     (
        #         -params["α"] * params["h"] - params["∂h/∂x"] @ params["f(x)"],
        #         epsilon - params["h"] - params["∂h/∂x"] @ params["f(x)"] * dt,
        #     )
        # )
        # ub = np.array([[np.inf], [np.inf]])

        C = params["∂h/∂x"] @ params["g(x)"]
        lb = 5e-2 - params["α"] * params["h"] - params["∂h/∂x"] @ params["f(x)"]
        ub = np.array([[np.inf]])

        return H, g, C, lb, ub
