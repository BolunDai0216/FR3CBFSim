from pdb import set_trace

import numpy as np
import proxsuite


class CombinedQPSolver:
    def __init__(self, n):
        self.n = n
        self.n_eq = int(n / 2)
        self.n_ieq = 0
        self.qp = proxsuite.proxqp.dense.QP(self.n, self.n_eq, self.n_ieq)
        self.initialized = False

    def solve(self, params):
        self.H, self.g, self.A, self.b, self.C, self.lb, self.ub = self.compute_params(
            params
        )

        if not self.initialized:
            self.qp.init(H=self.H, g=self.g, A=self.A, b=self.b)
            self.qp.settings.eps_abs = 1.0e-6
            self.initialized = True
        else:
            self.qp.update(H=self.H, g=self.g, A=self.A, b=self.b)

        self.qp.solve()

    def compute_params(self, params):
        H = np.zeros((self.n, self.n))
        H[:9, :9] = (
            2 * params["Jacobian"].T @ params["Jacobian"]
            + 2 * params["nullspace_proj"].T @ params["nullspace_proj"]
        )

        a = (
            params["ddp_target"]
            + params["Kp"] @ params["p_error"]
            + params["Kd"] @ params["dp_error"]
            - params["dJ"] @ params["dq_measured"]
        )

        g = np.zeros((self.n))
        g[:9] = (
            -2
            * (
                a.T @ params["Jacobian"]
                + params["ddq_nominal"].T
                @ params["nullspace_proj"].T
                @ params["nullspace_proj"]
            )[0, :]
        )

        A = np.hstack((params["M(q)"], -np.eye(9)))
        b = -params["nle"]

        C, lb, ub = 0, 0, 0

        return H, g, A, b, C, lb, ub
