import numpy as np
import proxsuite


class KinematicsControllerSolver:
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

    def compute_params(self, params):
        H = (
            2 * params["Jacobian"].T @ params["Jacobian"]
            + 2 * params["nullspace_proj"].T @ params["nullspace_proj"]
        )

        a = params["Kp"] @ params["p_error"] + params["dp_target"]

        g = (
            -2
            * (
                a.T @ params["Jacobian"]
                + params["dq_nominal"].T
                @ params["nullspace_proj"].T
                @ params["nullspace_proj"]
            )[0, :]
        )

        if params["cbf_type"] == "x":
            C = -params["Jacobian"][:1, :]
            x_wall = 0.5
            lb = np.array([[-1 * (x_wall - params["p_current"][0, 0])]])
            ub = np.array([[np.inf]])

        elif params["cbf_type"] == "z":
            C = -params["Jacobian"][2:3, :]
            z_wall = 0.6
            lb = np.array([[-1 * (z_wall - params["p_current"][2, 0])]])
            ub = np.array([[np.inf]])

        return H, g, C, lb, ub
