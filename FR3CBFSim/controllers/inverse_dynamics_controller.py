from pdb import set_trace

import numpy as np
import numpy.linalg as LA
from scipy.spatial.transform import Rotation as R

from FR3CBFSim.cbfs import box_cbf_ee
from FR3CBFSim.controllers.utils import axis_angle_from_rot_mat, sinusoid_path_gen


class InverseDynamicsController:
    START = 1
    UPDATE = 2
    WAIT = 3

    def __init__(self) -> None:
        self.p_start = None
        self.R_start = None
        self.movement_duration = None
        self.clock = 0.0

        self.status = None

    def start(self, p_start, R_start):
        # Set status to START
        self.status = self.START

        # Start time counter
        self.clock = 0.0

        # Get p_start, R_start
        self.p_start = p_start
        self.R_start = R_start

        # Set status to UPDATE
        self.status = self.UPDATE

    def update(self, p_current, R_current, duration, info):
        self.clock += duration
        q = info["q"]
        dq = info["dq"]
        pinv_jac = info["pJ_EE"]
        jacobian = info["J_EE"]

        # Generate path targets for current time step
        path_targets = sinusoid_path_gen(
            self.clock,
            self.p_start,
            self.R_start,
            T=20.0,
        )

        # Error rotation matrix
        R_err = path_targets["R_target"] @ R_current.T

        # Orientation error in axis-angle form
        rotvec_err = R.from_matrix(R_err).as_rotvec()

        # Compute EE position error
        p_error = np.zeros((6, 1))
        p_error[:3] = path_targets["p_target"] - p_current
        p_error[3:] = rotvec_err[:, np.newaxis]

        # Compute EE velocity error
        dp_target = np.vstack(
            (path_targets["v_target"], path_targets["ω_target"][:, np.newaxis])
        )
        dp_measured = jacobian @ dq[:, np.newaxis]
        dp_error = dp_target - dp_measured

        # Compute target EE acceleration
        ddp_target = np.vstack(
            (path_targets["a_target"], path_targets["dω_target"][:, np.newaxis])
        )

        # Compute CBF
        cbf, dcbf_dq = box_cbf_ee(
            info, d_max=0.3, alpha=10.0, n_vec=np.array([[0.0], [1.0], [0.0]])
        )

        # Solve for τ
        params = {
            "Jacobian": jacobian,
            "dJ": info["dJ_EE"],
            "M(q)": info["M(q)"],
            "nle": info["nle"],
            "q_measured": q[:, np.newaxis],
            "dq_measured": dq[:, np.newaxis],
            "p_error": p_error,
            "dp_error": dp_error,
            "ddp_target": ddp_target,
            "nullspace_proj": np.eye(9) - pinv_jac @ jacobian,
            "Kp": 2.0 * np.eye(6),
            "Kd": 0.2 * np.eye(6),
            "h": cbf,
            "∂h/∂x": dcbf_dq,
            "α": 1.0,
            "f(x)": info["f(x)"],
            "g(x)": info["g(x)"],
        }

        ddq_des = pinv_jac @ (
            ddp_target
            + 10 * p_error
            + 0.1 * dp_error
            - params["dJ"] @ params["dq_measured"]
        )

        τ = params["M(q)"] @ ddq_des + params["nle"][:, np.newaxis]
        τ = τ[:, 0]

        # Control for the fingers
        τ[-1] = 1.0 * (0.01 - q[-1]) + 0.1 * (0 - dq[-1])
        τ[-2] = 1.0 * (0.01 - q[-2]) + 0.1 * (0 - dq[-2])

        # Compute end-effector pose error
        _R_err = self.R_start @ R_current.T
        _rotvec_err = R.from_matrix(_R_err).as_rotvec()
        error_vec = np.concatenate(
            [_rotvec_err, (path_targets["p_target"] - p_current)[:, 0]]
        )
        error = LA.norm(error_vec)

        # sol_info for logging
        sol_info = {"error": error, "cbf": cbf}

        return τ, sol_info
