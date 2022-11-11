import numpy as np
import numpy.linalg as LA
from scipy.spatial.transform import Rotation as R

from FR3CBFSim.cbfs import box_cbf_ee
from FR3CBFSim.controllers.combined_cbfqp_solver import CombinedCBFQPSolver
from FR3CBFSim.controllers.utils import axis_angle_from_rot_mat, smooth_trig_path_gen


class CombinedCBFQP:
    START = 1
    UPDATE = 2
    WAIT = 3

    def __init__(self) -> None:
        self.p_start = None
        self.p_end = None
        self.R_start = None
        self.R_end = None
        self.movement_duration = None
        self.clock = 0.0
        self.qp_solver = CombinedCBFQPSolver(2 * 9)  # 9 represents there are 9 joints

        self.status = None

    def start(self, p_start, p_end, R_start, R_end, movement_duration):
        # Set status to START
        self.status = self.START

        # Start time counter
        self.clock = 0.0

        # Get p_start, R_start
        self.p_start = p_start
        self.R_start = R_start

        # Set p_end, R_end
        self.p_end = p_end
        self.R_end = R_end
        self.movement_duration = movement_duration

        # Compute R_error, ω_error, θ_error
        self.R_error = R_end @ R_start.T
        self.ω_error, self.θ_error = axis_angle_from_rot_mat(self.R_error)

        # Set status to UPDATE
        self.status = self.UPDATE

    def update(self, p_current, R_current, duration, q_nominal, info):
        self.clock += duration
        q = info["q"]
        dq = info["dq"]
        pinv_jac = info["pJ_EE"]
        jacobian = info["J_EE"]

        # Compute end-effector pose error
        _R_err = self.R_end @ R_current.T
        _rotvec_err = R.from_matrix(_R_err).as_rotvec()
        error_vec = np.concatenate([_rotvec_err, (self.p_end - p_current)[:, 0]])
        error = LA.norm(error_vec)

        if error <= 1e-3:
            self.status = self.WAIT

        if self.status == self.WAIT:
            path_targets = {}
            path_targets["p_target"] = self.p_end
            path_targets["R_target"] = self.R_end
            path_targets["v_target"] = np.zeros((3, 1))
            path_targets["ω_target"] = np.zeros((3,))
            path_targets["a_target"] = np.zeros((3, 1))
            path_targets["dω_target"] = np.zeros((3,))
        elif self.status == self.UPDATE:
            # Generate path targets for current time step
            path_targets = smooth_trig_path_gen(
                self.clock,
                self.p_start,
                self.p_end,
                self.R_start,
                self.ω_error,
                self.θ_error,
                T=30.0,
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

        # Compute joint-centering joint acceleration
        ddq_nominal = 0.5 * (q_nominal - q[:, np.newaxis]) - 0.2 * dq[:, np.newaxis]

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
            "ddq_nominal": ddq_nominal,
            "Kp": 2.0 * np.eye(6),
            "Kd": 0.2 * np.eye(6),
            "h": cbf,
            "∂h/∂x": dcbf_dq,
            "α": 1.0,
            "f(x)": info["f(x)"],
            "g(x)": info["g(x)"],
        }

        self.qp_solver.solve(params)
        τ = self.qp_solver.qp.results.x[9:]

        # Control for the fingers
        τ[-1] = 1.0 * (0.01 - q[-1]) + 0.1 * (0 - dq[-1])
        τ[-2] = 1.0 * (0.01 - q[-2]) + 0.1 * (0 - dq[-2])

        # sol_info for logging
        sol_info = {"error": error, "cbf": cbf}

        return τ, sol_info
