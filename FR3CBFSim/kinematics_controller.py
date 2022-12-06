import copy
from pdb import set_trace

import numpy as np
import pybullet as p
from FR3Env.fr3_env import FR3Sim
from scipy.spatial.transform import Rotation as R

from FR3CBFSim import getDataPath
from FR3CBFSim.controllers.kinematics_controller_solver import (
    KinematicsControllerSolver,
)
from FR3CBFSim.controllers.utils import (
    axis_angle_from_rot_mat,
    get_R_end_from_start,
    smooth_trig_path_gen,
)


def main():
    dt = 1 / 1000

    # create environment
    env = FR3Sim(render_mode="human", record_path=None)
    p.setTimeStep(dt)

    # define solver
    solver = KinematicsControllerSolver(9)

    # reset environment
    info = env.reset()

    # get initial rotation and position
    R_start, _p_start = info["R_EE"], info["P_EE"]
    p_start = _p_start[:, np.newaxis]

    # get target rotation and position
    p_end = np.array([[0.4], [0], [0.8]])
    R_end = get_R_end_from_start(0, -90, 0, R_start)
    movement_duration = 10.0

    # Compute R_error, ω_error, θ_error
    R_error = R_end @ R_start.T
    ω_error, θ_error = axis_angle_from_rot_mat(R_error)

    # Data storage
    history = []

    # Initialize time
    t = 0.0

    cbf_type = "x"

    for i in range(100000):
        t += dt

        # Get data from info
        q = info["q"]
        dq = info["dq"]
        pinv_jac = info["pJ_EE"]
        jacobian = info["J_EE"]

        # Get end-effector position
        p_current = info["P_EE"][:, np.newaxis]

        # Get end-effector orientation
        R_current = info["R_EE"]

        path_targets = smooth_trig_path_gen(
            t, p_start, p_end, R_start, ω_error, θ_error, T=movement_duration
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

        # Get gravitational vector
        G = info["G"][:, np.newaxis]

        # Compute joint-centering joint acceleration
        dq_nominal = 0.5 * (env.q_nominal[:, np.newaxis] - q[:, np.newaxis])

        params = {
            "Jacobian": jacobian,
            "p_error": p_error,
            "p_current": p_current,
            "dp_target": dp_target,
            "Kp": 2.0 * np.eye(6),
            "dq_nominal": dq_nominal,
            "nullspace_proj": np.eye(9) - pinv_jac @ jacobian,
            "cbf_type": cbf_type,
        }

        solver.solve(params)
        dq_target = solver.qp.results.x

        τ = (
            1.0 * (dq_target[:, np.newaxis] - dq[:, np.newaxis])
            + G
            - 0.5 * dq[:, np.newaxis]
        )

        # Send joint commands to motor
        info = env.step(τ)

        if i == 20000:
            p_start = copy.deepcopy(p_end)
            R_start = copy.deepcopy(R_end)

            # get target rotation and position
            p_end = np.array([[0.7], [0], [0.5]])
            R_end = get_R_end_from_start(0, 1e-6, 0, R_start)
            movement_duration = 20.0

            # Compute R_error, ω_error, θ_error
            R_error = R_end @ R_start.T
            ω_error, θ_error = axis_angle_from_rot_mat(R_error)

            # Reinitialize time
            t = 0.0

        if i == 80000:
            p_start = p_current
            R_start = R_current

            # get target rotation and position
            p_end = np.array([[0.7], [0], [0.5]])
            R_end = get_R_end_from_start(0, 1e-6, 0, R_start)
            movement_duration = 20.0

            # Compute R_error, ω_error, θ_error
            R_error = R_end @ R_start.T
            ω_error, θ_error = axis_angle_from_rot_mat(R_error)

            # Reinitialize time
            t = 0.0

            cbf_type = "z"

        print(f"x: {p_current[0, 0]}")


if __name__ == "__main__":
    main()
