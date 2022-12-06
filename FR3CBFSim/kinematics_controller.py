import argparse
import copy
import pickle

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--recordPath",
        help="path where the recording is saved",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-i",
        "--iterationNum",
        help="number of iterations of the simulation",
        type=int,
        default=100000,
    )
    parser.add_argument(
        "-d",
        "--dataPath",
        help="path where the data is saved",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    dt = 1 / 1000

    # create environment
    env = FR3Sim(render_mode="human", record_path=args.recordPath)
    p.setTimeStep(dt)

    # Load wall
    wall_urdf_path = getDataPath() + "/robots/hole_in_wall.urdf"
    p.loadURDF(wall_urdf_path, useFixedBase=True)

    # define solver
    solver = KinematicsControllerSolver(9)

    # reset environment
    info = env.reset(cameraDistance=1.8, cameraYaw=-12.0, cameraPitch=-26.0)

    # get initial rotation and position
    R_start, _p_start = info["R_EE"], info["P_EE"]
    p_start = _p_start[:, np.newaxis]

    # get target rotation and position
    p_end = np.array([[0.4], [0], [0.8]])
    R_end = get_R_end_from_start(0, -90, 0, R_start)
    movement_duration = 10.0

    # compute R_error, ω_error, θ_error
    R_error = R_end @ R_start.T
    ω_error, θ_error = axis_angle_from_rot_mat(R_error)

    # data storage
    history = []

    # initialize time
    t = 0.0

    # set initial CBF type
    cbf_type = "x"

    for i in range(args.iterationNum):
        t += dt

        # get data from info
        q = info["q"]
        dq = info["dq"]
        pinv_jac = info["pJ_EE"]
        jacobian = info["J_EE"]

        # get end-effector position
        p_current = info["P_EE"][:, np.newaxis]

        # get end-effector orientation
        R_current = info["R_EE"]

        path_targets = smooth_trig_path_gen(
            t, p_start, p_end, R_start, ω_error, θ_error, T=movement_duration
        )

        # compute error rotation matrix
        R_err = path_targets["R_target"] @ R_current.T

        # compute orientation error in axis-angle form
        rotvec_err = R.from_matrix(R_err).as_rotvec()

        # compute EE position error
        p_error = np.zeros((6, 1))
        p_error[:3] = path_targets["p_target"] - p_current
        p_error[3:] = rotvec_err[:, np.newaxis]

        # compute EE velocity error
        dp_target = np.vstack(
            (path_targets["v_target"], path_targets["ω_target"][:, np.newaxis])
        )

        # get gravitational vector
        G = info["G"][:, np.newaxis]

        # compute joint-centering joint acceleration
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

        # send joint commands to motor
        info = env.step(τ)

        if i == 20000:
            p_start = copy.deepcopy(p_end)
            R_start = copy.deepcopy(R_end)

            # get target rotation and position
            p_end = np.array([[0.7], [0], [0.5]])
            R_end = get_R_end_from_start(0, 1e-6, 0, R_start)
            movement_duration = 20.0

            # compute R_error, ω_error, θ_error
            R_error = R_end @ R_start.T
            ω_error, θ_error = axis_angle_from_rot_mat(R_error)

            # reinitialize time
            t = 0.0

        if i == 40000:
            p_start = p_current
            R_start = R_current

            # get target rotation and position
            p_end = np.array([[0.8], [0], [0.5]])
            R_end = get_R_end_from_start(0, 1e-6, 0, R_start)
            movement_duration = 20.0

            # compute R_error, ω_error, θ_error
            R_error = R_end @ R_start.T
            ω_error, θ_error = axis_angle_from_rot_mat(R_error)

            # reinitialize time
            t = 0.0

            # change cbf type
            cbf_type = "z"

        if i == 60000:
            p_start = np.array([[0.8], [0], [0.5]])
            R_start = R_current

            # get target rotation and position
            p_end = np.array([[0.5], [0], [0.5]])
            R_end = get_R_end_from_start(0, 1e-6, 0, R_start)
            movement_duration = 20.0

            # compute R_error, ω_error, θ_error
            R_error = R_end @ R_start.T
            ω_error, θ_error = axis_angle_from_rot_mat(R_error)

            # reinitialize time
            t = 0.0

            # change cbf type
            cbf_type = "z"

        if i == 80000:
            p_start = np.array([[0.5], [0], [0.5]])
            R_start = R_current

            # get target rotation and position
            p_end = np.array([[0.3], [0], [0.5]])
            R_end = get_R_end_from_start(0, 90, 0, R_start)
            movement_duration = 20.0

            # compute R_error, ω_error, θ_error
            R_error = R_end @ R_start.T
            ω_error, θ_error = axis_angle_from_rot_mat(R_error)

            # reinitialize time
            t = 0.0

            # change cbf type
            cbf_type = "x"

        if i % 500 == 0:
            print("Iter {:.2e} \t x: {:.2e}".format(i, p_current[0, 0]))

        info["τ"] = τ
        history.append(info)

        # For camera calibration
        # camera_info = p.getDebugVisualizerCamera()
        # print(camera_info[-2], camera_info[-4], camera_info[-3], camera_info[-1])

    if args.dataPath is not None:
        with open(args.dataPath, "wb") as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
