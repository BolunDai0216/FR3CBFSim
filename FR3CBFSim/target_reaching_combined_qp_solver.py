import pickle

import numpy as np
import numpy.linalg as LA
from FR3Env.fr3_env import FR3Sim
from FR3Env.hello_world import axis_angle_from_rot_mat
from scipy.spatial.transform import Rotation as R

from FR3CBFSim.cbfs import box_cbf_ee
from FR3CBFSim.controllers.combined_qp_solver import CombinedQPSolver


def alpha_func(t, T=5.0):
    if t <= T:
        β = (np.pi / 4) * (1 - np.cos(np.pi * t / T))
        s = np.sin(np.pi * t / T)
        c = np.cos(np.pi * t / T)

        α = np.sin(β)
        dα = ((np.pi ** 2) / (4 * T)) * np.cos(β) * s
        ddα = ((np.pi ** 3) / (4 * T ** 2)) * c * np.cos(β) - (
            ((np.pi ** 4) / (16 * T ** 2)) * (s ** 2) * np.sin(β)
        )
    else:
        α = 1.0
        dα = 0.0
        ddα = 0.0

    return α, dα, ddα


def main():
    env = FR3Sim(render_mode="human")
    qp_solver = CombinedQPSolver(2 * 9)  # 9 represents there are 9 joints

    info = env.reset()

    p_end = np.array([[0.3], [0.4], [0.2]])

    # get initial rotation and position
    q, dq, R_start, _p_start = info["q"], info["dq"], info["R_EE"], info["P_EE"]
    p_start = _p_start[:, np.newaxis]

    # Get target orientation based on initial orientation
    _R_end = (
        R.from_euler("x", 0, degrees=True).as_matrix()
        @ R.from_euler("z", 0, degrees=True).as_matrix()
        @ R_start
    )
    R_end = R.from_matrix(_R_end).as_matrix()
    R_error = R_end @ R_start.T
    axis_error, angle_error = axis_angle_from_rot_mat(R_error)

    # Data storage
    history = []

    for i in range(7500):
        # Get simulation time
        sim_time = i * (1 / 240)

        # Compute α and dα
        alpha, dalpha, ddalpha = alpha_func(sim_time, T=30.0)

        # Compute p_target
        p_target = p_start + alpha * (p_end - p_start)

        # Compute v_target
        v_target = dalpha * (p_end - p_start)

        # Compute a_target
        a_target = ddalpha * (p_end - p_start)

        # Compute R_target
        theta_t = alpha * angle_error
        R_target = R.from_rotvec(axis_error * theta_t).as_matrix() @ R_start

        # Compute ω_target
        ω_target = dalpha * axis_error * angle_error

        # Compute dω_target
        dω_target = ddalpha * axis_error * angle_error

        # Get end-effector position
        p_current = info["P_EE"][:, np.newaxis]

        # Get end-effector orientation
        R_current = info["R_EE"]

        # Error rotation matrix
        R_err = R_target @ R_current.T

        # Orientation error in axis-angle form
        rotvec_err = R.from_matrix(R_err).as_rotvec()

        # Get Jacobian from grasp target frame
        jacobian = info["J_EE"]

        # Get pseudo-inverse of frame Jacobian
        pinv_jac = info["pJ_EE"]

        # Compute controller
        p_error = np.zeros((6, 1))
        p_error[:3] = p_target - p_current
        p_error[3:] = rotvec_err[:, np.newaxis]

        dp_target = np.vstack((v_target, ω_target[:, np.newaxis]))
        dp_measured = jacobian @ dq[:, np.newaxis]
        dp_error = dp_target - dp_measured

        ddp_target = np.vstack((a_target, dω_target[:, np.newaxis]))
        ddq_nominal = (
            0.5 * (env.q_nominal[:, np.newaxis] - q[:, np.newaxis])
            - 0.2 * dq[:, np.newaxis]
        )

        # Compute CBF
        cbf, dcbf_dq = box_cbf_ee(
            q, dq, info, d_max=-0.3, alpha=10.0, n_vec=np.array([[0.0], [0.0], [-1.0]])
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
            "q_min": env.observation_space.low[:9][:, np.newaxis],
            "q_max": env.observation_space.high[:9][:, np.newaxis],
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

        qp_solver.solve(params)
        τ = qp_solver.qp.results.x[9:]

        # Send joint commands to motor
        info = env.step(τ)
        q, dq = info["q"], info["dq"]

        # compute _rotvec_err
        _R_err = R_end @ R_current.T
        _rotvec_err = R.from_matrix(_R_err).as_rotvec()

        if i % 500 == 0:
            print(
                "Iter {:.2e} \t ǁeₒǁ₂: {:.2e} \t ǁeₚǁ₂: {:.2e}".format(
                    i, LA.norm(_rotvec_err), LA.norm(p_end - p_current)
                ),
            )

        info["cbf"] = cbf
        info["τ"] = τ
        history.append(info)

    env.close()

    with open("data/_z_limit.pickle", "wb") as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
