import pickle

import numpy as np
import pinocchio as pin
import pybullet as p
from FR3Env.controller.waypoint_controller_hierarchical_proxqp import WaypointController
from FR3Env.fr3_env import FR3Sim

from FR3CBFSim import getDataPath
from FR3CBFSim.cbfs import box_cbf_ee
from FR3CBFSim.controllers.cbfqp import CBFQP
from FR3CBFSim.controllers.utils import get_R_end_from_start


def main():
    recordPath = None
    iterationNum = 100000

    p_ends = [
        np.array([[0.4], [0.4], [0.2]]),
        np.array([[0.4], [-0.4], [0.2]]),
        np.array([[0.3], [0.0], [0.5]]),
    ]
    p_end_id = 0

    env = FR3Sim(render_mode="human", record_path=recordPath)

    # Load wall
    wall_urdf_path = getDataPath() + "/robots/wall.urdf"
    p.loadURDF(wall_urdf_path, useFixedBase=True)

    controller = WaypointController()
    cbfqp_solver = CBFQP(9)

    info = env.reset()

    p_end = p_ends[p_end_id]
    p_end_id = (p_end_id + 1) % len(p_ends)

    # get initial rotation and position
    q, dq, R_start, _p_start = info["q"], info["dq"], info["R_EE"], info["P_EE"]
    p_start = _p_start[:, np.newaxis]
    __R_start = info["R_EE"]
    R_end = get_R_end_from_start(0, 0, 0, R_start)

    controller.start(p_start, p_end, R_start, R_end, 30.0)
    q_min = env.observation_space.low[:9][:, np.newaxis]
    q_max = env.observation_space.high[:9][:, np.newaxis]
    q_nominal = env.q_nominal[:, np.newaxis]

    # Data storage
    history = []

    for i in range(iterationNum):
        # Get end-effector position
        p_current = info["P_EE"][:, np.newaxis]

        # Get end-effector orientation
        R_current = info["R_EE"]

        # Get frame ID for grasp target
        jacobian_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED

        # Get Jacobian from grasp target frame
        # preprocessing is done in get_state_update_pinocchio()
        jacobian = env.robot.getFrameJacobian(env.EE_FRAME_ID, jacobian_frame)

        # Get pseudo-inverse of frame Jacobian
        pinv_jac = np.linalg.pinv(jacobian)

        # Get gravitational vector
        G = info["G"][:, np.newaxis]

        if i == 0:
            dt = 0
        else:
            dt = 1 / 240

        q_target, error = controller.update(
            q,
            dq,
            p_current,
            R_current,
            pinv_jac,
            jacobian,
            G,
            dt,
            q_min,
            q_max,
            q_nominal,
        )
        if i % (240 * 30) == 0 and i > 1:
            p_end = p_ends[p_end_id]
            p_end_id = (p_end_id + 1) % len(p_ends)

            # get initial rotation and position
            dq, R_start, _p_start = info["dq"], info["R_EE"], info["P_EE"]
            p_start = _p_start[:, np.newaxis]
            R_end = get_R_end_from_start(0, 0, 0, __R_start)

            controller.start(p_start, p_end, R_start, R_end, 30.0)

        if i % 500 == 0:
            print("Iter {:.2e} \t error: {:.2e}".format(i, error))

        # Compute controller
        Δq = (q_target - q)[:, np.newaxis]
        Kp = 10 * np.eye(9)
        τ = Kp @ Δq - 1.0 * dq[:, np.newaxis] + G

        # CBFQP Filter
        cbf, dcbf_dq = box_cbf_ee(
            info, d_max=0.3, alpha=10.0, n_vec=np.array([[0.0], [1.0], [0.0]])
        )

        cbfqp_params = {
            "u_ref": τ,
            "h": cbf,
            "∂h/∂x": dcbf_dq,
            "α": 1.0,
            "f(x)": info["f(x)"],
            "g(x)": info["g(x)"],
        }

        cbfqp_solver.solve(cbfqp_params)
        _τ_cbf = cbfqp_solver.qp.results.x

        dh = dcbf_dq @ (info["f(x)"] + info["g(x)"] @ _τ_cbf[:, np.newaxis])

        # Set control for the two fingers to zero
        _τ_cbf[-1] = 1.0 * (0.01 - q[-1]) + 0.1 * (0 - dq[-1])
        _τ_cbf[-2] = 1.0 * (0.01 - q[-2]) + 0.1 * (0 - dq[-2])
        τ_cbf = _τ_cbf[:, np.newaxis]

        # Send joint commands to motor
        info = env.step(τ_cbf)

        q, dq = info["q"], info["dq"]
        info["cbf"] = cbf
        info["dcbf"] = dh
        info["τ_cbf"] = τ_cbf
        history.append(info)

    env.close()

    with open("data/y_limit.pickle", "wb") as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
