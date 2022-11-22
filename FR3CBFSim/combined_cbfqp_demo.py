import argparse
import pickle

import numpy as np
import pybullet as p
from FR3Env.fr3_env import FR3Sim

from FR3CBFSim import getDataPath
from FR3CBFSim.controllers.combined_cbfqp import CombinedCBFQP
from FR3CBFSim.controllers.utils import get_R_end_from_start


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

    p_ends = [
        np.array([[0.4], [0.4], [0.2]]),
        np.array([[0.4], [-0.4], [0.2]]),
        np.array([[0.3], [0.0], [0.5]]),
    ]
    p_end_id = 0

    dt = 1 / 240

    env = FR3Sim(render_mode="human", record_path=args.recordPath)
    p.setTimeStep(dt)
    controller = CombinedCBFQP()

    # Load wall
    wall_urdf_path = getDataPath() + "/robots/wall.urdf"
    p.loadURDF(wall_urdf_path, useFixedBase=True)

    info = env.reset()

    p_end = p_ends[p_end_id]
    p_end_id = (p_end_id + 1) % len(p_ends)

    # get initial rotation and position
    R_start, _p_start = info["R_EE"], info["P_EE"]
    p_start = _p_start[:, np.newaxis]

    R_end = get_R_end_from_start(0, 0, 0, R_start)

    controller.start(p_start, p_end, R_start, R_end, 30.0)
    q_nominal = env.q_nominal[:, np.newaxis]

    # Data storage
    history = []

    for i in range(args.iterationNum):
        # Get end-effector position
        p_current = info["P_EE"][:, np.newaxis]
        # Get end-effector orientation
        R_current = info["R_EE"]

        if i == 0:
            _dt = 0
        else:
            _dt = dt

        τ, sol_info = controller.update(p_current, R_current, _dt, q_nominal, info)

        if i % 10000 == 0 and i > 1:
            p_end = p_ends[p_end_id]
            p_end_id = (p_end_id + 1) % len(p_ends)

            # get initial rotation and position
            R_start, _p_start = info["R_EE"], info["P_EE"]
            p_start = _p_start[:, np.newaxis]
            R_end = get_R_end_from_start(0, 0, 0, R_start)

            controller.start(p_start, p_end, R_start, R_end, 30.0)

        if i % 500 == 0:
            print("Iter {:.2e} \t error: {:.2e}".format(i, sol_info["error"]))

        # Send joint commands to motor
        info = env.step(τ)

        # Store data for plotting
        info["cbf"] = sol_info["cbf"]
        info["τ"] = τ
        history.append(info)

    env.close()

    if args.dataPath is not None:
        with open(args.dataPath, "wb") as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
