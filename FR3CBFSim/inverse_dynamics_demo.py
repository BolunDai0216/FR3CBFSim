import argparse
import pickle

import numpy as np
import pybullet as p
from FR3Env.fr3_env import FR3Sim

from FR3CBFSim import getDataPath
from FR3CBFSim.controllers.inverse_dynamics_controller import InverseDynamicsController


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

    # create environment
    env = FR3Sim(render_mode="human", record_path=args.recordPath)

    # create controller
    controller = InverseDynamicsController()

    # reset environment
    info = env.reset()

    # get initial rotation and position
    R_start, _p_start = info["R_EE"], info["P_EE"]
    p_start = _p_start[:, np.newaxis]

    # initialize controller
    controller.start(p_start, R_start)

    # Data storage
    history = []

    for i in range(args.iterationNum):
        # Get end-effector position
        p_current = info["P_EE"][:, np.newaxis]
        # Get end-effector orientation
        R_current = info["R_EE"]

        if i == 0:
            dt = 0
        else:
            dt = 1 / 240

        τ, sol_info = controller.update(p_current, R_current, dt, info)

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
