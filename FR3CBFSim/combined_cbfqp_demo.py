import argparse

import numpy as np
from FR3Env.fr3_env import FR3Sim
from scipy.spatial.transform import Rotation as R

from FR3CBFSim.controllers.combined_cbfqp import CombinedCBFQP


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
    args = parser.parse_args()

    p_ends = [
        np.array([[0.4], [0.4], [0.2]]),
        np.array([[0.4], [-0.4], [0.2]]),
        np.array([[0.3], [0.0], [0.5]]),
    ]
    p_end_id = 0

    env = FR3Sim(render_mode="human", record_path=args.recordPath)
    controller = CombinedCBFQP()
    info = env.reset()

    p_end = p_ends[p_end_id]
    p_end_id = (p_end_id + 1) % len(p_ends)

    # get initial rotation and position
    R_start, _p_start = info["R_EE"], info["P_EE"]
    p_start = _p_start[:, np.newaxis]

    # Get target orientation based on initial orientation
    _R_end = (
        R.from_euler("x", 0, degrees=True).as_matrix()
        @ R.from_euler("z", 0, degrees=True).as_matrix()
        @ R_start
    )
    R_end = R.from_matrix(_R_end).as_matrix()

    controller.start(p_start, p_end, R_start, R_end, 30.0)
    q_nominal = env.q_nominal[:, np.newaxis]

    for i in range(args.iterationNum):
        # Get end-effector position
        p_current = info["P_EE"][:, np.newaxis]
        # Get end-effector orientation
        R_current = info["R_EE"]

        if i == 0:
            dt = 0
        else:
            dt = 1 / 240

        τ, error = controller.update(p_current, R_current, dt, q_nominal, info)

        if i % 10000 == 0 and i > 1:
            p_end = p_ends[p_end_id]
            p_end_id = (p_end_id + 1) % len(p_ends)

            # get initial rotation and position
            R_start, _p_start = info["R_EE"], info["P_EE"]
            p_start = _p_start[:, np.newaxis]

            # Get target orientation based on initial orientation
            _R_end = (
                R.from_euler("x", 0, degrees=True).as_matrix()
                @ R.from_euler("z", 0, degrees=True).as_matrix()
                @ R_start
            )
            R_end = R.from_matrix(_R_end).as_matrix()

            controller.start(p_start, p_end, R_start, R_end, 30.0)

        if i % 500 == 0:
            print("Iter {:.2e} \t error: {:.2e}".format(i, error))

        # Send joint commands to motor
        info = env.step(τ)

    env.close()


if __name__ == "__main__":
    main()
