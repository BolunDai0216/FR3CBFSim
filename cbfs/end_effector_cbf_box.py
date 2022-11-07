import numpy as np


def end_effector_cbf_box(
    q, dq, info, d_max=0.4, alpha=10.0, n_vec=np.array([[-1.0], [0.0]])
):
    """
    cbf.shape = (1, 1), dcbf_dq.shape = (1, nx)
    """
    J, dJ, ee_pos = info["J"], info["dJ"], info["P_EE"]
    cbf = -n_vec.T @ J @ dq + alpha * (d_max - n_vec.T @ ee_pos)
    dcbf_dq = np.hstack((-n_vec.T @ (dJ + alpha * J), -n_vec.T @ J))

    return cbf, dcbf_dq
