import numpy as np
import numpy.linalg as LA
from scipy.spatial.transform import Rotation as R


def axis_angle_from_rot_mat(rot_mat):
    rotation = R.from_matrix(rot_mat)
    axis_angle = rotation.as_rotvec()
    angle = LA.norm(axis_angle)
    axis = axis_angle / angle

    return axis, angle


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


def smooth_trig_path_gen(t, p_start, p_end, R_start, axis_error, angle_error, T=30.0):
    # Compute α and dα
    alpha, dalpha, ddalpha = alpha_func(t, T=T)

    # Compute postion target
    p_target = p_start + alpha * (p_end - p_start)

    # Compute velocity target
    v_target = dalpha * (p_end - p_start)

    # Compute accleration target
    a_target = ddalpha * (p_end - p_start)

    # Compute Rotation target
    theta_t = alpha * angle_error
    R_target = R.from_rotvec(axis_error * theta_t).as_matrix() @ R_start

    # Compute ω (angular velocity) target
    ω_target = dalpha * axis_error * angle_error

    # Compute dω (angular acceleration) target
    dω_target = ddalpha * axis_error * angle_error

    path_targets = {
        "p_target": p_target,
        "v_target": v_target,
        "a_target": a_target,
        "R_target": R_target,
        "ω_target": ω_target,
        "dω_target": dω_target,
    }

    return path_targets


def get_R_end_from_start(x_ang, y_ang, z_ang, R_start):
    """Get target orientation based on initial orientation"""
    _R_end = (
        R.from_euler("x", x_ang, degrees=True).as_matrix()
        @ R.from_euler("y", y_ang, degrees=True).as_matrix()
        @ R.from_euler("z", z_ang, degrees=True).as_matrix()
        @ R_start
    )
    R_end = R.from_matrix(_R_end).as_matrix()

    return R_end
