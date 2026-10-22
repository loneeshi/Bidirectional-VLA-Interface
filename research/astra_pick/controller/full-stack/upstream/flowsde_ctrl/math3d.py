"""Small SO(3) / planar helpers shared by all components (embodiment-agnostic).

Clean copies of flowsde/classical.py (rotvec, expm, Rz), flowsde/task_track.py (rot6d), flowsde/nav_scenes.py
(_rz, _wrap) and flowsde/zoo/common.py (slerp_R). Kept bit-identical: the regression tests compare actions exactly.
"""
import numpy as np


def rotvec(R):
    """Rotation vector (axis * angle) of a rotation matrix."""
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); th = np.arccos(c)
    w = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-7:
        return w
    if th > np.pi - 1e-3:
        from scipy.spatial.transform import Rotation
        return Rotation.from_matrix(R).as_rotvec()
    return th / np.sin(th) * w


def expm(w):
    """Rotation matrix of a rotation vector (Rodrigues)."""
    th = np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if th < 1e-9:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rot6d(R):
    """First two columns of R, flattened column-major (the observation encoding of a relative rotation)."""
    return R[:, :2].T.reshape(-1)


def from_rot6d(c0, c1):
    """Inverse of rot6d as the reference adapter decodes it (third column = c0 x c1, no re-orthonormalisation)."""
    return np.stack([c0, c1, np.cross(c0, c1)], -1)


def slerp_R(Ra, Rb, w):
    return Ra @ expm(w * rotvec(Ra.T @ Rb))


def unit(F):
    n = np.linalg.norm(F)
    return F / n if n > 1e-9 else np.zeros(3)
