import torch
import numpy as np
from scipy.linalg import expm, norm


def find_the_closest_rotation(mat):
    r"""
        mat.shape: (3, 3)
        distance: L2 Chordal distance
    """
    U, _, Vh = torch.linalg.svd(mat)
    R = U @ Vh

    if torch.det(R) < 0:
        ii = torch.tensor([1, 1, -1], dtype=U.dtype, device=U.device)
        R = U @ (torch.diag(ii) @ Vh)

    return R


def mean_rotations(rots):
    r"""
        rots.shape: (B, 3, 3)
        distance: L2 Chordal distance
    """
    return find_the_closest_rotation(rots.sum(0))


def cal_angular_std_from_rotations(rots):
    r"""
        rots.shape: (B, 3, 3)
        distance: Angular distance
    """
    R_mean = mean_rotations(rots)
    trace = torch.einsum('bii -> b', rots @ R_mean.transpose(0, 1))
    cosine = (trace - 1) / 2
    angle = torch.acos(cosine.clamp(-1, 1)) * 180 / torch.pi
    std = torch.sqrt(torch.mean(angle ** 2))

    return std


def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


# According to Canonical Capsules, this is not uniform.
# TODO: Use scipy.spatial.transform.Rotation.random() instead.
def sample_random_rotation(rotation_range=360):
    R = M(
        np.random.random(3) - 0.5,
        rotation_range * np.pi / 180.0 * (np.random.random(1) - 0.5)
    )

    return R


def ortho2rotation(poses):
    r"""
    poses: batch x 2 x 3
    From https://github.com/chrischoy/DeepGlobalRegistration/blob/master/core
    /registration.py#L16
    Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) 
    and Wei Dong (weidong@andrew.cmu.edu)
    """

    def normalize_vector(v):
        r"""
        Batch x 3
        """
        v_mag = torch.sqrt((v ** 2).sum(1, keepdim=True))
        v_mag = torch.clamp(v_mag, min=1e-8)
        v = v / (v_mag + 1e-10)
        return v

    def cross_product(u, v):
        r"""
        u: batch x 3
        v: batch x 3
        """
        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

        i = i[:, None]
        j = j[:, None]
        k = k[:, None]
        return torch.cat((i, j, k), 1)

    def proj_u2a(u, a):
        r"""
        u: batch x 3
        a: batch x 3
        """
        inner_prod = (u * a).sum(1, keepdim=True)
        norm2 = (u ** 2).sum(1, keepdim=True)
        norm2 = torch.clamp(norm2, min=1e-8)
        factor = inner_prod / (norm2 + 1e-10)
        return factor * u

    assert poses.dim() == 3 and poses.shape[1] == 2 and poses.shape[2] == 3

    x_raw = poses[:, 0, :]
    y_raw = poses[:, 1, :]

    x = normalize_vector(x_raw)
    y = normalize_vector(y_raw - proj_u2a(x, y_raw))
    z = cross_product(x, y)

    x = x[:, :, None]
    y = y[:, :, None]
    z = z[:, :, None]

    return torch.cat((x, y, z), 2)


if __name__ == "__main__":
    from scipy.spatial.transform import Rotation

    B = 2048
    np.random.seed(0)

    # Test case 1: Uniformly distributed rotations
    random_rots = Rotation.random(B).as_matrix()
    std = cal_angular_std_from_rotations(torch.from_numpy(random_rots)).item()
    print(f"[Case #1] std = {std}")

    # Test case 2-6: Random axis + random angle with range 360, 180, 90, 45, 5
    for i, angle_range in enumerate([360, 180, 90, 45, 5]):
        random_rots = np.vstack([sample_random_rotation(angle_range)[None, ...] for _ in range(B)])
        std = cal_angular_std_from_rotations(torch.from_numpy(random_rots)).item()
        print(f"[Case #{2 + i}] std = {std}")