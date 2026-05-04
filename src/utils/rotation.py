"""Rotation utilities for SO(3) operations."""

import torch


def ortho2rotation(poses: torch.Tensor) -> torch.Tensor:
    """Convert 2x3 orthonormal basis to 3x3 rotation matrix via Gram-Schmidt.

    Reference: Choy & Dong, DeepGlobalRegistration.

    Args:
        poses: (B, 2, 3) two basis vectors.

    Returns:
        (B, 3, 3) valid rotation matrix (det = +1).
    """

    def normalize_vector(v: torch.Tensor) -> torch.Tensor:
        v_mag = torch.clamp(torch.sqrt((v**2).sum(1, keepdim=True)), min=1e-8)
        return v / (v_mag + 1e-10)

    def cross_product(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        return torch.stack((i, j, k), dim=1)

    def proj_u2a(u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        inner_prod = (u * a).sum(1, keepdim=True)
        norm2 = torch.clamp((u**2).sum(1, keepdim=True), min=1e-8)
        return (inner_prod / (norm2 + 1e-10)) * u

    assert poses.dim() == 3 and poses.shape[1] == 2 and poses.shape[2] == 3

    x_raw = poses[:, 0, :]
    y_raw = poses[:, 1, :]

    x = normalize_vector(x_raw)
    y = normalize_vector(y_raw - proj_u2a(x, y_raw))
    z = cross_product(x, y)

    return torch.cat((x[:, :, None], y[:, :, None], z[:, :, None]), dim=2)


def project_to_rotation(mat: torch.Tensor) -> torch.Tensor:
    """Project a batch of 3x3 matrices to the nearest rotation matrices (SVD).

    Args:
        mat: (B, 3, 3) matrices.

    Returns:
        (B, 3, 3) rotation matrices with det = +1.
    """
    U, _, Vh = torch.linalg.svd(mat)
    R = U @ Vh
    # Fix reflections (det = -1)
    det = torch.det(R)
    sign = torch.ones_like(det)
    sign[det < 0] = -1
    correction = torch.diag_embed(
        torch.stack([torch.ones_like(sign), torch.ones_like(sign), sign], dim=-1)
    )
    return U @ correction @ Vh


def find_the_closest_rotation(mat: torch.Tensor) -> torch.Tensor:
    """Project a matrix to the closest rotation matrix (L2 Chordal distance).

    Args:
        mat: (3, 3) matrix.

    Returns:
        (3, 3) rotation matrix with det = +1.
    """
    U, _, Vh = torch.linalg.svd(mat)
    R = U @ Vh
    if torch.det(R) < 0:
        ii = torch.tensor([1, 1, -1], dtype=U.dtype, device=U.device)
        R = U @ (torch.diag(ii) @ Vh)
    return R


def mean_rotations(rots: torch.Tensor) -> torch.Tensor:
    """Compute the mean rotation (L2 Chordal distance).

    Args:
        rots: (B, 3, 3) rotation matrices.

    Returns:
        (3, 3) mean rotation matrix.
    """
    return find_the_closest_rotation(rots.sum(0))


def angular_std(rots: torch.Tensor) -> torch.Tensor:
    """Compute angular standard deviation from a set of rotation matrices.

    Args:
        rots: (B, 3, 3) rotation matrices.

    Returns:
        Scalar angular std in degrees.
    """
    R_mean = mean_rotations(rots)
    trace = torch.einsum("bii -> b", rots @ R_mean.transpose(0, 1))
    cosine = (trace - 1) / 2
    angle = torch.acos(cosine.clamp(-1, 1)) * 180 / torch.pi
    return torch.sqrt(torch.mean(angle**2))
