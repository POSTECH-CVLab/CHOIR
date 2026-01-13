from scipy.spatial.transform import Rotation
import torch
from torch_batch_svd import svd as fast_svd


def random_rotate(x):

    """
    x - B, N, 3
    out - B, N, 3
    Randomly rotate point cloud
    """
    
    out = perform_rotation(torch.from_numpy(Rotation.random(x.shape[0]).as_matrix()).type_as(x), x)

    return out

def mean_center(x):
    """
    x - B, N, 3
    x_mean - B, N, 3
    Mean center point cloud
    """

    out = x - x.mean(-2, keepdims = True)
    return out

def perform_rotation(R, x):
    '''
    Perform rotation on point cloud
    R - B, 3, 3
    x - B, N, 3

    out - B, N, 3
    '''
    out = torch.einsum("bij,bpj->bpi", R.type_as(x), x)

    return out

def orthonormalize_basis_legacy(basis):
    """
    Returns orthonormal basis vectors
    basis - B, 3, 3

    out - B, 3, 3
    """
    u, s, v = torch.svd(basis)
    out = u @ v.transpose(-2, -1)

    return out


def orthonormalize_basis(basis):
    """
    Returns orthonormal basis vectors
    basis - B, 5, 3, 3

    out - B, 5, 3, 3
    """
    out = []
    for mat in basis:
        U, _, V = fast_svd(mat)
        S = torch.eye(3).repeat(U.shape[0], 1, 1).to(U.device)
        det = U.det() * V.det()
        S[det < 0, -1, -1] = -1
        R = U @ S @ V.transpose(1, 2)
        out.append(R)

    return torch.stack(out)