"""Vector Neuron (VN) layers for SO(3)-equivariant feature processing.

Reference: Deng et al., "Vector Neurons: A General Framework for SO(3)-Equivariant Networks", ICCV 2021.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.init as init

EPS = 1e-6


@torch.inference_mode()
def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Compute k-nearest neighbors based on pairwise distances.

    Args:
        x: (B, C, N) point features.
        k: number of nearest neighbors.

    Returns:
        idx: (B, N, k) neighbor indices.
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


class VNLinear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        init.kaiming_uniform_(self.map_to_feat.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N_feat, 3, N_samples, ...)"""
        return self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels: int, share_nonlinearity: bool = False, negative_slope: float = 0.2):
        super().__init__()
        out = 1 if share_nonlinearity else in_channels
        self.map_to_dir = nn.Linear(in_channels, out, bias=False)
        self.negative_slope = negative_slope
        init.kaiming_uniform_(self.map_to_dir.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1 - self.negative_slope) * (
            mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d)
        )
        return x_out


class VNLinearLeakyReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int = 5,
        share_nonlinearity: bool = False,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.dim = dim
        self.negative_slope = negative_slope

        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)

        out = 1 if share_nonlinearity else out_channels
        self.map_to_dir = nn.Linear(in_channels, out, bias=False)

        init.kaiming_uniform_(self.map_to_feat.weight)
        init.kaiming_uniform_(self.map_to_dir.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        p = self.batchnorm(p)
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (p * d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1 - self.negative_slope) * (
            mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + EPS)) * d)
        )
        return x_out


class VNBatchNorm(nn.Module):
    def __init__(self, num_features: int, dim: int):
        super().__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        return x / norm * norm_bn


class VNMaxPool(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        nn.init.kaiming_uniform_(self.map_to_dir.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid(
            [torch.arange(j) for j in x.size()[:-1]], indexing="ij"
        ) + (idx,)
        return x[index_tuple]


def mean_pool(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    return x.mean(dim=dim, keepdim=keepdim)


class VNStdFeature(nn.Module):
    """Extract rotation-invariant features from SO(3)-equivariant features."""

    def __init__(
        self,
        in_channels: int,
        dim: int = 4,
        normalize_frame: bool = False,
        share_nonlinearity: bool = False,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame

        self.vn1 = VNLinearLeakyReLU(
            in_channels, in_channels // 2, dim=dim,
            share_nonlinearity=share_nonlinearity, negative_slope=negative_slope,
        )
        self.vn2 = VNLinearLeakyReLU(
            in_channels // 2, in_channels // 4, dim=dim,
            share_nonlinearity=share_nonlinearity, negative_slope=negative_slope,
        )
        frame_out = 2 if normalize_frame else 3
        self.vn_lin = nn.Linear(in_channels // 4, frame_out, bias=False)
        nn.init.kaiming_uniform_(self.vn_lin.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z0 = self.vn2(self.vn1(x))
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)

        if self.normalize_frame:
            v1 = z0[:, 0, :]
            v1_norm = torch.sqrt((v1 * v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm + EPS)
            v2 = z0[:, 1, :]
            v2 = v2 - (v2 * u1).sum(1, keepdims=True) * u1
            v2_norm = torch.sqrt((v2 * v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm + EPS)
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)

        if self.dim == 4:
            x_std = torch.einsum("bijm,bjkm->bikm", x, z0)
        elif self.dim == 3:
            x_std = torch.einsum("bij,bjk->bik", x, z0)
        elif self.dim == 5:
            x_std = torch.einsum("bijmn,bjkmn->bikmn", x, z0)

        return x_std


class STNkd(nn.Module):
    """Spatial Transformer Network for VN features."""

    def __init__(self, d: int = 64, pooling: Literal["max", "mean"] = "max"):
        super().__init__()

        self.conv1 = VNLinearLeakyReLU(d, 64 // 3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 128 // 3, dim=4, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU(128 // 3, 1024 // 3, dim=4, negative_slope=0.0)

        self.fc1 = VNLinearLeakyReLU(1024 // 3, 512 // 3, dim=3, negative_slope=0.0)
        self.fc2 = VNLinearLeakyReLU(512 // 3, 256 // 3, dim=3, negative_slope=0.0)

        if pooling == "max":
            self.pool = VNMaxPool(1024 // 3)
        else:
            self.pool = mean_pool

        self.fc3 = VNLinear(256 // 3, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv3(self.conv2(self.conv1(x)))
        x = self.pool(x)
        x = self.fc3(self.fc2(self.fc1(x)))
        return x
