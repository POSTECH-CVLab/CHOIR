"""Vector Neuron Transformer (VNT) layers.

Translation-invariant extensions of Vector Neuron layers using
convex-combination weights instead of learned linear maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


class VNTLinear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, which_norm: str = "norm"):
        super().__init__()
        self.in_channels = in_channels
        self.which_norm = which_norm
        self.weight = nn.Parameter(torch.rand(out_channels, in_channels))
        if which_norm == "softmax":
            self.weight.data = F.softmax(self.weight.data, dim=1)
        else:
            self.weight.data = self.weight.data / (
                torch.sum(self.weight.data, dim=1, keepdim=True) + EPS
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N_feat, 3, N_samples, ...)"""
        if self.which_norm == "softmax":
            weight = F.softmax(self.weight, dim=1)
        else:
            weight = self.weight / (torch.sum(self.weight, dim=1, keepdim=True) + EPS)
        return torch.matmul(x.transpose(1, -1), weight.t()).transpose(1, -1)


class VNTLinearLeakyReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int = 5,
        share_nonlinearity: bool = False,
        negative_slope: float = 0.2,
        use_batchnorm: bool = True,
        which_norm_VNT: str = "norm",
    ):
        super().__init__()
        self.dim = dim
        self.negative_slope = negative_slope

        self.map_to_feat = VNTLinear(in_channels, out_channels, which_norm_VNT)

        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.batchnorm = VNTBatchNorm(out_channels, dim=dim)

        dir_out = 1 if share_nonlinearity else out_channels
        self.map_to_dir = VNTLinear(in_channels, dir_out, which_norm_VNT)
        self.map_to_src = VNTLinear(in_channels, 1, which_norm_VNT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.map_to_feat(x)
        if self.use_batchnorm:
            p = self.batchnorm(p)
        d = self.map_to_dir(x)
        o = self.map_to_src(x)
        d = d - o
        dotprod = ((p - o) * d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1 - self.negative_slope) * (
            mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + EPS)) * d)
        )
        return x_out


class VNTBatchNorm(nn.Module):
    def __init__(self, num_features: int, dim: int):
        super().__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim > 3:
            mean_val = torch.mean(x, dim=3, keepdim=True)
            x = x - mean_val
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn
        if self.dim > 3:
            x = x + mean_val
        return x


class VNTMaxPool(nn.Module):
    def __init__(self, in_channels: int, which_norm_VNT: str = "norm"):
        super().__init__()
        self.map_to_dir = VNTLinear(in_channels, in_channels, which_norm_VNT)
        self.map_to_src = VNTLinear(in_channels, 1, which_norm_VNT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.map_to_dir(x)
        o = self.map_to_src(x)
        dotprod = ((x - o) * (d - o)).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid(
            [torch.arange(j) for j in x.size()[:-1]], indexing="ij"
        ) + (idx,)
        return x[index_tuple]


def mean_pool(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    return x.mean(dim=dim, keepdim=keepdim)
