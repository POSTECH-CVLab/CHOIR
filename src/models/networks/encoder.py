"""Equivariant Encoder (φ): SO(3)-equivariant and translation-invariant feature extraction.

Uses cross-product graph convolutions to maintain SO(3)-equivariance
while achieving translation invariance through mean subtraction.
"""

from typing import Literal

import torch
import torch.nn as nn

from src.models.networks.vn_layers import (
    VNLinear,
    VNBatchNorm,
    VNLinearLeakyReLU,
    STNkd,
    knn,
)
from src.models.networks.vnt_layers import VNTLinearLeakyReLU, VNTMaxPool, mean_pool


def _get_graph_feature_cross(
    x: torch.Tensor, k: int, idx: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cross-product graph features for SO(3)-equivariance (global variant).

    Args:
        x: (B, C, 3, N) equivariant features.
        k: number of nearest neighbors.
        idx: precomputed kNN indices, or None.

    Returns:
        feature: (B, 3*C, 3, N, k) graph features with cross-product.
        idx: (B, N, k) kNN indices.
    """
    batch_size = x.size(0)
    num_points = x.size(3)

    # Translation invariance via mean subtraction
    bias = torch.mean(x, dim=-1, keepdim=True)
    x = x - bias

    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    batched_idx = (idx + idx_base).view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[batched_idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)

    feature = torch.cat((feature - x, x, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()
    feature = bias.unsqueeze(4) + feature
    return feature, idx


class EquivariantEncoder(nn.Module):
    """φ: SO(3)-equivariant and translation-invariant point cloud encoder.

    Extracts rotation-equivariant features (B, C, 3, N) from input point clouds (B, N, 3)
    using cross-product graph convolutions and VNT layers.
    """

    def __init__(
        self,
        k: int = 40,
        pooling: Literal["max", "mean"] = "mean",
        base_ch: int = 64,
        nlatent: int = 1020,
        feature_transform: bool = False,
        which_norm_VNT: Literal["norm", "softmax"] = "norm",
    ):
        super().__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.output_ch = nlatent // 2

        self.conv_pos = VNTLinearLeakyReLU(
            3, base_ch // 3, dim=5, negative_slope=0.0, which_norm_VNT=which_norm_VNT
        )
        self.conv_center_ = VNTLinearLeakyReLU(
            base_ch // 3, base_ch // 3, dim=4, negative_slope=0.0, which_norm_VNT=which_norm_VNT
        )
        self.conv_center = VNTLinearLeakyReLU(
            base_ch // 3, 1, dim=4, negative_slope=0.0, which_norm_VNT=which_norm_VNT
        )

        self.conv1 = VNLinearLeakyReLU(base_ch // 3, base_ch // 3, dim=4, negative_slope=0.0)

        if self.feature_transform:
            self.fstn = STNkd(d=base_ch // 3, pooling=pooling)
            self.conv2 = VNLinearLeakyReLU(
                base_ch // 3 * 2, (2 * base_ch) // 3, dim=4, negative_slope=0.0
            )
        else:
            self.conv2 = VNLinearLeakyReLU(
                base_ch // 3, (2 * base_ch) // 3, dim=4, negative_slope=0.0
            )

        self.conv3 = VNLinear((2 * base_ch) // 3, self.output_ch // 3)
        self.bn3 = VNBatchNorm(self.output_ch // 3, dim=4)

        if pooling == "max":
            self.pool = VNTMaxPool(base_ch // 3)
        else:
            self.pool = mean_pool

    def forward(self, x: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, 3) input point cloud.
            idx: optional precomputed kNN indices.

        Returns:
            (B, C, 3, N) SO(3)-equivariant features.
        """
        x = x.transpose(1, 2)  # (B, 3, N)
        _, _, N = x.size()

        x = x.unsqueeze(1)  # (B, 1, 3, N)
        feat, idx = _get_graph_feature_cross(x, k=self.k, idx=idx)

        x = self.conv_pos(feat)
        x = self.pool(x)
        x_center = self.conv_center(self.conv_center_(x))

        x = x - x_center
        x = self.conv1(x)

        if self.feature_transform:
            x_global = self.fstn(x).unsqueeze(-1).repeat(1, 1, 1, N)
            x = torch.cat((x, x_global), 1)

        x = self.conv2(x)
        x = self.bn3(self.conv3(x))

        return x
