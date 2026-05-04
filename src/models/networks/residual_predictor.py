"""Residual Predictor (g): Predicts invariant residual rotation.

Takes SO(3)-invariant features and a pre-canonicalized point cloud,
processes them through Point Transformer blocks, and predicts a
residual rotation to refine the orientation hypothesis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.networks.vn_layers import knn
from src.models.networks.point_transformer import PointTransformerBlock
from src.utils.rotation import ortho2rotation


class _GeMPool(nn.Module):
    """Generalized Mean Pooling."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool1d(x.clamp(min=self.eps).pow(self.p), x.size(-1)).pow(1.0 / self.p)


class ResidualPredictor(nn.Module):
    """g: Predicts residual rotation from invariant features and canonicalized point cloud.

    Architecture: Conv1d → PointTransformerBlocks → GeMPool → MLP → ortho2rotation.
    """

    def __init__(self, in_channels: int, mid_channels: int = 64, num_blocks: int = 4):
        super().__init__()

        self.pre_transformer = nn.Sequential(
            nn.Conv1d(in_channels * 3, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(True),
        )
        self.transformer = nn.ModuleList(
            [PointTransformerBlock(mid_channels) for _ in range(num_blocks)]
        )
        self.final = nn.Sequential(
            nn.Linear(mid_channels, mid_channels, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(True),
            nn.Linear(mid_channels, 6),
        )
        self.gem_pool = _GeMPool()

    def forward(self, inv_feat: torch.Tensor, p_can: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inv_feat: (B, C, 3, N) SO(3)-invariant features from VNStdFeature.
            p_can: (B, N, 3) pre-canonicalized point cloud.

        Returns:
            (B, 3, 3) residual rotation matrix.
        """
        p = p_can.transpose(1, 2)  # (B, 3, N)
        knn_idx = knn(p, 16)

        bsz, _, _, num_points = inv_feat.shape
        x = inv_feat.contiguous().view(bsz, -1, num_points)  # (B, C*3, N)

        x = self.pre_transformer(x)
        for block in self.transformer:
            x = block(x, p, knn_idx)

        global_feat = self.gem_pool(x).squeeze(-1)  # (B, mid_channels)
        out = self.final(global_feat)  # (B, 6)
        return ortho2rotation(out.view(-1, 2, 3))
