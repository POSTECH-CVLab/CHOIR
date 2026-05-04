"""Orientation Hypothesizer (h): SO(3)-equivariant orientation hypothesis prediction.

h = ψ ∘ φ, where:
  φ (EquivariantEncoder): extracts SO(3)-equivariant features from point clouds.
  ψ (EquivariantRotationPredictor): predicts rotation matrix from equivariant features.
  VR (VNStdFeature): extracts SO(3)-invariant features for the residual predictor.
"""

from typing import Literal

import torch
import torch.nn as nn

from src.models.networks.vn_layers import VNLinear, VNBatchNorm
from src.utils.rotation import ortho2rotation, project_to_rotation


class EquivariantRotationPredictor(nn.Module):
    """ψ: Predicts rotation matrix from SO(3)-equivariant features.

    Supports two rotation representations:
      - "6d": 2-vector basis → Gram-Schmidt orthonormalization (Zhou et al., 2019)
      - "9d": 3-vector basis → raw 3x3 matrix, SVD projection at inference (Choy et al., 2020)
    """

    def __init__(
        self,
        in_channels: int,
        rotation_repr: Literal["6d", "9d"] = "6d",
    ):
        super().__init__()
        self.rotation_repr = rotation_repr
        out_vectors = 2 if rotation_repr == "6d" else 3
        self.to_basis = nn.Sequential(
            VNLinear(in_channels, out_vectors),
            VNBatchNorm(out_vectors, dim=4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, 3, N) equivariant features.

        Returns:
            (B, 3, 3) predicted rotation matrix.
        """
        x = self.to_basis(x)  # (B, 2or3, 3, N)
        x = x.permute(0, 3, 1, 2)  # (B, N, 2or3, 3)
        global_feat = x.mean(dim=1)  # (B, 2or3, 3)

        if self.rotation_repr == "6d":
            return ortho2rotation(global_feat)
        else:
            # 9d: raw 3x3, SVD projection only at inference
            mat = global_feat  # (B, 3, 3)
            if not self.training:
                mat = project_to_rotation(mat)
            return mat


class OrientationHypothesizer(nn.Module):
    """h: Produces orientation hypothesis and invariant features.

    Combines the equivariant encoder (φ), equivariant rotation predictor (ψ),
    and invariant feature extractor (VR) to output:
      - mid_rots: the equivariant orientation hypothesis h(PR)
      - inv_feat: SO(3)-invariant features for the residual predictor
    """

    def __init__(
        self,
        encoder: nn.Module,
        rotation_predictor: nn.Module,
        invariant_feature_extractor: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder  # φ
        self.rotation_predictor = rotation_predictor  # ψ
        self.invariant_feature_extractor = invariant_feature_extractor  # VR

    def forward(
        self, pcd: torch.Tensor, knn_idx: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pcd: (B, N, 3) input point cloud.
            knn_idx: optional precomputed kNN indices.

        Returns:
            mid_rots: (B, 3, 3) equivariant orientation hypothesis.
            inv_feat: (B, C', 3, N) SO(3)-invariant features.
        """
        feat = self.encoder(pcd, knn_idx)  # (B, C, 3, N)
        mid_rots = self.rotation_predictor(feat)  # (B, 3, 3)

        # Invariant feature extraction (VR)
        feat_mean = feat.mean(dim=-1, keepdim=True).expand(feat.size())
        feat_cat = torch.cat((feat, feat_mean), dim=1)  # (B, 2C, 3, N)
        inv_feat = self.invariant_feature_extractor(feat_cat)  # (B, 2C, 3, N)

        return mid_rots, inv_feat
