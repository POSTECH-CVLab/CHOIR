"""CHOIR (f): Characteristic Orientation predictor with Invariant Residual learning.

f = g ∘ h, where:
  h (OrientationHypothesizer): predicts equivariant orientation hypothesis.
  g (ResidualPredictor): predicts invariant residual rotation on the pre-canonicalized cloud.
"""

from typing import Any

import torch
import torch.nn as nn


class CHOIR(nn.Module):
    """f: Full orientation prediction model.

    Predicts the characteristic orientation of a 3D point cloud by:
    1. Computing an equivariant orientation hypothesis h(P) via the hypothesizer.
    2. Pre-canonicalizing the point cloud using h(P).
    3. Predicting a residual rotation g(·) on the canonicalized representation.
    4. Composing: f(P) = h(P) · g(P_can, VR).
    """

    def __init__(self, hypothesizer: nn.Module, residual_predictor: nn.Module):
        super().__init__()
        self.hypothesizer = hypothesizer  # h
        self.residual_predictor = residual_predictor  # g

    def forward(self, pcd: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        """
        Args:
            pcd: (B, N, 3) input point cloud.

        Returns:
            Dictionary with:
              - mid_rots: (B, 3, 3) equivariant orientation hypothesis.
              - end_rots: (B, 3, 3) final predicted orientation (h · g).
        """
        mid_rots, inv_feat = self.hypothesizer(pcd)  # h

        p_can = torch.bmm(pcd, mid_rots)  # pre-canonicalization
        res_rots = self.residual_predictor(inv_feat, p_can)  # g

        end_rots = torch.bmm(mid_rots, res_rots)  # f = h · g

        return {"mid_rots": mid_rots, "end_rots": end_rots}
