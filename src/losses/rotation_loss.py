"""Rotation loss functions."""

import torch
import torch.nn as nn


class RotationMSELoss(nn.Module):
    """MSE loss on relative rotation matrices.

    Given predicted rotations for a pair of point clouds, computes the MSE
    between the predicted relative rotation and the ground-truth relative rotation.
    """

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        rots_src: torch.Tensor,
        rots_trg: torch.Tensor,
        rots_diff: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            rots_src: (B, 3, 3) predicted rotation for source point cloud.
            rots_trg: (B, 3, 3) predicted rotation for target point cloud.
            rots_diff: (B, 3, 3) ground-truth relative rotation (R_trg @ R_src^T).

        Returns:
            Scalar MSE loss.
        """
        rots_diff_pred = torch.bmm(rots_trg, rots_src.transpose(1, 2))
        return self.mse_loss(rots_diff_pred, rots_diff)
