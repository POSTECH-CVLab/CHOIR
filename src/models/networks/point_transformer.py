"""Point Transformer layers with vector-attention and relative position encoding.

Reference: Zhao et al., "Point Transformer", ICCV 2021.
"""

import torch
import torch.nn as nn


def gather_neighbors(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather k-NN features.

    Args:
        x: (B, C, N) point features.
        idx: (B, N, K) neighbor indices.

    Returns:
        (B, C, N, K) gathered neighbor features.
    """
    batch_size, num_points, k = idx.shape
    x = x.transpose(1, 2).contiguous()  # (B, N, C)
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)
    neighbors = x.view(batch_size * num_points, -1)[idx, :]
    neighbors = neighbors.view(batch_size, num_points, k, -1)
    return neighbors.permute(0, 3, 1, 2).contiguous()


class PointTransformerLayer(nn.Module):
    """Vector-attention with relative position encodings."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        assert out_channels % groups == 0
        self.out_channels = out_channels
        self.groups = groups

        self.linear_qkv = nn.Conv1d(in_channels, 3 * out_channels, kernel_size=1, bias=False)
        self.mlp_p = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            nn.Conv2d(3, out_channels, kernel_size=1),
        )

        w_channels = out_channels // groups
        self.mlp_w = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, w_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(w_channels),
            nn.ReLU(True),
            nn.Conv2d(w_channels, w_channels, kernel_size=1),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, p: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, N) features.
            p: (B, 3, N) xyz coordinates.
            knn_idx: (B, N, K) neighbor indices.
        """
        q, k, v = self.linear_qkv(x).chunk(3, dim=1)

        knn_p = gather_neighbors(p, knn_idx)
        rel_p = self.mlp_p(p.unsqueeze(-1) - knn_p)

        knn_k = gather_neighbors(k, knn_idx)
        attn = self.mlp_w(q.unsqueeze(-1) - knn_k + rel_p)
        attn = self.softmax(attn)

        bsz, grouped_channels, num_points, num_neighbors = attn.shape
        knn_v = gather_neighbors(v, knn_idx)
        knn_v = knn_v.view(bsz, grouped_channels, self.groups, num_points, num_neighbors)
        knn_v = knn_v * attn.unsqueeze(2)
        return torch.sum(knn_v.view(bsz, -1, num_points, num_neighbors), dim=-1)


class PointTransformerBlock(nn.Module):
    """Residual block with Point Transformer layer."""

    def __init__(self, in_channels: int, groups: int = 8):
        super().__init__()
        self.linear_in = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(True),
        )
        self.transformer = PointTransformerLayer(in_channels, in_channels, groups)
        self.linear_out = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels),
        )
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor, p: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
        assert p.shape[1] == 3
        y = self.linear_in(x)
        y = self.transformer(y, p, knn_idx)
        y = self.linear_out(y)
        return self.relu(y + x)
