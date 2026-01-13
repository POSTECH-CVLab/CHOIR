from typing import Literal

import torch
import torch.nn as nn

from src.model.vn_layers import VNLinear, VNBatchNorm, VNLinearLeakyReLU, STNkd, knn
from src.model.vnt_layers import VNTLinearLeakyReLU, VNTMaxPool, mean_pool


def get_graph_feature_cross(x, k, use_global=False, idx=None):
    if use_global:
        return get_graph_feature_cross_global(x, k, idx)
    else:
        return get_graph_feature_cross_local(x, k, idx)


def get_graph_feature_cross_global(x, k, idx=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    #To preserve translation
    bias = torch.mean(x, dim=-1, keepdim=True)
    x = x - bias

    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    batched_idx = idx + idx_base
    batched_idx = batched_idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[batched_idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)
    
    feature = torch.cat((feature-x, x, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()
    feature = bias.unsqueeze(4) + feature
    return feature, idx


def get_graph_feature_cross_local(x, k, idx=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    batched_idx = idx + idx_base
    batched_idx = batched_idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[batched_idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)

    feature_mean = torch.mean(feature, dim=2, keepdim=True)
    cross = torch.cross(feature - feature_mean, x - feature_mean, dim=-1)
    feature = torch.cat((2*feature - x, x, x+cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature, idx


class IdentityEncoder(nn.Module):
    def __init__(self):
        super(IdentityEncoder, self).__init__()
        self.output_ch = 3

    def forward(self, x, idx=None):
        x = x.transpose(1, 2) # (B, C, N)
        return x.unsqueeze(1)


# default args: https://github.com/orenkatzir/VN-SPD/blob/master/options/base_options.py
class VNTSimpleEncoder(nn.Module):
    r"""
        A translation-invariant and rotation-equivariant encoder
    """

    def __init__(
        self,
        k: int,
        pooling: Literal["max", "mean"],
        base_ch: int,
        nlatent: int,
        feature_transform: bool,
        which_norm_VNT: Literal["norm", "softmax"],
    ):
        super(VNTSimpleEncoder, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.output_ch = nlatent // 2

        self.conv_pos = VNTLinearLeakyReLU(3, base_ch // 3, dim=5, negative_slope=0.0, which_norm_VNT=which_norm_VNT)
        self.conv_center_ = VNTLinearLeakyReLU(base_ch // 3, base_ch // 3, dim=4, negative_slope=0.0, which_norm_VNT=which_norm_VNT)
        self.conv_center = VNTLinearLeakyReLU(base_ch // 3, 1, dim=4, negative_slope=0.0, which_norm_VNT=which_norm_VNT)

        self.conv1 = VNLinearLeakyReLU(base_ch // 3, base_ch // 3, dim=4, negative_slope=0.0)
        
        if self.feature_transform:
            self.fstn = STNkd(d=base_ch // 3, pooling=pooling)
            self.conv2 = VNLinearLeakyReLU(base_ch // 3 * 2, (2 * base_ch) // 3, dim=4, negative_slope=0.0)
        else:
            self.conv2 = VNLinearLeakyReLU(base_ch // 3, (2 * base_ch) // 3, dim=4, negative_slope=0.0)
        
        self.conv3 = VNLinear((2 * base_ch) // 3, self.output_ch // 3)
        self.bn3 = VNBatchNorm(self.output_ch // 3, dim=4)

        if pooling == "max":
            self.pool = VNTMaxPool(base_ch // 3)
        else:
            self.pool = mean_pool


    def forward(self, x, idx=None):
        x = x.transpose(1, 2)
        _, _, N = x.size()

        x = x.unsqueeze(1)
        feat, idx = get_graph_feature_cross(x, k=self.k, use_global=True, idx=idx)

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


class VNTLargeEncoder(nn.Module):
    r"""
        A translation-invariant and rotation-equivariant encoder
    """

    def __init__(
        self,
        k: int,
        pooling: Literal["max", "mean"],
        base_ch: int,
        nlatent: int,
        feature_transform: bool,
        which_norm_VNT: Literal["norm", "softmax"],
    ):
        super(VNTLargeEncoder, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.output_ch = nlatent // 2

        self.conv_pos = VNTLinearLeakyReLU(3, base_ch // 3, dim=5, negative_slope=0.0, which_norm_VNT=which_norm_VNT)
        self.conv_center_ = VNTLinearLeakyReLU(base_ch // 3, base_ch // 3, dim=4, negative_slope=0.0, which_norm_VNT=which_norm_VNT)
        self.conv_center = VNTLinearLeakyReLU(base_ch // 3, 1, dim=4, negative_slope=0.0, which_norm_VNT=which_norm_VNT)

        self.conv1 = VNLinearLeakyReLU(base_ch // 3, base_ch // 3, dim=4, negative_slope=0.0)
        
        if self.feature_transform:
            self.fstn = STNkd(d=base_ch // 3, pooling=pooling)
            self.conv2 = VNLinearLeakyReLU(base_ch // 3 * 2, (2 * base_ch) // 3, dim=4, negative_slope=0.0)
        else:
            self.conv2 = VNLinearLeakyReLU(base_ch // 3, (2 * base_ch) // 3, dim=4, negative_slope=0.0)
        
        self.conv3 = VNLinearLeakyReLU((2 * base_ch) // 3, (2 * base_ch) // 3, dim=4, negative_slope=0.0)
        self.conv4 = VNLinearLeakyReLU((2 * base_ch) // 3, (2 * base_ch) // 3, dim=4, negative_slope=0.0)
        self.conv5 = VNLinearLeakyReLU((2 * base_ch) // 3, (2 * base_ch) // 3, dim=4, negative_slope=0.0)
        self.conv6 = VNLinearLeakyReLU((2 * base_ch) // 3, (2 * base_ch) // 3, dim=4, negative_slope=0.0)
        self.conv7 = VNLinear((2 * base_ch) // 3, self.output_ch // 3)
        self.bn7 = VNBatchNorm(self.output_ch // 3, dim=4)

        if pooling == "max":
            self.pool = VNTMaxPool(base_ch // 3)
        else:
            self.pool = mean_pool


    def forward(self, x, idx=None):
        x = x.transpose(1, 2)
        _, _, N = x.size()

        x = x.unsqueeze(1)
        feat, idx = get_graph_feature_cross(x, k=self.k, use_global=True, idx=idx)

        x = self.conv_pos(feat)
        x = self.pool(x)
        x_center = self.conv_center(self.conv_center_(x))

        x = x - x_center
        x = self.conv1(x)

        if self.feature_transform:
            x_global = self.fstn(x).unsqueeze(-1).repeat(1, 1, 1, N)
            x = torch.cat((x, x_global), 1)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.bn7(self.conv7(x))

        return x



if __name__ == "__main__":
    assert torch.cuda.is_available()
    B, N = 64, 1024
    
    x = -1 + 2*torch.rand(B, N, 3) # [-1, 1)
    x_shifted = x + torch.tensor([100, 200, 300], dtype=x.dtype).view(1, 1, 3)

    vnt = VNTSimpleEncoder(k=40).cuda()
    vnt = vnt.eval()

    with torch.inference_mode():
        out = vnt(x.cuda()).cpu()
        out_shifted = vnt(x_shifted.cuda()).cpu()

    torch.testing.assert_close(out, out_shifted, atol=5, rtol=10)
