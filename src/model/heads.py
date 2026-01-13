import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_batch_svd import svd as fast_svd
except:
    print("torch_batch_svd is not installed. You can install it with install_torch_batch_svd.sh.")
try:
    from ema_pytorch import EMA
except:
    print("ema_pytorch is not installed. If you want, pip install ema-pytorch")


from src.model.vn_layers import VNLinear, VNBatchNorm, VNLinearLeakyReLU, VNStdFeature, knn
from src.model.point_transformer import PointTransformerBlock
from src.rotation import ortho2rotation


class GeMPool(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeMPool,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool1d(x.clamp(min=eps).pow(p), x.size(-1)).pow(1./p)


#####################
# Equivariant heads #
#####################
class GlobalPoolingHeadBase(nn.Module):
    def __init__(self, in_channels):
        super(GlobalPoolingHeadBase, self).__init__()

        self.to_basis = nn.Sequential(
            VNLinear(in_channels, 2),
            VNBatchNorm(2, dim=4)
        )

    def pool(self, local_feat):
        raise NotImplementedError

    def forward(self, x, p, knn_idx, eval=False):
        x = self.to_basis(x)
        
        local_feat = x.permute(0, 3, 1, 2)
        global_feat = self.pool(local_feat)
        global_ori = ortho2rotation(global_feat)

        return {"end_rots": global_ori}


class GlobalAvgPoolingHead(GlobalPoolingHeadBase):
    def pool(self, local_feat):
        return local_feat.mean(dim=1) # (B, C, 3)


# GlobalPoolingHeadBase with LeakyReLU activation
class GlobalAvgPoolingReLUHead(GlobalAvgPoolingHead):
    def __init__(self, in_channels):
        super(GlobalAvgPoolingReLUHead, self).__init__(in_channels)
        self.to_basis = VNLinearLeakyReLU(in_channels, 2, dim=4, negative_slope=0.0)


class GlobalGeMPoolingHead(GlobalPoolingHeadBase):
    def __init__(self, in_channels):
        super(GlobalGeMPoolingHead, self).__init__(in_channels)
        self.gem_pool = GeMPool()

    def pool(self, local_feat):
        bsz, num_points, num_channels, _ = local_feat.shape
        local_feat = local_feat.reshape(bsz, num_points, -1)
        local_feat = local_feat.permute(0, 2, 1)
        return self.gem_pool(local_feat).squeeze(-1).view(bsz, num_channels, -1) # TODO(chrockey): fix this.


class GlobalGeMPoolingReLUHead(GlobalGeMPoolingHead):
    def __init__(self, in_channels):
        super(GlobalGeMPoolingReLUHead, self).__init__(in_channels)
        self.to_basis = VNLinearLeakyReLU(in_channels, 2, dim=4, negative_slope=0.0)

class TwoStreamInvHead(nn.Module):
    def __init__(self, in_channels):
        super(TwoStreamInvHead, self).__init__()

        self.equi_head = GlobalAvgPoolingHead(in_channels)
        # rotation-invariant head
        self.std_feature = VNStdFeature(in_channels * 2, dim=4, normalize_frame=False, negative_slope=0.0)
        self.final = nn.Sequential(
            nn.Linear(in_channels * 2 * 3, in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(True),
            nn.Linear(in_channels, 6)
        )

    def forward(self, x, p, knn_idx, eval=False):
        bsz = len(x)
        equi_out_dict = self.equi_head(x, p, knn_idx)

        # rotation-invariant layer
        x_mean_out = x.mean(dim=-1, keepdim=True)
        x_mean = x_mean_out.expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x = self.std_feature(x)

        # (b, c, 3, n)
        x = torch.max(x, -1, keepdim=False)[0] # (b, c, 3)
        x = x.view(bsz, -1)
        res_feat = self.final(x) # (b, 6)
        res_ori = ortho2rotation(res_feat.view(-1, 2, 3))

        main_ori = equi_out_dict["end_rots"]
        out_ori = torch.bmm(main_ori.detach(), res_ori)

        return {
            "mid_rots": main_ori,
            "res_rots": res_ori,
            "end_rots": out_ori,
        }


class TwoStreamInvNaiveHead(TwoStreamInvHead):
    def forward(self, x, p, knn_idx, eval=False):
        bsz = len(x)
        equi_out_dict = self.equi_head(x, p, knn_idx)

        # rotation-invariant layer
        x_mean_out = x.mean(dim=-1, keepdim=True)
        x_mean = x_mean_out.expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x = self.std_feature(x)

        # (b, c, 3, n)
        x = torch.max(x, -1, keepdim=False)[0] # (b, c, 3)
        x = x.view(bsz, -1)
        res_feat = self.final(x) # (b, 6)
        res_ori = ortho2rotation(res_feat.view(-1, 2, 3))

        main_ori = equi_out_dict["end_rots"]
        out_ori = torch.bmm(main_ori, res_ori) # no detach here!!!!

        return {
            "mid_rots": main_ori,
            "res_rots": res_ori,
            "end_rots": out_ori,
        }


class TwoStreamInvNaivePointTransformerHead(nn.Module):
    def __init__(self, in_channels, mid_channels, num_blocks):
        super(TwoStreamInvNaivePointTransformerHead, self).__init__()

        self.equi_head = GlobalAvgPoolingHead(in_channels)
        # rotation-invariant head
        self.std_feature = VNStdFeature(in_channels * 2, dim=4, normalize_frame=False, negative_slope=0.0)
        self.inv_head = PointTransformerHead(in_channels * 2, mid_channels, num_blocks)

    def forward(self, x, p, knn_idx, eval=False):
        equi_out_dict = self.equi_head(x, p, knn_idx)

        # rotation-invariant layer
        x_mean_out = x.mean(dim=-1, keepdim=True)
        x_mean = x_mean_out.expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x = self.std_feature(x)

        # (pre) canonicalized p
        mid_rots = equi_out_dict["end_rots"]
        p_can = torch.bmm(p, mid_rots)

        # (b, c, 3, n)
        res_out_dict = self.inv_head(x, p_can)
        res_ori = res_out_dict["end_rots"]
        out_ori = torch.bmm(mid_rots, res_ori)

        return {
            "mid_rots": mid_rots,
            "res_rots": res_ori,
            "end_rots": out_ori,
        }


class TwoStreamInvNaivePointTransformerClsHead(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, num_blocks):
        super(TwoStreamInvNaivePointTransformerClsHead, self).__init__()

        self.equi_head = GlobalAvgPoolingHead(in_channels)
        # rotation-invariant head
        self.std_feature = VNStdFeature(in_channels * 2, dim=4, normalize_frame=False, negative_slope=0.0)
        self.inv_head = PointTransformerClsHead(in_channels * 2, out_channels, mid_channels, num_blocks)

    def forward(self, x, p, knn_idx, eval=False):
        equi_out_dict = self.equi_head(x, p, knn_idx)

        # rotation-invariant layer
        x_mean_out = x.mean(dim=-1, keepdim=True)
        x_mean = x_mean_out.expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x = self.std_feature(x)

        # (pre) canonicalized p
        mid_rots = equi_out_dict["end_rots"]
        p_can = torch.bmm(p, mid_rots)

        # (b, c, 3, n)
        res_out_dict = self.inv_head(x, p_can)
        res_ori = res_out_dict["end_rots"]
        cls_out = res_out_dict["cls_out"]
        out_ori = torch.bmm(mid_rots, res_ori)

        return {
            "mid_rots": mid_rots,
            "res_rots": res_ori,
            "end_rots": out_ori,
            "cls_out": cls_out,
        }


#########################
# Non-equivariant heads #
#########################
class PointTransformerHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, num_blocks: int):
        super(PointTransformerHead, self).__init__()

        self.pre_transformer = nn.Sequential(
            nn.Conv1d(in_channels * 3, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(True)
        )
        self.transformer = nn.ModuleList([])
        for _ in range(num_blocks):
            self.transformer.append(PointTransformerBlock(mid_channels))

        self.final = nn.Sequential(
            nn.Linear(mid_channels, mid_channels, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(True),
            nn.Linear(mid_channels, 6)
        )
        self.gem_pool = GeMPool()

    def forward(self, x, p, knn_idx=None, eval=False):
        if p.shape[1] != 3:
            p = p.transpose(1, 2)

        if knn_idx is None:
            knn_idx = knn(p, 16)

        bsz, _, _, num_points = x.shape

        # Global
        x = x.contiguous().view(bsz, -1, num_points)
        x = self.pre_transformer(x)
        for module in self.transformer:
            x = module(x, p, knn_idx)

        global_feat = self.gem_pool(x).squeeze(-1)
        global_feat = self.final(global_feat)

        global_ori = global_feat.view(-1, 2, 3)
        global_ori = ortho2rotation(global_ori)

        return {"end_rots": global_ori}


class PointTransformerConcatFusionHead(nn.Module):
    def __init__(self, in_channel: int, mid_channel: int, num_blocks: int):
        super(PointTransformerConcatFusionHead, self).__init__()

        self.pre_transformer = nn.Sequential(
            nn.Conv1d(in_channel * 3, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_channel),
            nn.ReLU(True)
        )
        self.transformer = nn.ModuleList([])
        for _ in range(num_blocks):
            self.transformer.append(PointTransformerBlock(mid_channel))
        
        self.final = nn.Sequential(
            nn.Linear(in_channel * 3 + mid_channel, mid_channel, bias=False),
            nn.BatchNorm1d(mid_channel),
            nn.ReLU(True),
            nn.Linear(mid_channel, 6)
        )

        self.gem_pool = GeMPool()

    def forward(self, x, p, knn_idx=None, eval=False):
        if p.shape[1] != 3:
            p = p.transpose(1, 2)

        if knn_idx is None:
            knn_idx = knn(p, 16)
            
        bsz, _, _, num_points = x.shape # (B, C, 3, N)

        # Equivariant global feature
        global_feat_equi = x.mean(dim=-1) # (B, self.in_channel, 3)

        # Global
        x = x.contiguous().view(bsz, -1, num_points)
        x = self.pre_transformer(x)
        for module in self.transformer:
            x = module(x, p, knn_idx)

        # Non-equivariant global feature
        global_feat = self.gem_pool(x).squeeze(-1)

        # Fusion
        global_feat_concat = torch.cat([global_feat_equi.view(bsz, -1), global_feat.view(bsz, -1)], dim=-1)
        global_ori = self.final(global_feat_concat).view(-1, 2, 3)
        global_ori = ortho2rotation(global_ori)

        return {"end_rots": global_ori}


class PointTransformerClsHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int, num_blocks: int):
        super(PointTransformerClsHead, self).__init__()

        self.pre_transformer = nn.Sequential(
            nn.Conv1d(in_channels * 3, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(True)
        )
        self.transformer = nn.ModuleList([])
        for _ in range(num_blocks):
            self.transformer.append(PointTransformerBlock(mid_channels))

        self.cls_head = nn.Sequential(
            nn.Linear(mid_channels, mid_channels, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(True),
            nn.Linear(mid_channels, out_channels)
        )
        self.rot_head = nn.Sequential(
            nn.Linear(mid_channels, mid_channels, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(True),
            nn.Linear(mid_channels, 6)
        )
        self.gem_pool = GeMPool()

    def forward(self, x, p, knn_idx=None, eval=False):
        if p.shape[1] != 3:
            p = p.transpose(1, 2)

        if knn_idx is None:
            knn_idx = knn(p, 16)

        bsz, _, _, num_points = x.shape
        x = x.view(bsz, -1, num_points)
        x = self.pre_transformer(x)
        for module in self.transformer:
            x = module(x, p, knn_idx)

        global_inv_feat = self.gem_pool(x).squeeze(-1) # (B, mid_channels)
        cls_out = self.cls_head(global_inv_feat) # (B, out_channels)
        rot_out = self.rot_head(global_inv_feat)

        axis_out = rot_out.view(-1, 2, 3)
        ori_out = ortho2rotation(axis_out)

        return {"end_rots": ori_out, "cls_out": cls_out}