import torch.nn as nn

from src.model.vnt_dgcnn import VNTSimpleEncoder, VNTLargeEncoder, IdentityEncoder
from src.model.heads import (
    GlobalAvgPoolingHead,
    GlobalAvgPoolingReLUHead,
    GlobalGeMPoolingHead,
    GlobalGeMPoolingReLUHead,
    PointTransformerHead,
    PointTransformerConcatFusionHead,
    TwoStreamInvHead,
    TwoStreamInvNaiveHead,
    TwoStreamInvNaivePointTransformerHead,
    TwoStreamInvNaivePointTransformerClsHead,
)


def build_head(equi_feat_dim, cfg):
    if cfg.head == "avg":
        head = GlobalAvgPoolingHead(equi_feat_dim)
    elif cfg.head == "avg-relu":
        head = GlobalAvgPoolingReLUHead(equi_feat_dim)
    elif cfg.head == "gem":
        head = GlobalGeMPoolingHead(equi_feat_dim)
    elif cfg.head == "gem-relu":
        head = GlobalGeMPoolingReLUHead(equi_feat_dim)
    elif cfg.head == "pt":
        head = PointTransformerHead(equi_feat_dim, cfg.pt_mid_channels, cfg.pt_num_blocks)
    elif cfg.head == "pt-concat_fusion":
        head = PointTransformerConcatFusionHead(
            equi_feat_dim, cfg.pt_mid_channels, cfg.pt_num_blocks
        )
    elif cfg.head == "twostream-inv":
        head = TwoStreamInvHead(equi_feat_dim)
    elif cfg.head == "twostream-inv-naive":
        head = TwoStreamInvNaiveHead(equi_feat_dim)
    elif cfg.head == "twostream-inv-naive-pt":
        head = TwoStreamInvNaivePointTransformerHead(
            equi_feat_dim, cfg.pt_mid_channels, cfg.pt_num_blocks
        )
    elif cfg.head == "twostream-inv-naive-pt-ema":
        raise NotImplementedError("Deprecated")
    elif cfg.head == "twostream-inv-naive-pt-cls":
        head = TwoStreamInvNaivePointTransformerClsHead(
            equi_feat_dim, cfg.pt_out_channels, cfg.pt_mid_channels, cfg.pt_num_blocks
        )
    else:
        raise NotImplementedError

    return head


def build_backbone(cfg):
    if cfg.backbone == "vnt":
        backbone = VNTSimpleEncoder(
            cfg.vnt_k, cfg.vnt_pooling, cfg.vnt_base_ch, cfg.vnt_nlatent, cfg.vnt_not_use_ft, cfg.vnt_norm
        )
    elif cfg.backbone == "vnt-large":
        backbone = VNTLargeEncoder(
            cfg.vnt_k, cfg.vnt_pooling, cfg.vnt_base_ch, cfg.vnt_nlatent, cfg.vnt_not_use_ft, cfg.vnt_norm
        )
    elif cfg.backbone == "identity":
        backbone = IdentityEncoder()
    else:
        raise NotImplementedError

    return backbone


def build_model(cfg):
    backbone = build_backbone(cfg)
    head = build_head(backbone.output_ch // 3, cfg)

    return OrientationPredictor(backbone, head)


class OrientationPredictor(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super(OrientationPredictor, self).__init__()

        self.backbone = backbone
        self.head = head


    def forward(self, pcd, knn_idx=None, eval=False):
        feat = self.backbone(pcd, knn_idx) # translation-invariant and rotation-equivariant features
        out_dict = self.head(feat, pcd, knn_idx, eval)

        return out_dict