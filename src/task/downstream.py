import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class SmoothedCrossEntropyLoss(nn.Module):
    def __init__(self, eps=0.2):
        super(SmoothedCrossEntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        target = target.contiguous().view(-1)

        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
            
        return loss


class LitOrientationDownstreamBase(pl.LightningModule):
    def __init__(
        self,
        orientation_model,
        downstream_model,
        num_gpus_in_use,
        label2name,
        cfg
    ):
        super(LitOrientationDownstreamBase, self).__init__()
        self.save_hyperparameters(cfg)

        self.orientation_model = orientation_model
        self.downstream_model = downstream_model
        self.sync_dist = num_gpus_in_use > 1
        self.label2name = label2name

        self.freeze_orientation_model()


    def freeze_orientation_model(self):
        for p in self.orientation_model.parameters():
            p.requires_grad = False
        self.orientation_model.eval() # to freeze the batch norm buffers


    @torch.inference_mode()
    def align(self, pcd):
        assert not self.orientation_model.training

        pred_rots = self.orientation_model(pcd)["end_rots"]
        pcd_aligned = torch.bmm(pcd, pred_rots)
        
        return pcd_aligned