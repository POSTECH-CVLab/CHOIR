import torch
import torch.nn as nn
from torchmetrics import CatMetric, MeanMetric
import pytorch_lightning as pl

from src.rotation import cal_angular_std_from_rotations


class GlobalMSE(nn.Module):
    def __init__(self):
        super(GlobalMSE, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, rots_src, rots_trg, rots_diff):
        rots_diff_pred = torch.bmm(rots_trg, rots_src.transpose(1, 2))
        return self.mse_loss(rots_diff_pred, rots_diff)


def mean(values: list):
    return sum(values) / len(values)


class LitOrientationModule(pl.LightningModule):
    def __init__(
        self,
        model,
        num_gpus_in_use,
        label2name,
        cfg
    ):
        super(LitOrientationModule, self).__init__()
        self.save_hyperparameters(cfg)

        self.model = model
        self.sync_dist = num_gpus_in_use > 1
        self.label2name = label2name

        self.criterions = nn.ModuleDict({
            "orientation": GlobalMSE(),
            "classification": nn.CrossEntropyLoss()
        })
        self.use_residual = "twostream" in self.hparams.head
        self.use_cls = "cls" in self.hparams.head
        self.val_cns_metrics = nn.ModuleList([CatMetric(dist_snyc_on_step=self.sync_dist) for _ in range(len(label2name))])
        self.val_stb_metric = MeanMetric(dist_sync_on_step=self.sync_dist)


    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay
            )
        else:
            raise NotImplementedError

        if self.hparams.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.hparams.num_epochs, 0
            )
        else:
            raise NotImplementedError

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }
    

    def training_step(self, batch, batch_idx):
        pcd = torch.cat([batch["pcd_src"], batch["pcd_trg"]], dim=0)
        rots_diff = batch["rots"]
        is_cross = batch["is_cross"]
        num_cross = is_cross.sum().item()
        bsz = len(rots_diff)
        is_cross_only = num_cross == bsz
        is_intra_only = num_cross == 0

        output = self.model(pcd)

        # orientation losses: stability and consistency
        end_rots1, end_rots2 = output["end_rots"].chunk(2, dim=0)
        if self.use_residual:
            mid_rots1, mid_rots2 = output["mid_rots"].chunk(2, dim=0)
        else:
            mid_rots1, mid_rots2 = end_rots1, end_rots2
            
        stb_loss = self.criterions["orientation"](
            mid_rots1[~is_cross], mid_rots2[~is_cross], rots_diff[~is_cross]
        ) if not is_cross_only else torch.zeros(1, device=pcd.device)
        cns_loss = self.criterions["orientation"](
            end_rots1[is_cross], end_rots2[is_cross], rots_diff[is_cross]
        ) if not is_intra_only else torch.zeros(1, device=pcd.device)

        # (optional) classification loss
        if self.use_cls:
            is_cross_double = is_cross.repeat(2)
            label = batch["label"].repeat(2)
            cls_loss = self.criterions["classification"](
                output["cls_out"][is_cross_double], label[is_cross_double]
            )
        else:
            cls_loss = torch.zeros(1, device=pcd.device)

        loss = stb_loss + cns_loss + cls_loss
        
        log_dict = {
            "train/loss": loss.item(),
            "train/stability_loss": stb_loss.item(),
            "train/consistency_loss": cns_loss.item(),
            "train/classification_loss": cls_loss.item()
        }
        self.log_dict(log_dict, batch_size=bsz, logger=True)

        return loss


    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            self.consistency_step(batch, batch_idx)
        elif dataloader_idx == 1:
            self.stability_step(batch, batch_idx)
        else:
            raise ValueError("Invalid number of dataloaders")


    def test_step(self, batch, batch_idx, dataloader_idx):
        self.validation_step(batch, batch_idx, dataloader_idx)


    def consistency_step(self, batch, batch_idx):
        output = self.model(batch["pcd"], eval=True)
        label = batch["label"].tolist()

        for l, rot in zip(label, output["end_rots"]):
            self.val_cns_metrics[l].update(rot.unsqueeze(0))


    def stability_step(self, batch, batch_idx):
        output = self.model(batch["pcd"], eval=True)

        batched_rots_can = torch.bmm(batch["rots"].transpose(1, 2), output["end_rots"])
        list_rots_can = batched_rots_can.chunk(batch["batch_size"], dim=0)
        for rots_can in list_rots_can:
            std = cal_angular_std_from_rotations(rots_can).item()
            self.val_stb_metric.update(std)


    def calculate_metrics(self, outputs):
        val_cns_classwise = dict()
        for l, val_cns_metric in enumerate(self.val_cns_metrics):
            rots = val_cns_metric.compute()
            std = cal_angular_std_from_rotations(rots)
            val_cns_classwise[self.label2name[l]] = std
            val_cns_metric.reset()
        val_stb = self.val_stb_metric.compute()
        self.val_stb_metric.reset()

        return val_cns_classwise, val_stb


    def validation_epoch_end(self, outputs):
        val_cns_classwise, val_stb = self.calculate_metrics(outputs)
        val_cns = mean(list(val_cns_classwise.values()))

        log_dict = {
            "val/consistency": val_cns,
            "val/stability": val_stb,
            "val/average": (val_cns + val_stb) / 2
        }
        for k, v in val_cns_classwise.items():
            log_dict[f"val_consistency/{k}"] = v
        self.log_dict(log_dict, logger=True, sync_dist=self.sync_dist)


    def test_epoch_end(self, outputs):
        test_cns_classwise, test_stb = self.calculate_metrics(outputs)
        test_cns = mean(list(test_cns_classwise.values()))

        log_dict = {
            "test/consistency": test_cns,
            "test/stability": test_stb,
            "test/average": (test_cns + test_stb) / 2
        }
        for k, v in test_cns_classwise.items():
            log_dict[f"test_consistency/{k}"] = v
        self.log_dict(log_dict, logger=True)


class LitOrientationModuleV2(LitOrientationModule):    
    def training_step(self, batch, batch_idx):
        pcd = torch.cat([batch["pcd_src"], batch["pcd_trg"]], dim=0)
        rots_diff = batch["rots"]
        is_cross = batch["is_cross"]
        num_cross = is_cross.sum().item()
        bsz = len(rots_diff)
        is_cross_only = num_cross == bsz
        is_intra_only = num_cross == 0

        output = self.model(pcd)

        # orientation losses: stability and consistency
        end_rots1, end_rots2 = output["end_rots"].chunk(2, dim=0)            
        stb_loss = self.criterions["orientation"](
            end_rots1[~is_cross], end_rots2[~is_cross], rots_diff[~is_cross]
        ) if not is_cross_only else torch.zeros(1, device=pcd.device)
        cns_loss = self.criterions["orientation"](
            end_rots1[is_cross], end_rots2[is_cross], rots_diff[is_cross]
        ) if not is_intra_only else torch.zeros(1, device=pcd.device)

        # (optional) classification loss
        if self.use_cls:
            is_cross_double = is_cross.repeat(2)
            label = batch["label"].repeat(2)
            cls_loss = self.criterions["classification"](
                output["cls_out"][is_cross_double], label[is_cross_double]
            )
        else:
            cls_loss = torch.zeros(1, device=pcd.device)

        loss = stb_loss + cns_loss + cls_loss
        
        log_dict = {
            "train/loss": loss.item(),
            "train/stability_loss": stb_loss.item(),
            "train/consistency_loss": cns_loss.item(),
            "train/classification_loss": cls_loss.item()
        }
        self.log_dict(log_dict, batch_size=bsz, logger=True)

        return loss