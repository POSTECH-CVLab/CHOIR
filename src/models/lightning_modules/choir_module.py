"""CHOIRModule: Lightning module for training and evaluating CHOIR."""

from typing import Any

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchmetrics import CatMetric, MeanMetric

import lightning.pytorch as L

from src.losses import RotationMSELoss
from src.utils.rotation import angular_std


class CHOIRModule(L.LightningModule):
    """Lightning module for CHOIR orientation prediction.

    Trains with cross-instance consistency loss: given pairs of different instances
    from the same class with known relative rotation, minimizes MSE between predicted
    and ground-truth relative rotation.
    """

    def __init__(
        self,
        net: nn.Module,
        optimizer: DictConfig,
        scheduler: DictConfig | None = None,
        scheduler_interval: str = "step",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net", "optimizer", "scheduler"], logger=False)
        self.net = net
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler
        self.criterion = RotationMSELoss()

    def setup(self, stage: str | None = None) -> None:
        """Initialize metrics after datamodule is available."""
        if stage in ("fit", "test", None):
            dm = self.trainer.datamodule
            self.label2name = dm.cns_dset.label2name
            num_classes = len(self.label2name)
            self.val_cns_metrics = nn.ModuleList(
                [CatMetric() for _ in range(num_classes)]
            )
            self.val_stb_metrics = nn.ModuleList(
                [MeanMetric() for _ in range(num_classes)]
            )

    def forward(self, pcd: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.net(pcd)

    def configure_optimizers(self) -> dict[str, Any]:
        if callable(self.optimizer_cfg):
            optimizer = self.optimizer_cfg(params=self.parameters())
        else:
            optimizer = hydra.utils.instantiate(self.optimizer_cfg, params=self.parameters())

        result: dict[str, Any] = {"optimizer": optimizer}

        if self.scheduler_cfg is not None:
            total_steps = self.trainer.estimated_stepping_batches
            if callable(self.scheduler_cfg):
                scheduler = self.scheduler_cfg(optimizer=optimizer, total_steps=total_steps)
            else:
                scheduler = hydra.utils.instantiate(
                    self.scheduler_cfg, optimizer=optimizer, total_steps=total_steps,
                )
            result["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": self.hparams.scheduler_interval,
            }

        return result

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        pcd = torch.cat([batch["pcd_src"], batch["pcd_trg"]], dim=0)
        rots_diff = batch["rots"]
        bsz = len(rots_diff)

        output = self.net(pcd)
        end_rots1, end_rots2 = output["end_rots"].chunk(2, dim=0)

        loss = self.criterion(end_rots1, end_rots2, rots_diff)

        self.log("train/loss", loss.item(), batch_size=bsz, prog_bar=True)
        return loss

    def validation_step(
        self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0,
    ) -> None:
        if dataloader_idx == 0:
            self._consistency_step(batch)
        elif dataloader_idx == 1:
            self._stability_step(batch)

    def test_step(
        self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0,
    ) -> None:
        self.validation_step(batch, batch_idx, dataloader_idx)

    def _consistency_step(self, batch: dict[str, torch.Tensor]) -> None:
        output = self.net(batch["pcd"])
        for label, rot in zip(batch["label"].tolist(), output["end_rots"]):
            self.val_cns_metrics[label].update(rot.unsqueeze(0))

    def _stability_step(self, batch: dict[str, torch.Tensor]) -> None:
        output = self.net(batch["pcd"])
        batched_rots_can = torch.bmm(batch["rots"].transpose(1, 2), output["end_rots"])
        list_rots_can = batched_rots_can.chunk(batch["batch_size"], dim=0)
        labels = batch["label"].tolist()
        for label, rots_can in zip(labels, list_rots_can):
            std = angular_std(rots_can).item()
            self.val_stb_metrics[label].update(std)

    def _compute_metrics(self) -> dict[str, dict[str, torch.Tensor]]:
        """Returns {class_name: {"consistency": ..., "stability": ...}} per class."""
        classwise = {}
        for label_idx, (cns_metric, stb_metric) in enumerate(
            zip(self.val_cns_metrics, self.val_stb_metrics)
        ):
            name = self.label2name[label_idx]
            cns = angular_std(cns_metric.compute())
            stb = stb_metric.compute()
            classwise[name] = {"consistency": cns, "stability": stb}
            cns_metric.reset()
            stb_metric.reset()
        return classwise

    def _log_metrics(self, prefix: str, classwise: dict) -> None:
        log_dict = {}
        avg_list = []
        for name, metrics in classwise.items():
            cns, stb = metrics["consistency"], metrics["stability"]
            avg = (cns + stb) / 2
            log_dict[f"{prefix}/{name}_cns"] = cns
            log_dict[f"{prefix}/{name}_stb"] = stb
            log_dict[f"{prefix}/{name}_avg"] = avg
            avg_list.append(avg)
        log_dict[f"{prefix}/avg"] = sum(avg_list) / len(avg_list)

        self.log_dict(log_dict, logger=True, sync_dist=(prefix == "val"))

    def on_validation_epoch_end(self) -> None:
        self._log_metrics("val", self._compute_metrics())

    def on_test_epoch_end(self) -> None:
        self._log_metrics("test", self._compute_metrics())
