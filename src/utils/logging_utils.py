"""Logging utilities for experiment tracking."""

from typing import Any

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf


def log_hyperparameters(object_dict: dict[str, Any]) -> None:
    """Log hyperparameters to all loggers.

    Args:
        object_dict: Dictionary containing "cfg", "model", "trainer" keys.
    """
    hparams: dict[str, Any] = {}

    cfg: DictConfig = object_dict.get("cfg", {})
    trainer: Trainer | None = object_dict.get("trainer")

    if not trainer or not trainer.logger:
        return

    hparams["cfg"] = OmegaConf.to_container(cfg, resolve=True)

    # Save number of model parameters
    model = object_dict.get("model")
    if model is not None:
        hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
        hparams["model/params/trainable"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

    loggers: list[Logger] = trainer.logger if isinstance(trainer.logger, list) else [trainer.logger]
    for logger in loggers:
        logger.log_hyperparams(hparams)
