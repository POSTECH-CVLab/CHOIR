"""CHOIR training entry point."""

import os
from typing import Any

import hydra
import rootutils
from omegaconf import DictConfig, OmegaConf

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import lightning.pytorch as L  # noqa: E402

from src.utils.instantiators import instantiate_callbacks, instantiate_loggers  # noqa: E402
from src.utils.logging_utils import log_hyperparameters  # noqa: E402


def train(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra config.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Instantiate components
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    loggers = instantiate_loggers(cfg.get("logger"))

    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers,
    )

    # Log hyperparameters
    object_dict = {"cfg": cfg, "model": model, "trainer": trainer}
    log_hyperparameters(object_dict)

    # Train
    trainer.fit(model=model, datamodule=datamodule)

    # Test after training
    if cfg.get("test_after_training"):
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path:
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        else:
            trainer.test(model=model, datamodule=datamodule)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
