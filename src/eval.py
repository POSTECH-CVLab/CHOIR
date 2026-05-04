"""CHOIR evaluation entry point."""

import hydra
import rootutils
from omegaconf import DictConfig

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import lightning.pytorch as L  # noqa: E402

from src.models.lightning_modules.choir_module import CHOIRModule  # noqa: E402
from src.utils.instantiators import instantiate_callbacks  # noqa: E402


def evaluate(cfg: DictConfig) -> None:
    """Main evaluation function.

    Args:
        cfg: Hydra config.
    """
    assert cfg.ckpt_path, "ckpt_path is required for evaluation"

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    model = CHOIRModule.load_from_checkpoint(cfg.ckpt_path)

    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks,
    )

    trainer.test(model=model, datamodule=datamodule)


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
