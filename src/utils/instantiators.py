"""Hydra instantiation helpers for callbacks and loggers."""

from typing import Any

import hydra
from lightning.pytorch import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig


def instantiate_callbacks(callbacks_cfg: DictConfig | None) -> list[Callback]:
    """Instantiate callbacks from Hydra config.

    Args:
        callbacks_cfg: DictConfig with callback definitions.

    Returns:
        List of instantiated callbacks.
    """
    callbacks: list[Callback] = []
    if not callbacks_cfg:
        return callbacks

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig | None) -> list[Logger]:
    """Instantiate loggers from Hydra config.

    Args:
        logger_cfg: DictConfig with logger definitions.

    Returns:
        List of instantiated loggers.
    """
    loggers: list[Logger] = []
    if not logger_cfg:
        return loggers

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers
