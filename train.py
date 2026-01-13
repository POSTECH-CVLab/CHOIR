import os
from datetime import datetime

import torch
import pytorch_lightning as pl

from src.data.shapenet import ShapeNetDataModule
from src.task.orientation import LitOrientationModule, LitOrientationModuleV2
from src.model.models import build_model
import src.data.transforms as T


def train(cfg):
    # Setup wandb and others.
    now = datetime.now().strftime('%m-%d-%H-%M-%S')
    name = now + "_" + cfg.name
    save_dir = f"logs/{name}"
    os.makedirs(save_dir, exist_ok=True)
    num_gpus_in_use = min(torch.cuda.device_count(), cfg.num_gpus)

    callbacks = [
        pl.callbacks.TQDMProgressBar(),
        pl.callbacks.ModelCheckpoint(
            dirpath=save_dir, filename="best", monitor="val/average", save_last=True, save_top_k=1, mode="min"
        ),
        pl.callbacks.LearningRateMonitor(),
    ]
    loggers = [
        pl.loggers.WandbLogger(
            name=name,
            save_dir=save_dir,
            project="char_rot",
            log_model=True,
            entity="postech_cvlab",
            config=cfg
        )
    ]

    # Dataloaders: augmentation + dataset
    transforms = [T.PointSampling(4 * cfg.num_points, cfg.sample_alg)]
    if cfg.use_kpr: # knn patch removal
        transforms.append(T.KnnPatchRemoval(cfg.kpr_k, cfg.num_patches))
    if cfg.use_gn: # Gaussian noise
        transforms.append(T.GaussianNoise(cfg.noise_amp))
    transforms.append(T.PointSampling(cfg.num_points, cfg.sample_alg))
    transforms = T.Compose(transforms)

    if cfg.dataset == "shapenet":
        data_module = ShapeNetDataModule(
            transforms, cfg.label_in_use, cfg.cross_p, cfg.num_points,
            cfg.resample, cfg.batch_size, cfg.num_workers, cfg.shapenet_path
        )
        data_module.setup("fit")
    else:
        raise NotImplementedError

    # Model and optimizer TODO(chrockey): resuming
    if cfg.task == "orientation":
        cfg.pt_out_channels = len(data_module.train_dset.label_in_use) # out_channels == num_classes
        model = build_model(cfg)

        module_args = (model, num_gpus_in_use, data_module.cns_dset.label2name, cfg)
        if cfg.version == 1:
            pl_module = LitOrientationModule(*module_args)
        elif cfg.version == 2:
            pl_module = LitOrientationModuleV2(*module_args)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # Train
    kwargs = dict()
    kwargs["default_root_dir"] = save_dir
    kwargs["max_epochs"] = cfg.num_epochs
    kwargs["accelerator"] = "gpu"
    kwargs["devices"] = num_gpus_in_use
    kwargs["callbacks"] = callbacks
    kwargs["logger"] = loggers
    kwargs["log_every_n_steps"] = cfg.log_freq
    kwargs["check_val_every_n_epoch"] = cfg.val_freq
    kwargs["num_sanity_val_steps"] = 0
    if num_gpus_in_use > 1:
        kwargs["replace_sampler_ddp"] = True
        kwargs["sync_batchnorm"] = True
        kwargs["strategy"] = "ddp_find_unused_parameters_false"
    
    trainer = pl.Trainer(**kwargs)

    trainer.fit(pl_module, data_module)

    if num_gpus_in_use > 1:
        torch.distributed.destroy_process_group()

    if trainer.is_global_zero:
        kwargs["devices"] = 1
        kwargs["replace_sampler_ddp"] = False
        kwargs["sync_batchnorm"] = False
        kwargs["strategy"] = None
        kwargs.pop("callbacks")

        trainer = pl.Trainer(**kwargs)
        trainer.test(pl_module, data_module, ckpt_path=os.path.join(save_dir, "best.ckpt"))


if __name__ == "__main__":
    from config import get_config

    cfg = get_config()
    pl.seed_everything(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = cfg.not_use_tf32

    train(cfg)