"""ShapeNet dataset and data module for orientation prediction."""

import logging
import os
from typing import Any

import h5py
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset

import lightning.pytorch as L

from src.data.transforms import Compose

NUM_CLASSES = 13

CLASSIDX2NAME = {
    0: "plane", 1: "bench", 2: "cabinet", 3: "car", 4: "chair",
    5: "monitor", 6: "lamp", 7: "speaker", 8: "firearm", 9: "couch",
    10: "table", 11: "cellphone", 12: "watercraft",
}


def _parse_label_in_use(label_input: str | int | None, num_total: int = NUM_CLASSES) -> list[int]:
    if label_input is None:
        return list(range(num_total))
    if isinstance(label_input, int):
        return [label_input]
    return list(map(int, str(label_input).split(",")))


def _load_data(
    mode: str, label_in_use: list[int], data_path: str,
) -> tuple[list[np.ndarray], list[int]]:
    synset_path = os.path.join(data_path, "synsetoffset2category.txt")
    with open(synset_path) as fp:
        fp_lines = fp.readlines()

    fp_dict = {i: line.rstrip("\n").split()[-1] for i, line in enumerate(fp_lines)}
    label_list = [fp_dict[label] for label in label_in_use]

    data, labels = [], []
    train = mode == "train"

    for label, label_name in zip(label_in_use, label_list):
        h5_path = os.path.join(data_path, f"{label_name}.h5")
        assert os.path.exists(h5_path), f"File not found: {h5_path}"
        h5s = h5py.File(h5_path, "r")
        h5s = {int(k): np.asarray(v) for k, v in h5s["pcd"]["point"].items()}
        keys = sorted(h5s.keys())
        split = int(0.8 * len(keys))
        keys = keys[:split] if train else keys[split:]

        for i in keys:
            data.append(np.asarray(h5s[i]))
            labels.append(label)

    return data, labels


def _collate_stability(list_data: list) -> dict[str, Any]:
    list_data = [d for d in list_data if d is not None]
    if not list_data:
        raise ValueError("No data in the batch")
    pcds, rots, labels = zip(*list_data)
    return {
        "batch_size": len(rots),
        "pcd": torch.vstack(pcds),
        "rots": torch.vstack(rots),
        "label": torch.LongTensor(labels),
    }


class ShapeNetDataModule(L.LightningDataModule):
    """Lightning DataModule for ShapeNet orientation prediction."""

    def __init__(
        self,
        label_in_use: str = "0,3,4,10",
        num_points: int = 1024,
        resample: bool = False,
        batch_size: int = 32,
        num_workers: int = 8,
        data_path: str = "data",
        train_transforms: list[DictConfig] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        """Download and preprocess ShapeNet data (called on rank 0 only)."""
        from src.data.prepare import download_shapenet, convert_ply_to_h5, generate_stability_npz

        hp = self.hparams
        label_in_use = _parse_label_in_use(hp.label_in_use)

        # Check if all required h5 files exist
        synset_map = {i: sid for i, (sid, _) in enumerate(
            [("02691156", "plane"), ("02828884", "bench"), ("02933112", "cabinet"),
             ("02958343", "car"), ("03001627", "chair"), ("03211117", "monitor"),
             ("03636649", "lamp"), ("03691459", "speaker"), ("04090263", "firearm"),
             ("04256520", "couch"), ("04379243", "table"), ("04401088", "cellphone"),
             ("04530566", "watercraft")]
        )}
        missing = [
            synset_map[l] for l in label_in_use
            if not os.path.exists(os.path.join(hp.data_path, f"{synset_map[l]}.h5"))
        ]

        if missing:
            print(f"Missing h5 files for synsets: {missing}. Preparing data...")
            os.makedirs(hp.data_path, exist_ok=True)
            ply_dir = download_shapenet(hp.data_path)
            convert_ply_to_h5(ply_dir, hp.data_path)

        # Check stability npz
        npz_path = os.path.join(hp.data_path, "preprocessed_stability_spectral.npz")
        if not os.path.exists(npz_path):
            generate_stability_npz(hp.data_path)

    def setup(self, stage: str | None = None) -> None:
        hp = self.hparams
        label_in_use = _parse_label_in_use(hp.label_in_use)

        # Build transforms
        transforms = None
        if hp.train_transforms:
            transform_list = []
            for t_cfg in hp.train_transforms:
                if isinstance(t_cfg, (DictConfig, dict)):
                    transform_list.append(hydra.utils.instantiate(t_cfg))
                else:
                    transform_list.append(t_cfg)
            transforms = Compose(transform_list)

        if stage == "fit" or stage is None:
            self.train_dset = ShapeNetTrainPair(
                transforms=transforms,
                label_in_use=label_in_use,
                data_path=hp.data_path,
            )

        self.cns_dset = ShapeNetConsistencyTest(
            label_in_use=label_in_use,
            num_points=hp.num_points,
            resample=hp.resample,
            data_path=hp.data_path,
        )
        self.stb_dset = ShapeNetStabilityTest(
            label_in_use=label_in_use,
            num_points=hp.num_points,
            resample=hp.resample,
            data_path=hp.data_path,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> list[DataLoader]:
        cns_loader = DataLoader(
            self.cns_dset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
            pin_memory=True,
        )
        stb_loader = DataLoader(
            self.stb_dset,
            batch_size=self.hparams.batch_size // 8,
            num_workers=self.hparams.num_workers,
            collate_fn=_collate_stability,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
            pin_memory=True,
        )
        return [cns_loader, stb_loader]

    def test_dataloader(self) -> list[DataLoader]:
        return self.val_dataloader()


class ShapeNetTrainPair(Dataset):
    """Training dataset: samples cross-instance pairs with random rotations."""

    def __init__(
        self,
        transforms: Compose | None,
        label_in_use: list[int],
        data_path: str,
    ):
        super().__init__()
        assert transforms is not None
        self.transforms = transforms
        self.label_in_use = label_in_use
        self.data, self.label = _load_data("train", label_in_use, data_path)

        self.index_dict: dict[int, list[int]] = {k: [] for k in label_in_use}
        for idx, lbl in enumerate(self.label):
            self.index_dict[lbl].append(idx)

    def _sample_cross_idx(self, label: int, src_idx: int) -> int:
        trg_idx = np.random.choice(self.index_dict[label])
        if trg_idx == src_idx:
            return self._sample_cross_idx(label, src_idx)
        return trg_idx

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pcd_src = self.data[idx]
        label = self.label[idx]

        # Always sample cross-instance pair
        trg_idx = self._sample_cross_idx(label, idx)
        pcd_trg = self.data[trg_idx]

        pcd_src = self.transforms(pcd_src)[0]
        pcd_trg = self.transforms(pcd_trg)[0]

        rots = Rotation.random(2).as_matrix()
        rotated_pcd_src = np.einsum("nj, ij -> ni", pcd_src, rots[0])
        rotated_pcd_trg = np.einsum("nj, ij -> ni", pcd_trg, rots[1])
        rot_diff = rots[1] @ rots[0].T

        return {
            "pcd_src": torch.Tensor(rotated_pcd_src),
            "pcd_trg": torch.Tensor(rotated_pcd_trg),
            "rots": torch.Tensor(rot_diff),
            "label": self.label_in_use.index(label),
        }

    def __len__(self) -> int:
        return len(self.data)


class ShapeNetConsistencyTest(Dataset):
    """Test dataset for consistency evaluation: predict orientation per instance."""

    def __init__(
        self,
        label_in_use: list[int],
        num_points: int,
        resample: bool,
        data_path: str,
    ):
        super().__init__()
        self.label_in_use = label_in_use
        self.num_points = num_points
        self.resample = resample
        self.data, self.label = _load_data("test", label_in_use, data_path)
        self.label2name = {i: CLASSIDX2NAME[c] for i, c in enumerate(label_in_use)}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pcd = self.data[idx]
        if self.resample:
            pcd = pcd[np.random.choice(len(pcd), self.num_points, replace=False)]
        else:
            pcd = pcd[: self.num_points]

        return {
            "pcd": torch.Tensor(pcd),
            "label": self.label_in_use.index(self.label[idx]),
        }

    def __len__(self) -> int:
        return len(self.data)


class ShapeNetStabilityTest(Dataset):
    """Test dataset for stability evaluation: same instance with multiple random rotations."""

    def __init__(
        self,
        label_in_use: list[int],
        num_points: int,
        resample: bool,
        data_path: str,
    ):
        super().__init__()
        fpath = os.path.join(data_path, "preprocessed_stability_spectral.npz")
        test_data = np.load(fpath)

        self.rotated_pcd = test_data["rotated_pcd"]
        self.rots = test_data["rots"]
        self.label = test_data["label"]
        self.num_points = num_points
        self.resample = resample

        self.label_in_use = label_in_use
        is_in_use = np.zeros_like(self.label, dtype=bool)
        for lbl in label_in_use:
            is_in_use = is_in_use | (self.label == lbl)
        idx_in_use = np.where(is_in_use)[0]

        self.rotated_pcd = self.rotated_pcd[idx_in_use]
        self.rots = self.rots[idx_in_use]
        self.label = self.label[idx_in_use]
        self.label2name = {i: CLASSIDX2NAME[c] for i, c in enumerate(label_in_use)}

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        pcds = self.rotated_pcd[idx][:, : 4 * self.num_points, :]
        rots = self.rots[idx]
        label = self.label[idx]

        if self.resample:
            sampled = [p[np.random.choice(len(p), self.num_points, replace=False)] for p in pcds]
            sampled_pcds = np.stack(sampled)
        else:
            sampled_pcds = pcds[:, : self.num_points, :]

        return (
            torch.Tensor(sampled_pcds),
            torch.Tensor(rots),
            self.label_in_use.index(label),
        )

    def __len__(self) -> int:
        return len(self.rotated_pcd)
