import os
import json

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation

from src.utils import parse_label_in_use
from src.data.collate_fns import collate_shapenet_stability


NUM_CLASSES = 13


class ShapeNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_transforms,
        label_in_use,
        cross_p,
        num_points,
        resample,
        batch_size,
        num_workers,
        data_path,
    ):
        super(ShapeNetDataModule, self).__init__()

        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dset = ShapeNetTrainPair(
                self.train_transforms, self.label_in_use, self.cross_p, self.data_path
            )
        self.cns_dset = ShapeNetConsistencyTest(
            self.label_in_use, self.num_points, self.resample, self.data_path
        )
        self.stb_dset = ShapeNetStabilityTest(
            self.label_in_use, self.num_points, self.resample, self.data_path
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=True, drop_last=True, persistent_workers=True, pin_memory=True
        )

    def val_dataloader(self):
        cns_loader = DataLoader(
            self.cns_dset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False, drop_last=False, persistent_workers=True, pin_memory=True
        )
        stb_loader = DataLoader(
            self.stb_dset, batch_size=self.batch_size // 8, num_workers=self.num_workers, collate_fn=collate_shapenet_stability,
            shuffle=False, drop_last=False, persistent_workers=True, pin_memory=True
        )
        return [cns_loader, stb_loader]

    def test_dataloader(self):
        return self.val_dataloader()


##########################
# Orientation Prediction #
##########################
def load_data(mode, label_in_use, data_path):    
    with open("data/synsetoffset2category.txt") as fp:
        fp_lines = fp.readlines()

    fp_dict = {i: k for (i, k) in enumerate([line.rstrip("\n").split()[-1] for line in fp_lines])}
    
    label_list = [fp_dict[label] for label in label_in_use]
    data = []
    labels = []

    train = mode == "train"

    for (label, label_name) in zip(label_in_use, label_list):
        h5_path = os.path.join(data_path, f"{label_name}.h5")
        assert os.path.exists(h5_path)
        h5s = h5py.File(h5_path, "r")
        h5s = {int(k): np.asarray(v) for (k, v) in h5s["pcd"]["point"].items()}
        keys = list(h5s.keys())
        if train:
            keys = keys[:int(0.8 * len(keys))]
        else:
            keys = keys[int(0.8 * len(keys)):]

        for i in sorted(keys):
            data.append(np.asarray(h5s[i]))
            labels.append(label)

    return data, labels


class ShapeNetTrainPair(Dataset):
    def __init__(
        self,
        transforms,
        label_in_use,
        cross_p,
        data_path,
    ):
        super(ShapeNetTrainPair, self).__init__()
        assert transforms is not None

        self.transforms = transforms
        self.label_in_use = parse_label_in_use(label_in_use, NUM_CLASSES)
        self.data, self.label = load_data("train", self.label_in_use, data_path)
        self.cross_p = cross_p

        self.index_dict = {k: [] for k in self.label_in_use}
        for idx, label in enumerate(self.label):
            self.index_dict[label].append(idx)


    def sample_cross_idx(self, label, src_idx):
        trg_idx = np.random.choice(self.index_dict[label])
        
        if trg_idx == src_idx:
            trg_idx = self.sample_cross_idx(label, src_idx)

        return trg_idx


    def __getitem__(self, idx):
        pcd_src = self.data[idx]
        label = self.label[idx]

        is_cross = np.random.random() < self.cross_p
        trg_idx = self.sample_cross_idx(label, idx) if is_cross else idx
        pcd_trg = self.data[trg_idx]

        pcd_src = self.transforms(pcd_src)[0]
        pcd_trg = self.transforms(pcd_trg)[0]

        rots = Rotation.random(2).as_matrix() # one for source the other for target. det = 1
        rotated_pcd_src = np.einsum("nj, ij -> ni", pcd_src, rots[0])
        rotated_pcd_trg = np.einsum("nj, ij -> ni", pcd_trg, rots[1])
        rot_diff = rots[1] @ rots[0].T

        ret = dict()
        ret["pcd_src"] = torch.Tensor(rotated_pcd_src)
        ret["pcd_trg"] = torch.Tensor(rotated_pcd_trg)
        ret["rots"] = torch.Tensor(rot_diff)
        ret["label"] = self.label_in_use.index(label)
        ret["cls_idx"] = label
        ret["is_cross"] = is_cross

        return ret


    def __len__(self):
        return len(self.data)


class ShapeNetConsistencyTest(Dataset):
    def __init__(
        self,
        label_in_use,
        num_points,
        resample,
        data_path,
    ):
        super(ShapeNetConsistencyTest, self).__init__()

        self.label_in_use = parse_label_in_use(label_in_use, NUM_CLASSES)
        self.num_points = num_points
        self.resample = resample
        self.data, self.label = load_data("test", self.label_in_use, data_path)
        self.classidx2name = {
            0: "plane", 1: "bench", 2: "cabinet", 3: "car", 4: "chair",
            5: "monitor", 6: "lamp", 7: "speaker", 8: "firearm", 9: "couch",
            10: "table", 11: "cellphone", 12: "watercraft"
        }
        self.label2name = {i: self.classidx2name[c] for i, c in enumerate(self.label_in_use)}

    def __getitem__(self, idx):
        pcd = self.data[idx]
        label = self.label[idx]

        if self.resample:
            pcd = pcd[np.random.choice(len(pcd), self.num_points, replace=False)]
        else:
            pcd = pcd[:self.num_points]

        ret = dict()
        ret["pcd"] = torch.Tensor(pcd)
        ret["label"] = self.label_in_use.index(label)

        return ret


    def __len__(self):
        return len(self.data)


class ShapeNetStabilityTest(Dataset):
    def __init__(
        self,
        label_in_use,
        num_points,
        resample,
        data_path,
        noise=False,
        knn_removal=False
    ):
        super(ShapeNetStabilityTest, self).__init__()

        fpath = f"{data_path}/preprocessed_stability_c=0-3-4-10_n=10000.npz"
        test_preprocessed = np.load(fpath)

        self.rotated_pcd = test_preprocessed["rotated_pcd"]
        self.rots = test_preprocessed["rots"]
        self.label = test_preprocessed["label"]
        self.num_points = num_points
        self.resample = resample
        self.noise = noise
        self.knn_removal = knn_removal

        self.label_in_use = parse_label_in_use(label_in_use, NUM_CLASSES)
        is_in_use = np.zeros_like(self.label, bool)
        for label in self.label_in_use:
            is_in_use = is_in_use | (self.label == label)
        idx_in_use = np.where(is_in_use)[0]

        self.rotated_pcd = self.rotated_pcd[idx_in_use]
        self.rots = self.rots[idx_in_use]
        self.label = self.label[idx_in_use]
        self.classidx2name = {
            0: "plane", 1: "bench", 2: "cabinet", 3: "car", 4: "chair",
            5: "monitor", 6: "lamp", 7: "speaker", 8: "firearm", 9: "couch",
            10: "table", 11: "cellphone", 12: "watercraft"
        }
        self.label2name = {i: self.classidx2name[c] for i, c in enumerate(self.label_in_use)}

    def __getitem__(self, idx):
        pcds = self.rotated_pcd[idx] # (10, 10000, 3)
        pcds = pcds[:, :4*self.num_points, :]
        rots = self.rots[idx] # (10, 3, 3)
        label = self.label[idx] # int

        if self.knn_removal:
            sampled_pcds = np.stack([self.remove_knn_patch(x, 100) for x in pcds])
            sampled_pcds = sampled_pcds[:, :self.num_points, :]

        if self.resample:
            sampled_pcds = []
            for pcd in pcds:
                sampled_pcds.append(pcd[np.random.choice(len(pcd), self.num_points, replace=False)])
            sampled_pcds = np.stack(sampled_pcds)
        else:
            sampled_pcds = pcds[:, :self.num_points, :]

        if self.noise:
            sampled_pcds = sampled_pcds + 0.0125*np.random.randn(*sampled_pcds.shape)

        return (
            torch.Tensor(sampled_pcds),
            torch.Tensor(rots),
            self.label_in_use.index(label)
        )
    

    def remove_knn_patch(self, coord, k):
        query = coord[np.random.choice(range(len(coord)), 1, replace=False)]
        rel = query[None, None, ...] - coord[None, ...]
        squared_dist = np.sum(rel ** 2, axis=-1, keepdims=False)
        indices = np.argsort(squared_dist, axis=-1)[:, k:]
        return coord[indices]


    def __len__(self):
        return len(self.rotated_pcd)