# *_*coding:utf-8 *_*
# Ref: https://github.com/FlyingGiraffe/vnn/blob/master/data_utils/ShapeNetDataLoader.py
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class PartNormalFourClassesDataset(Dataset):
    def __init__(
        self,
        root='data/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0_normal',
        npoints=2500,
        split='train',
        class_choice=["airplane", "car", "chair", "table"], # label_in_use
        normal_channel=False
    ):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            all_classes = [k.lower() for k in self.cat.keys()]
            for c in class_choice:
                if c not in all_classes:
                    raise ValueError("Invalid class choices")

            self.cat = {k:v for k,v in self.cat.items() if k.lower() in class_choice}

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))

            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {
            'Airplane': [0, 1, 2, 3],
            'Car': [8, 9, 10, 11],
            'Chair': [12, 13, 14, 15],
            'Table': [47, 48, 49]
        }
        self.seg_classes_mapped = {
            'Airplane': [0, 1, 2, 3],
            'Car': [4, 5, 6, 7],
            'Chair': [8, 9, 10, 11],
            'Table': [12, 13, 14]
        }
        self.cls_map = {
            'Airplane': 0,
            'Car': 1,
            'Chair': 2,
            'Table': 3
        }
        self.seg_map = {
            0: 0, 1: 1, 2: 2, 3: 3,
            8: 4, 9: 5, 10: 6, 11: 7,
            12: 8, 13: 9, 14: 10, 15: 11,
            47: 12, 48: 13, 49: 14
        }
        self.seg_label_to_cat = {}
        for cat in self.seg_classes_mapped.keys():
            for label in self.seg_classes_mapped[cat]:
                self.seg_label_to_cat[label] = cat

        self.num_classes = 4
        self.num_part = 15

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000


    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.cls_map[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1]
            seg = np.array([self.seg_map[x] for x in seg]).astype(np.int32)

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg, index

    def __len__(self):
        return len(self.datapath)


if __name__ == "__main__":
    dset = PartNormalFourClassesDataset(
        root="data/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0_normal/",
        npoints=1024,
        split="trainval",
        normal_channel=False
    )
    sample = dset.__getitem__(0)

    import pdb;pdb.set_trace()