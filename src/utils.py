import random
import numpy as np
import torch
import open3d as o3d


class DotDict(dict):
    """
    from https://stackoverflow.com/questions/13520421/recursive-dotdict
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        if dct is not None:
            for key, value in dct.items():
                if hasattr(value, "keys"):
                    value = DotDict(value)
                self[key] = value


def fix_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic


def parse_label_in_use(label_input, num_total_label):

    if label_input is None:
        label_in_use = [i for i in range(num_total_label)]
    elif isinstance(label_input, int):
        label_in_use = [label_input]
    else:
        label_in_use = list(map(int, label_input.split(",")))

    return label_in_use


def save_pcd(pcd, fname, color=None):
    if isinstance(pcd, torch.Tensor):
        if pcd.device != torch.device("cpu"):
            pcd = pcd.cpu().numpy()
        else:
            pcd = pcd.numpy()

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd)
    if color is not None:
        assert isinstance(color, np.ndarray)
        o3d_pcd.colors = o3d.utility.Vector3dVector(color)

    o3d.io.write_point_cloud(fname, o3d_pcd)