import os
import argparse

from tqdm import tqdm
import numpy as np

from src.data.shapenet import NUM_CLASSES, load_data
from src.utils import fix_seed
from scipy.spatial.transform import Rotation

# classname: classid -> (min, max)
# --------------------------------
# airplane: 0 -> (12614, 30000)
# car: 3 -> (30000, 30000)
# chair: 4 -> (21270, 30000)
# table: 10 -> (20692, 30000)
# --------------------------------


def preprocess(args):
    os.makedirs(args.out_path, exist_ok=True)

    all_data, all_label = load_data("test", list(range(NUM_CLASSES)), args.in_path)
    ret = {
        "rotated_pcd": [], # (B, R, N, 3)
        "rots": [], # (B, R, 3, 3)
        "label": [], # (B,)
    }
    for pcd, label in tqdm(zip(all_data, all_label)):
        if label not in [0, 3, 4, 10]: # plane, car, chair, table
            continue

        # sampling
        pcd = pcd[:args.num_points]
        rots = Rotation.random(args.num_rotations).as_matrix()
        rot_pcds = np.stack([np.einsum("nj, ij -> ni", pcd, rot) for rot in rots])

        ret["rotated_pcd"].append(rot_pcds)
        ret["rots"].append(rots)
        ret["label"].append(label)

    ret["rotated_pcd"] = np.stack(ret["rotated_pcd"])
    ret["rots"] = np.stack(ret["rots"])
    ret["label"] = np.array(ret["label"])

    fpath = os.path.join(args.out_path, f"preprocessed_stability_c=0-3-4-10_n={args.num_points}")
    np.savez(
        fpath,
        **ret,
    )


if __name__ == "__main__":
    # To evaluate the stability, we should fix the dataset.
    # For consistency evaluation, we don't need additional preprocessing steps. 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--in_path", type=str, default="data/shapenet", help="path to the original ShapeNet"
    )
    parser.add_argument(
        "-o", "--out_path", type=str, default="data/shapenet",
        help="path to store preprocessed data to evaluate the stability"
    )
    parser.add_argument(
        "-n", "--num_points", type=int, default=10000,
        help="the number of points to sample"
    )
    parser.add_argument(
        "-r", "--num_rotations", type=int, default=10,
        help="the number of rotations for the stability"
    )
    args = parser.parse_args()

    fix_seed(0)
    preprocess(args)