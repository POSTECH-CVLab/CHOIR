import argparse
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import pytorch_lightning as pl

from src.data.shapenet import load_data


def mean(values: list):
    return sum(values) / len(values)


# @torch.inference_mode()
# def knn(x, k):
#     inner = -2 * torch.matmul(x, x.transpose(2, 1)) # (B, N, N)
#     xx = torch.sum(x**2, dim=-1, keepdim=True)
#     pairwise_distance = -xx - inner -xx.transpose(2, 1)
#     idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
#     return idx


@torch.inference_mode()
def knn(x, k):
    rel = x.unsqueeze(2) - x.unsqueeze(1)
    dist = torch.sum(rel ** 2, dim=-1)
    knn = dist.topk(k=k, dim=-1, largest=False, sorted=True)
    return knn.indices


def main_scannet(args):
    pcd = torch.load("/root/data/scannetv2/scannetv2_preprocessed/train/scene0000_00.pth")
    pcd = pcd['coord']
    pcd = pcd[np.random.choice(len(pcd), args.num_points)]
    total_edges = args.num_points * args.num_rotations * args.k

    pcd = torch.from_numpy(pcd)

    rots = Rotation.random(args.num_rotations).as_matrix()
    rots = torch.from_numpy(rots)

    if args.dtype == "float":
        pcd = pcd.float()
        rots = rots.float()
    else:
        pcd = pcd.double()
    rot_pcd = torch.einsum("nj, bij -> bni", pcd, rots)

    pcd = pcd.unsqueeze_(0).cuda()
    rot_pcd = rot_pcd.cuda()

    ref_knn_idx = knn(pcd, args.k)
    rot_knn_idx = knn(rot_pcd, args.k)
    wrong = torch.ne(ref_knn_idx.repeat(args.num_rotations, 1, 1), rot_knn_idx)

    print(f"[ScanNetV2] Err.: {wrong.sum()} / {total_edges}")


def main(args):
    data, _ = load_data("test", [args.label_in_use], "data/shapenet")

    total_edges = args.num_points * args.k
    errors = []
    for pcd in tqdm(data):
        pcd = pcd[:args.num_points]
        pcd = torch.from_numpy(pcd)

        rots = Rotation.random(args.num_rotations).as_matrix()
        rots = torch.from_numpy(rots)

        if args.dtype == "float":
            pcd = pcd.float()
            rots = rots.float()
        rot_pcd = torch.einsum("nj, bij -> bni", pcd, rots)

        pcd = pcd.unsqueeze_(0).cuda()
        rot_pcd = rot_pcd.cuda()

        ref_knn_idx = knn(pcd, args.k)
        rot_knn_idx = knn(rot_pcd, args.k)
        wrong = torch.ne(ref_knn_idx.repeat(args.num_rotations, 1, 1), rot_knn_idx)
        errors.append(wrong.sum() / args.num_rotations)
    
    avg_error = mean(errors)
    print(f"[Class {args.label_in_use}] [Min, Avg., Max] Err.: [{min(errors)}, {avg_error}, {max(errors)}] / {total_edges}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Knn Index Error Evaluation")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--num_rotations", type=int, default=10)
    parser.add_argument("--label_in_use", type=int, default=0)
    parser.add_argument("--dtype", type=str, choices=["float", "double"], required=True)
    args = parser.parse_args()

    pl.seed_everything(0)
    main(args)
    # main_scannet(args)