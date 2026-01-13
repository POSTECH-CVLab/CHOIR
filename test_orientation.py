import os
import argparse

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.data.shapenet import ShapeNetConsistencyTest, ShapeNetStabilityTest
from src.data.collate_fns import collate_shapenet_stability
from src.rotation import cal_angular_std_from_rotations
from src.model.vn_layers import knn

from test_cls import (
    load_pl_model, load_caca_model, load_vnspd_model, load_condor_model
)
from src.utils import save_pcd


def remove_prefix(k, prefix):
    return k[len(prefix) :] if k.startswith(prefix) else k


def mean(values: list):
    return sum(values) / len(values)


@torch.inference_mode()
def eval_consistency(model, data_loader, save_dir):
    model.eval()

    outputs = dict()
    sample_idx = 0
    for batch in tqdm(data_loader):
        pcd = batch["pcd"].cuda()
        label = batch["label"].tolist()
        out_dict = model(pcd)

        if save_dir is None:
            for l, rot in zip(label, out_dict["end_rots"]):
                if l in outputs.keys():
                    outputs[l].append(rot)
                else:
                    outputs[l] = [rot]
        else:
            for l, rot, p in zip(label, out_dict["end_rots"], pcd):
                save_task_dir = os.path.join(save_dir, "consistency")
                os.makedirs(save_task_dir, exist_ok=True)

                p_fname = os.path.join(save_task_dir, f"{sample_idx:05d}.ply")
                can_p_fname = os.path.join(save_task_dir, f"{sample_idx:05d}_can.ply")
                can_p = p @ rot
                save_pcd(p, p_fname)
                save_pcd(can_p, can_p_fname)

                if l in outputs.keys():
                    outputs[l].append(rot)
                else:
                    outputs[l] = [rot]

                sample_idx += 1

    classwise_std = dict()
    for l, rot in outputs.items():
        rot = torch.stack(rot)
        std = cal_angular_std_from_rotations(rot).item()
        classwise_std[l] = std

    mean_std = mean(list(classwise_std.values()))

    return mean_std, classwise_std


@torch.inference_mode()
def eval_consistency_caca(model, data_loader, save_dir):
    model.eval()

    outputs = dict()
    sample_idx = 0
    for batch in tqdm(data_loader):
        pcd = batch["pcd"].cuda()
        label = batch["label"].tolist()
        out_dict = model.aligner.forward_align(pcd)

        if save_dir is None:
            for l, rot in zip(label, out_dict["end_rots"]):
                if l in outputs.keys():
                    outputs[l].append(rot)
                else:
                    outputs[l] = [rot]
        else:
            for l, rot, p in zip(label, out_dict["end_rots"], pcd):
                save_task_dir = os.path.join(save_dir, "consistency")
                os.makedirs(save_task_dir, exist_ok=True)

                p_fname = os.path.join(save_task_dir, f"{sample_idx:05d}.ply")
                can_p_fname = os.path.join(save_task_dir, f"{sample_idx:05d}_can.ply")
                can_p = p @ rot
                save_pcd(p, p_fname)
                save_pcd(can_p, can_p_fname)

                if l in outputs.keys():
                    outputs[l].append(rot)
                else:
                    outputs[l] = [rot]

                sample_idx += 1

    classwise_std = dict()
    for l, rot in outputs.items():
        rot = torch.stack(rot)
        std = cal_angular_std_from_rotations(rot).item()
        classwise_std[l] = std

    mean_std = mean(list(classwise_std.values()))

    return mean_std, classwise_std


@torch.inference_mode()
def eval_stability(args, model, data_loader, save_dir, force_stb=False):
    model.eval()

    outputs = dict()
    sample_idx = 0
    for batch in tqdm(data_loader):
        pcd = batch["pcd"].cuda()

        if force_stb:
            assert len(pcd) % 10 == 0
            ref_indices = torch.arange(0, len(pcd), 10, dtype=torch.long, device=pcd.device)
            ref_pcd = pcd[ref_indices]
            ref_knn_indices = knn(ref_pcd.transpose(1, 2), model.backbone.k)
            knn_indices = ref_knn_indices.unsqueeze(1).repeat(1, 10, 1, 1).view(len(pcd), -1, model.backbone.k)
        else:
            knn_indices = None

        label = batch["label"].tolist()

        if args.align_pl_ckpt is not None:
            out_dict = model(pcd, knn_idx=knn_indices)
        else:
            out_dict = model(pcd)

        batched_rots_can = torch.bmm(batch["rots"].transpose(1, 2).cuda(), out_dict["end_rots"])
        pcd_can = torch.bmm(pcd, out_dict["end_rots"]) # (10 * B, N, 3)
        list_pcd = pcd.chunk(batch["batch_size"], dim=0) # [(10, N, 3)] * B
        list_pcd_can = pcd_can.chunk(batch["batch_size"], dim=0) # [(10, N, 3)] * B
        list_rots_can = batched_rots_can.chunk(batch["batch_size"], dim=0)

        if save_dir is None:
            for l, rots_can in zip(label, list_rots_can):
                std = cal_angular_std_from_rotations(rots_can).item()
                
                if l in outputs.keys():
                    outputs[l].append(std)
                else:
                    outputs[l] = [std]
        else:
            for l, pcd, pcd_can, rots_can in zip(label, list_pcd, list_pcd_can, list_rots_can):
                save_task_dir = os.path.join(save_dir, "stability")
                os.makedirs(save_task_dir, exist_ok=True)

                for i, (p, p_can) in enumerate(zip(pcd, pcd_can)):
                    p_fname = os.path.join(save_task_dir, f"{sample_idx:05d}_{i:02d}.ply")
                    can_p_fname = os.path.join(save_task_dir, f"{sample_idx:05d}_{i:02d}_can.ply")

                    save_pcd(p, p_fname)
                    save_pcd(p_can, can_p_fname)

                std = cal_angular_std_from_rotations(rots_can).item()
                
                if l in outputs.keys():
                    outputs[l].append(std)
                else:
                    outputs[l] = [std]
                
                sample_idx += 1

    classwise_std = dict()
    for k, v in outputs.items():
        classwise_std[k] = mean(v)

    mean_std = mean(list(classwise_std.values()))

    return mean_std, classwise_std


@torch.inference_mode()
def eval_stability_caca(args, model, data_loader, save_dir, force_stb=False):
    model.eval()

    outputs = dict()
    sample_idx = 0
    for batch in tqdm(data_loader):
        pcd = batch["pcd"].cuda()
        label = batch["label"].tolist()
        out_dict = model.aligner.forward_align(pcd)

        batched_rots_can = torch.bmm(batch["rots"].transpose(1, 2).cuda(), out_dict["end_rots"])
        pcd_can = torch.bmm(pcd, out_dict["end_rots"]) # (10 * B, N, 3)
        list_pcd = pcd.chunk(batch["batch_size"], dim=0) # [(10, N, 3)] * B
        list_pcd_can = pcd_can.chunk(batch["batch_size"], dim=0) # [(10, N, 3)] * B
        list_rots_can = batched_rots_can.chunk(batch["batch_size"], dim=0)

        if save_dir is None:
            for l, rots_can in zip(label, list_rots_can):
                std = cal_angular_std_from_rotations(rots_can).item()
                
                if l in outputs.keys():
                    outputs[l].append(std)
                else:
                    outputs[l] = [std]
        else:
            for l, p, p_can, rots_can in zip(label, list_pcd, list_pcd_can, list_rots_can):
                save_task_dir = os.path.join(save_dir, "stability")
                os.makedirs(save_task_dir, exist_ok=True)

                for i, (p, p_can) in enumerate(zip(pcd, pcd_can)):
                    p_fname = os.path.join(save_task_dir, f"{sample_idx:05d}_{i:02d}.ply")
                    can_p_fname = os.path.join(save_task_dir, f"{sample_idx:05d}_{i:02d}_can.ply")

                    save_pcd(p, p_fname)
                    save_pcd(p_can, can_p_fname)

                std = cal_angular_std_from_rotations(rots_can).item()
                
                if l in outputs.keys():
                    outputs[l].append(std)
                else:
                    outputs[l] = [std]
                
                sample_idx += 1

    classwise_std = dict()
    for k, v in outputs.items():
        classwise_std[k] = mean(v)

    mean_std = mean(list(classwise_std.values()))

    return mean_std, classwise_std


def main(args):
    # Load the model
    eval_cns_fn = eval_consistency
    eval_stb_fn = eval_stability

    if args.align_pl_ckpt is not None:
        model = load_pl_model(args)
    elif args.align_caca_ckpt is not None:
        model = load_caca_model(args)
        eval_cns_fn = eval_consistency_caca
        eval_stb_fn = eval_stability_caca
    elif args.align_vnspd_dir is not None:
        model = load_vnspd_model(args)
    elif args.align_condor_ckpt is not None:
        model = load_condor_model(args)
    else:
        raise ValueError("No checkpoint file provided")

    # Dataset
    cns_dset = ShapeNetConsistencyTest(
        args.label_in_use, args.num_points, args.resample, args.shapenet_path
    )
    cns_dloader = DataLoader(
        cns_dset, batch_size=args.batch_size, drop_last=False, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    stb_dset = ShapeNetStabilityTest(
        args.label_in_use, args.num_points, args.resample, args.shapenet_path, args.add_gaussian, args.knn_removal
    )
    stb_dloader = DataLoader(
        stb_dset, batch_size=args.batch_size // 8, drop_last=False, shuffle=False, collate_fn=collate_shapenet_stability,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    # Evaluation
    model = model.cuda()
    cns, cns_classwise = eval_cns_fn(model, cns_dloader, args.save_dir)
    stb, stb_classwise = eval_stb_fn(args, model, stb_dloader, args.save_dir, args.force_stb)
    avg = (cns + stb) / 2
    avg_classwise = dict()
    for k in cns_classwise.keys():
        avg_classwise[k] = (cns_classwise[k] + stb_classwise[k]) / 2

    # Logging
    print("================ [Results] ================")
    print(f"Consistency: {cns:.3f}")
    for k, v in cns_classwise.items():
        print(f">>> {cns_dset.label2name[k]}: {v:.3f}")
    print(f"\nStability: {stb:.3f}")
    for k, v in stb_classwise.items():
        print(f">>> {cns_dset.label2name[k]}: {v:.3f}")
    print(f"\nAverage: {avg:.3f}")
    for k, v in avg_classwise.items():
        print(f">>> {cns_dset.label2name[k]}: {v:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Self-Supervised Learning of Canonical Orientation")
    parser.add_argument('--align_pl_ckpt', default=None, help='Align model checkpoint in PyTorch Lightning')
    parser.add_argument('--align_caca_ckpt', default=None, help='Align Canonical Capsules model checkpoint')
    parser.add_argument('--align_vnspd_dir', default=None, help='Align VN-SPD model checkpoint directory')
    parser.add_argument('--align_condor_ckpt', default=None, help='Align ConDor model checkpoint')
    parser.add_argument("--label_in_use", type=str, default="0", help="labels to test [plane: 0, car: 3, chair: 4, table: 10]")
    parser.add_argument("--num_points", type=str, default=1024, help="# points to sample")
    parser.add_argument("--resample", action="store_true", help="test-time resampling")
    parser.add_argument("--add_gaussian", action="store_true", help="test-time noise")
    parser.add_argument("--knn_removal", action="store_true", help="test-time knn removal")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of the dataloader")
    parser.add_argument("--shapenet_path", type=str, default="data/shapenet")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--force_stb", action="store_true")
    args = parser.parse_args()

    assert torch.cuda.is_available()

    pl.seed_everything(0)
    main(args)