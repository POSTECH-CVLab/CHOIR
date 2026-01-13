import os
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations

from src.model import dgcnn_cls
from src.data.shapenet_part import PartNormalFourClassesDataset
from src.model.models import build_model
from src.utils import DotDict

from VNSPD.models.networks import VNTSimpleEncoder, SimpleRot
try:
    from ConDor.ConDor_pytorch.models.ConDor import ConDor
except:
    print("Include the ConDor codes.")
from canonical_capsules.models.acne_ae import AcneAe


def remove_prefix(k, prefix):
    return k[len(prefix) :] if k.startswith(prefix) else k


@dataclass
class ConfigVNSPD:
    n_knn = 40
    base_ch = 64
    nlatent = 1020
    pooling = "mean"
    which_norm_VNT = "norm"
    which_strict_rot = "svd"
    global_bias = True


@dataclass
class ConfigCaCa:
    feat_net = "AcneKpEncoder"
    ae_decoder = "KpDecoder"
    acne_dim = 128
    indim = 3
    acne_net_depth = 3
    acne_num_g = 10
    acne_bn_type = "bn"
    cn_type = "acn_b"
    aligner = "init"
    grid_dim = 10
    decoder_grid = "learnable"
    decoder_bottleneck_size = 1280
    ref_kp_type = "None"
    mode = "test"
    num_view = 2
    random_range = "uni-180-0.2"


class AlignerVNSPD(nn.Module):
    def __init__(self, cfg):
        super(AlignerVNSPD, self).__init__()

        self.encoder = VNTSimpleEncoder(cfg, feature_transform=True)
        self.rotation = SimpleRot(
            cfg.nlatent // 2 // 3, cfg.which_strict_rot
        )

    def forward(self, p):
        p = p.transpose(1, 2) # (B, 3, N)

        inv_z, eq_z, t_vec = self.encoder(p)
        rot = self.rotation(eq_z)

        return {"end_rots": rot}


def load_pl_model(args):
    ckpt = torch.load(args.align_pl_ckpt)
    ckpt = DotDict(ckpt)
    
    model = build_model(ckpt["hyper_parameters"])
    state_dict = {remove_prefix(k, "model."): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()

    return model


def load_caca_model(args):
    caca_cfg = ConfigCaCa()
    if hasattr(args, "num_point"):
        caca_cfg.num_pts = args.num_point
    elif hasattr(args, "npoint"):
        caca_cfg.num_pts = args.npoint
    else:
        caca_cfg.num_pts = args.num_points
    ckpt = torch.load(args.align_caca_ckpt)
    
    model = AcneAe(caca_cfg)
    model.load_state_dict(ckpt["model"])
    model = model.cuda()
    model.eval()

    return model


def load_vnspd_model(args):
    vnspd_cfg = ConfigVNSPD()
    model = AlignerVNSPD(vnspd_cfg)

    encoder_ckpt = torch.load(os.path.join(args.align_vnspd_dir, "latest_net_Encoder.pth"))
    rotation_ckpt = torch.load(os.path.join(args.align_vnspd_dir, "latest_net_Rotation.pth"))

    model.encoder.load_state_dict(encoder_ckpt)
    model.rotation.load_state_dict(rotation_ckpt)

    model = model.cuda()
    model.eval()

    return model


def load_condor_model(args):
    ckpt = torch.load(args.align_condor_ckpt)
    state_dict = {remove_prefix(k, "ConDor."): v for k, v in ckpt["state_dict"].items()}

    model = ConDor(num_frames=5)
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()

    return model


@torch.inference_mode()
def align(aligner, points):
    assert not aligner.training
    assert points.shape[2] == 3 # xyz
    is_numpy = isinstance(points, np.ndarray)

    if is_numpy:
        points = torch.Tensor(points).cuda()
    else:
        points = points.cuda()

    output = aligner(points)
    pred_rots = output["end_rots"]
    aligned_points = torch.bmm(points, pred_rots)

    if is_numpy:
        aligned_points = aligned_points.cpu().numpy()
    else:
        aligned_points = aligned_points.cpu()

    return aligned_points


@torch.inference_mode()
def align_condor(aligner, points):
    assert not aligner.training
    assert points.shape[2] == 3 # xyz
    is_numpy = isinstance(points, np.ndarray)

    if is_numpy:
        points = torch.Tensor(points).cuda()
    else:
        points = points.cuda()

    output = aligner(points)

    if is_numpy:
        aligned_points = output["x_canonical"].cpu().numpy()
    else:
        aligned_points = output["x_canonical"].cpu()

    return aligned_points


@torch.inference_mode()
def align_caca(aligner, points):
    assert not aligner.training
    assert points.shape[2] == 3 # xyz
    is_numpy = isinstance(points, np.ndarray)

    if is_numpy:
        points = torch.Tensor(points).cuda()
    else:
        points = points.cuda()

    output = aligner.aligner.forward_align(points)

    if is_numpy:
        aligned_points = output["x_canonical"].cpu().numpy()
    else:
        aligned_points = output["x_canonical"].cpu()

    return aligned_points


@torch.inference_mode()
def test(args, classifier, aligner, loader, num_class=4):
    assert not classifier.training
    assert aligner is None or not aligner.training

    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    for data in tqdm(loader, total=len(loader)):
        points, target, _, _ = data
        
        trot = None
        if args.rot == 'z':
            trot = RotateAxisAngle(angle=torch.rand(points.shape[0])*360, axis="Z", degrees=True)
        elif args.rot == 'so3':
            trot = Rotate(R=random_rotations(points.shape[0]))
        if trot is not None:
            points = trot.transform_points(points)

        if aligner is not None:
            if isinstance(aligner, ConDor):
                points = align_condor(aligner, points)
            elif isinstance(aligner, AcneAe):
                points = align_caca(aligner, points)
            else:
                points = align(aligner, points)
        
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))

    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc


def main(args):
    assert args.cls_ckpt is not None, "Classifier checkpoint should be provided"

    '''DATA LOADING'''
    DATA_PATH = 'data/shapenet_part'
    TRAIN_DATASET = PartNormalFourClassesDataset(
        root=DATA_PATH, npoints=args.num_point, split='trainval', normal_channel=args.normal
    )
    TEST_DATASET = PartNormalFourClassesDataset(
        root=DATA_PATH, npoints=args.num_point, split='test', normal_channel=args.normal
    )
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    '''MODEL LOADING'''
    num_class = TRAIN_DATASET.num_classes
    MODEL = dgcnn_cls

    # Classifier
    classifier = MODEL.get_model(args, num_class, normal_channel=args.normal).cuda()
    cls_ckpt = torch.load(args.cls_ckpt)
    classifier.load_state_dict(cls_ckpt['model_state_dict'])
    classifier.eval()

    # Aligner
    if args.align_pl_ckpt is not None:
        aligner = load_pl_model(args)
    elif args.align_caca_ckpt is not None:
        aligner = load_caca_model(args)
    elif args.align_vnspd_dir is not None:
        aligner = load_vnspd_model(args)
    elif args.align_condor_ckpt is not None:
        aligner = load_condor_model(args)
    else:
        aligner = None

    instance_acc, class_acc = test(args, classifier, aligner, testDataLoader, 4)

    print("================ [Results] ================")
    print(f"Overall accuracy: {100 * instance_acc:.2f}")
    print(f"Mean accuracy: {100 * class_acc:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ShapeNet 4 classes classification')
    parser.add_argument('--align_pl_ckpt', default=None, help='Align model checkpoint in PyTorch Lightning')
    parser.add_argument('--align_caca_ckpt', default=None, help='Align Canonical Capsules model checkpoint')
    parser.add_argument('--align_vnspd_dir', default=None, help='Align VN-SPD model checkpoint directory')
    parser.add_argument('--align_condor_ckpt', default=None, help='Align ConDor model checkpoint')
    parser.add_argument('--cls_ckpt', default=None, help='Classifier checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size in training [default: 32]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--rot', type=str, default='aligned', help='Rotation augmentation to input data [default: aligned]',
                        choices=['aligned', 'z', 'so3'])
    parser.add_argument('--n_knn', default=20, type=int, help='Number of nearest neighbors to use, not applicable to PointNet [default: 20]')
    args = parser.parse_args()

    assert torch.cuda.is_available()

    pl.seed_everything(0)
    main(args)