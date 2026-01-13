import os
import sys
import argparse
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'src/model'))

import torch
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations

from src.data.shapenet_part import PartNormalFourClassesDataset
try:
    from ConDor.ConDor_pytorch.models.ConDor import ConDor
except:
    print("Include the ConDor codes.")
from canonical_capsules.models.acne_ae import AcneAe

from test_cls import (
    load_pl_model, load_caca_model, load_vnspd_model, load_condor_model,
    align, align_condor, align_caca
)
from src.utils import save_pcd


colors40 = [[88,170,108], [174,105,226], [78,194,83], [198,62,165], [133,188,52], [97,101,219], [190,177,52], [139,65,168], [75,202,137], [225,66,129],
        [68,135,42], [226,116,210], [146,186,98], [68,105,201], [219,148,53], [85,142,235], [212,85,42], [78,176,223], [221,63,77], [68,195,195],
        [175,58,119], [81,175,144], [184,70,74], [40,116,79], [184,134,219], [130,137,46], [110,89,164], [92,135,74], [220,140,190], [94,103,39],
        [144,154,219], [160,86,40], [67,107,165], [194,170,104], [162,95,150], [143,110,44], [146,72,105], [225,142,106], [162,83,86], [227,124,143]]


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def main(args):
    assert args.seg_ckpt is not None, "Segmenter checkpoint should be provided"

    '''DATA LOADING'''
    root = 'data/shapenet_part'
    TEST_DATASET = PartNormalFourClassesDataset(
        root=root, npoints=args.npoint, split='test', normal_channel=args.normal
    )
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    seg_classes = TEST_DATASET.seg_classes_mapped
    seg_label_to_cat = TEST_DATASET.seg_label_to_cat
    num_classes = TEST_DATASET.num_classes
    num_part = TEST_DATASET.num_part
    
    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)

    # Segmenter
    classifier = MODEL.get_model(args, num_classes, num_part, normal_channel=args.normal).cuda()
    seg_ckpt = torch.load(args.seg_ckpt)
    classifier.load_state_dict(seg_ckpt['model_state_dict'])
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

    with torch.inference_mode():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}

        if args.save_dir is not None:
            os.makedirs(args.save_dir, exist_ok=True)

        for points, label, target, data_indices in tqdm(testDataLoader, total=len(testDataLoader), smoothing=0.9):
            cur_batch_size, NUM_POINT, _ = points.size()
            
            trot = None
            if args.rot == 'z':
                trot = RotateAxisAngle(angle=torch.rand(points.shape[0])*360, axis="Z", degrees=True)
            elif args.rot == 'so3':
                trot = Rotate(R=random_rotations(points.shape[0]))
            if trot is not None:
                points = trot.transform_points(points)

            if aligner is not None:
                if isinstance(aligner, AcneAe):
                    points = align_caca(aligner, points)
                elif isinstance(aligner, ConDor):
                    points = align_condor(aligner, points)
                else:
                    points = align(aligner, points)
            
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            classifier = classifier.eval()
            seg_pred, _ = classifier(points, to_categorical(label, num_classes))
            cur_pred_val = seg_pred.cpu().data.numpy() # (B, N, 15)
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()
            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

            # cur_pred_val = (B, N)
            if args.save_dir is not None:
                points = points.transpose(2, 1).cpu().numpy() # (B, N, 3)
                result = cur_pred_val # (B, N)

                for pred_l, p, data_idx in zip(result, points, data_indices):
                    p_fname = os.path.join(args.save_dir, f"{data_idx:05d}.ply")
                    l_ = np.array([colors40[x] for x in pred_l]) / 255.
                    save_pcd(p, p_fname, l_)
            
            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float32))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

        print("================ [Results] ================")
        print(f"Categorical mIoU: {100 * test_metrics['class_avg_iou']:.2f}")
        print(f"Instance mIoU: {100 * test_metrics['inctance_avg_iou']:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', default='dgcnn_partseg', help='Model name [default: dgcnn_partseg]',
                        choices = ['dgcnn_partseg', 'vn_dgcnn_partseg', 'vn_pointnet_partseg'])
    parser.add_argument('--align_pl_ckpt', default=None, help='Align model checkpoint in PyTorch Lightning')
    parser.add_argument('--align_caca_ckpt', default=None, help='Align Canonical Capsules model checkpoint')
    parser.add_argument('--align_vnspd_dir', default=None, help='Align VN-SPD model checkpoint directory')
    parser.add_argument('--align_condor_ckpt', default=None, help='Align ConDor model checkpoint')
    parser.add_argument('--seg_ckpt', default=None, help='Part Segmenter checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 16]')
    parser.add_argument('--npoint', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--rot', type=str, default='aligned', help='Rotation augmentation to input data [default: aligned]',
                        choices=['aligned', 'z', 'so3'])
    parser.add_argument('--n_knn', default=20, type=int, help='Number of nearest neighbors to use, not applicable to PointNet [default: 20]')
    parser.add_argument('--pooling', type=str, default='mean', help='VNN only: pooling method [default: mean]',
                        choices=['mean', 'max'])
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()

    assert torch.cuda.is_available()

    pl.seed_everything(0, workers=True)
    main(args)
