import os
import sys
import datetime
import logging
import argparse
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'src/model'))

import wandb
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path
from tqdm import tqdm

import provider
try:
    from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations
except:
    print("pytorch3d is not installed: pip install pytorch3d")
from src.data.shapenet_part import PartNormalFourClassesDataset
try:
    from ConDor.ConDor_pytorch.models.ConDor import ConDor
except:
    print("Include the ConDor codes.")
from canonical_capsules.models.acne_ae import AcneAe

from test_cls import (
    load_pl_model, load_caca_model, load_vnspd_model, load_condor_model,
    align, align_caca, align_condor
)


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', default='dgcnn_partseg', help='Model name [default: dgcnn_partseg]',
                        choices = ['dgcnn_partseg', 'vn_dgcnn_partseg', 'vn_pointnet_partseg'])
    parser.add_argument('--align_pl_ckpt', default=None, help='Align model checkpoint in PyTorch Lightning')
    parser.add_argument('--align_caca_ckpt', default=None, help='Align Canonical Capsules model checkpoint')
    parser.add_argument('--align_vnspd_dir', default=None, help='Align VN-SPD model checkpoint directory')
    parser.add_argument('--align_condor_ckpt', default=None, help='Align ConDor model checkpoint')
    parser.add_argument('--wandb_name', type=str, default='partseg')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=200, type=int, help='Epoch to run [default: 250]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate (for SGD it is multiplied by 100) [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Weight decay [default: 1e-4]')
    parser.add_argument('--step_size', type=int, default=20, help='Decay step for lr decay [default: every 20 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Adam or SGD [default: SGD]')
    parser.add_argument('--log_dir', type=str, default='vn_dgcnn/aligned', help='Experiment root [default: vn_dgcnn/aligned]')
    parser.add_argument('--npoint', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--rot', type=str, default='aligned', help='Rotation augmentation to input data [default: aligned]',
                        choices=['aligned', 'z', 'so3'])
    parser.add_argument('--pooling', type=str, default='mean', help='VNN only: pooling method [default: mean]',
                        choices=['mean', 'max'])
    parser.add_argument('--n_knn', default=20, type=int, help='Number of nearest neighbors to use, not applicable to PointNet [default: 20]')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./downstream_logs/')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = experiment_dir.joinpath('partseg')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project="char_rot_downstream",
        entity="postech_cvlab",
        name=args.wandb_name,
        dir=experiment_dir
    )

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/shapenet_part'

    TRAIN_DATASET = PartNormalFourClassesDataset(root = root, npoints=args.npoint, split='trainval', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size,shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    TEST_DATASET = PartNormalFourClassesDataset(root = root, npoints=args.npoint, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    
    seg_classes = TRAIN_DATASET.seg_classes_mapped
    seg_label_to_cat = TRAIN_DATASET.seg_label_to_cat
    num_classes = TRAIN_DATASET.num_classes
    num_part = TRAIN_DATASET.num_part
    
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" %  len(TEST_DATASET))
    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)

    classifier = MODEL.get_model(args, num_classes, num_part, normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

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

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=args.learning_rate*100,
            momentum=0.9,
            weight_decay=args.decay_rate
        )

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        mean_correct = []
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x,momentum))

        '''learning one epoch'''
        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            points, label, target, _ = data
            
            trot = None
            if args.rot == 'z':
                trot = RotateAxisAngle(angle=torch.rand(points.shape[0])*360, axis="Z", degrees=True)
            elif args.rot == 'so3':
                trot = Rotate(R=random_rotations(points.shape[0]))
            if trot is not None:
                points = trot.transform_points(points)
            
            points = points.data.numpy()

            if aligner is not None:
                if isinstance(aligner, AcneAe):
                    points = align_caca(aligner, points)
                elif isinstance(aligner, ConDor):
                    points = align_condor(aligner, points)
                else:
                    points = align(aligner, points)

            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(),label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            optimizer.zero_grad()
            classifier = classifier.train()
            seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))
            loss = criterion(seg_pred, target, trans_feat)
            loss.backward()
            optimizer.step()
        train_instance_acc = np.mean(mean_correct)
        log_string('Train accuracy is: %.5f' % train_instance_acc)
        wandb.log({"train/accuracy": train_instance_acc}, step=epoch + 1)

        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}

            for points, label, target, _ in tqdm(testDataLoader, total=len(testDataLoader), smoothing=0.9):
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
                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()
                for i in range(cur_batch_size):
                    cat = seg_label_to_cat[target[i, 0]]
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
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
            for cat in sorted(shape_ious.keys()):
                log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)


        log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
                 epoch+1, test_metrics['accuracy'],test_metrics['class_avg_iou'],test_metrics['inctance_avg_iou']))
        test_log_dict = {}
        test_log_dict["test/accuracy"] = test_metrics['accuracy']
        test_log_dict["test/cat_miou"] = test_metrics['class_avg_iou']
        test_log_dict["test/ins_miou"] = test_metrics['inctance_avg_iou']
        for cat, miou in shape_ious.items():
            test_log_dict[f"test_classwise_miou/{cat}"] = miou

        if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s'% savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        log_string('Best accuracy is: %.5f'%best_acc)
        log_string('Best class avg mIOU is: %.5f'%best_class_avg_iou)
        log_string('Best inctance avg mIOU is: %.5f'%best_inctance_avg_iou)
        test_log_dict["test/best_accuracy"] = best_acc
        test_log_dict["test/best_cat_miou"] = best_class_avg_iou
        test_log_dict["test/best_ins_miou"] = best_inctance_avg_iou
        wandb.log(test_log_dict, step=epoch + 1)
        global_epoch+=1

if __name__ == '__main__':
    args = parse_args()
    pl.seed_everything(0) # to remove the randomness
    main(args)
