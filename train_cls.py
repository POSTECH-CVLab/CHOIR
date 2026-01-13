import os
import datetime
import logging
import argparse

import wandb
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path
from tqdm import tqdm

import provider
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations
from src.data.shapenet_part import PartNormalFourClassesDataset
from src.model import dgcnn_cls
from ConDor.ConDor_pytorch.models.ConDor import ConDor
from canonical_capsules.models.acne_ae import AcneAe

from test_cls import (
    load_pl_model, load_caca_model, load_vnspd_model, load_condor_model,
    align, align_condor, align_caca, test
)


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--model', default='dgcnn_cls', help='Model name [default: dgcnn_cls]',
                        choices = ['dgcnn_cls'])
    parser.add_argument('--align_pl_ckpt', default=None, help='Align model checkpoint in PyTorch Lightning')
    parser.add_argument('--align_caca_ckpt', default=None, help='Align Canonical Capsules model checkpoint')
    parser.add_argument('--align_vnspd_dir', default=None, help='Align VN-SPD model checkpoint directory')
    parser.add_argument('--align_condor_ckpt', default=None, help='Align ConDor model checkpoint')
    parser.add_argument('--wandb_name', type=str, default='cls')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size in training [default: 32]')
    parser.add_argument('--epoch', default=200, type=int, help='Number of epoch in training [default: 250]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate (for SGD it is multiplied by 100) [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Decay rate [default: 1e-4]')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Pptimizer for training [default: SGD]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--log_dir', type=str, default='vn_dgcnn/aligned', help='Experiment root [default: vn_dgcnn/aligned]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores with voting [default: 3]')
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
    experiment_dir = experiment_dir.joinpath('cls')
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

    '''DATA LOADING'''
    log_string('Load dataset ...')
    DATA_PATH = 'data/shapenet_part'

    TRAIN_DATASET = PartNormalFourClassesDataset(root=DATA_PATH, npoints=args.num_point, split='trainval', normal_channel=args.normal)
    TEST_DATASET = PartNormalFourClassesDataset(root=DATA_PATH, npoints=args.num_point, split='test', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    num_class = TRAIN_DATASET.num_classes
    MODEL = dgcnn_cls

    classifier = MODEL.get_model(args, num_class, normal_channel=args.normal).cuda()
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

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    mean_correct = []

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target, _, _ = data
            
            trot = None
            if args.rot == 'z':
                trot = RotateAxisAngle(angle=torch.rand(points.shape[0])*360, axis="Z", degrees=True)
            elif args.rot == 'so3':
                trot = Rotate(R=random_rotations(points.shape[0]))
            if trot is not None:
                points = trot.transform_points(points)
            
            points = points.data.numpy()

            if aligner is not None:
                if isinstance(aligner, ConDor):
                    points = align_condor(aligner, points)
                elif isinstance(aligner, AcneAe):
                    points = align_caca(aligner, points)
                else:
                    points = align(aligner, points)

            points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)
            target = target[:, 0]

            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            classifier = classifier.train()
            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)
        wandb.log({"train/overall_accuracy": train_instance_acc}, step=epoch + 1)


        with torch.no_grad():
            instance_acc, class_acc = test(args, classifier.eval(), aligner, testDataLoader)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))
            test_log_dict = {
                "test/overall_accuracy": instance_acc,
                "test/mean_accuracy": class_acc,
                "test/best_overall_accuracy": best_instance_acc,
                "test/best_mean_accuracy": best_class_acc
            }
            wandb.log(test_log_dict, step=epoch + 1)

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    pl.seed_everything(0)
    main(args)
