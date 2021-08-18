import os
import sys
import argparse
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

import math
import json
import numpy as np
from pathlib import Path

from region_dataset import Region
from transformations import PointcloudScale, PointcloudJitter, PointcloudTranslate, \
    PointcloudRotatePerturbation
import utils
from transformer_models import Backbone
from projection_models import DINOHead


def train_net(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    # create a list of transformations to be applied to the point cloud.
    # TODO: also run with augmentation.
    # transform = transforms.Compose([
    #     PointcloudRotatePerturbation(angle_sigma=0.06, angle_clip=0.18),
    #     PointcloudScale(),
    #     PointcloudTranslate(),
    #     PointcloudJitter(std=0.01, clip=0.05)
    # ])

    # create the training dataset
    dataset = Region(args.mesh_dir, args.pc_dir, args.scene_dir, args.metadata_path, args.accepted_cats_path,
                     num_local_crops=args.local_crops_number, num_global_crops=args.global_crops_number, mode='train')
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)

    # create the dataloaders
    # TODO: change number of workers and shuffle.
    data_loader = DataLoader(dataset,
                             sampler=sampler,
                             batch_size=args.batch_size_per_gpu,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             drop_last=True)

    # load a transformer encoder and create teacher and student
    student = Backbone(args)
    teacher = Backbone(args)
    student = utils.DINO(student, DINOHead(in_dim=args.transformer_dim, out_dim=args.out_dim,
                                           use_bn=args.use_bn_in_head, norm_last_layer=args.norm_last_layer))
    teacher = utils.DINO(teacher, DINOHead(in_dim=args.transformer_dim, out_dim=args.out_dim,
                                           use_bn=args.use_bn_in_head))

    # move networks to GPU
    student, teacher = student.cuda(), teacher.cuda()

    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])

    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())

    # no backprop for the teacher.
    student.train()
    for p in teacher.parameters():
        p.requires_grad = False

    # prepare the DINOLoss
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # define the optimizer and loss criteria
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
        # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # schedule the learning rate according to the DINO paper.
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # resume training if necessary
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.cp_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # train one epoch of DINO
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
                                      data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                                      epoch, fp16_scaler, args)

        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'dino_loss': dino_loss.state_dict(),
        }

        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.cp_dir, 'checkpoint.pth'))

        # see if you should save the checkpoint.
        if args.saveckp_freq and ((epoch + 1) % args.saveckp_freq) == 0:
            utils.save_on_master(save_dict, os.path.join(args.cp_dir, f'checkpoint{epoch+1:04}.pth'))

        # save the logs for the epoch.
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.cp_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, data in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # load data
        global_crops = data['global_crops']
        local_crops = data['local_crops']

        # move data to the right device
        global_crops = global_crops.to(dtype=torch.float32).cuda()
        local_crops = local_crops.to(dtype=torch.float32).cuda()

        # apply 3D DINO model.
        all_crops = torch.cat([global_crops, local_crops], dim=1)
        teacher_output = teacher(global_crops)
        student_output = student(all_crops)
        loss = dino_loss(student_output, teacher_output, epoch)

        # exit if the loss is infinity.
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    metric_logger.synchronize_between_processes()
    print("Averaged Stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def get_args():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # path parameters
    parser.add_argument('--accepted_cats_path', default='../../data/matterport3d/accepted_cats.json')
    parser.add_argument('--mesh_dir', default='../../data/matterport3d/mesh_regions')
    parser.add_argument('--pc_dir', default='../../data/matterport3d/point_cloud_regions')
    parser.add_argument('--scene_dir', default='../../data/matterport3d/scenes')
    parser.add_argument('--metadata_path', default='../../data/matterport3d/metadata.csv')
    parser.add_argument('--cp_dir', default='../../results/matterport3d/LearningBased/3D_DINO_exact_regions_transformer')

    # transformer parameters
    parser.add_argument('--num_point', default=4096, type=int)
    parser.add_argument('--nblocks', default=4, type=int)
    parser.add_argument('--nneighbor', default=16, type=int)
    parser.add_argument('--input_dim', default=3, type=int)
    parser.add_argument('--transformer_dim', default=512, type=int)
    parser.add_argument('--out_dim', dest='out_dim', default=2000, type=int)
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag)
    parser.add_argument('--momentum_teacher', default=0.996, type=float)
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag)

    # temerature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.02, type=float)
    parser.add_argument('--teacher_temp', default=0.02, type=float)
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int)

    # optimization parameters
    # TODO: also try this with True.
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False)
    parser.add_argument('--weight_decay', default=0.04, type=float)
    parser.add_argument('--weight_decay_end', default=0.4, type=float)
    parser.add_argument('--clip_grad', dest='clip_grad', default=3.0, type=float)
    parser.add_argument('--batch_size_per_gpu', default=1, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--freeze_last_layer', default=1, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--warmup_epochs', default=10, type=int)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adamw', 'sgd', 'lars'])
    parser.add_argument('--drop_path_rate', default=0.1, type=float, help="stochastic depth rate")

    # crop parameters
    # TODO: could also add the crop scales here
    parser.add_argument('--global_crops_number', default=2, type=int)
    parser.add_argument('--local_crops_number', default=2, type=int)

    # remaining params
    parser.add_argument('--saveckp_freq', default=20, type=int)
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument("--dist_url", default='env://', type=str)
    parser.add_argument("--local_rank", default=0, type=int)

    return parser


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            vars(args)[k] = os.path.join(base_dir, v)


def main():
    # get the arguments
    parser = argparse.ArgumentParser('DINO', parents=[get_args()])
    args = parser.parse_args()
    adjust_paths(args, exceptions=['dist_url'])

    # create a directory for checkpoints
    if not os.path.exists(args.cp_dir):
        os.makedirs(args.cp_dir)

    # time the training
    train_net(args)


if __name__ == '__main__':
    main()
