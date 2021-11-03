import os
import sys
from optparse import OptionParser
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

import math
import json
import numpy as np
from pathlib import Path

from region_dataset import Region
from transformations import PointcloudToTensor, PointcloudScale, PointcloudJitter, PointcloudTranslate, \
    PointcloudRotatePerturbation
import utils
from projection_models import DINOHead
from capsnet_models import PointCapsNet


def train_net(device, args):
    # create a list of transformations to be applied to the point cloud.
    # TODO: uncomment augmentation.
    transform = transforms.Compose([
        PointcloudToTensor(),
        PointcloudRotatePerturbation(angle_sigma=0.06, angle_clip=0.18),
        PointcloudScale(),
        PointcloudTranslate(),
        PointcloudJitter(std=0.01, clip=0.05)
    ])

    # create the training dataset
    dataset = Region(args.data_dir, args.scene_dir, num_local_crops=args.local_crops_number,
                     num_global_crops=args.global_crops_number, mode='train', transforms=transform)

    # create the dataloaders
    # TODO: increase num workers and shuffle
    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size_per_gpu,
                             num_workers=4,
                             pin_memory=True,
                             drop_last=True,
                             shuffle=True)

    # load the pre-trained capsulenet and set it to the right device
    capsule_net = PointCapsNet(1024, 16, 64, 64, args.num_points)
    capsule_net.load_state_dict(torch.load(args.best_capsule_net))
    capsule_net = capsule_net.to(device=device)
    capsule_net.train()

    # initialize an instance of the DINO projection heads and set them to the right device
    student = DINOHead(in_dim=64*64, out_dim=args.out_dim, nlayers=args.num_layers)
    teacher = DINOHead(in_dim=64*64, out_dim=args.out_dim, nlayers=args.num_layers)
    student = student.to(device=device)
    teacher = teacher.to(device=device)
    teacher_without_ddp = teacher

    # load the weights of the student to teacher
    teacher_without_ddp.load_state_dict(student.state_dict())

    # no backprop for the teacher.
    for p in teacher.parameters():
        p.requires_grad = False
    student.train()

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
    optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    # optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    # optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    fp16_scaler = None

    # schedule the learning rate according to the DINO paper.
    lr_schedule = utils.cosine_scheduler(
        args.lr * args.batch_size_per_gpu / 256.,  # linear scaling rule
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
        # train one epoch of DINO
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
                                      data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                                      epoch, fp16_scaler, capsule_net, device, args)

        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'dino_loss': dino_loss.state_dict(),
        }

        # see if you should save the checkpoint.
        if args.saveckp_freq and ((epoch + 1) % args.saveckp_freq) == 0:
            torch.save(save_dict, os.path.join(args.cp_dir, f'checkpoint{epoch:04}.pth'))
            torch.save(capsule_net.state_dict(), os.path.join(args.cp_dir, f'checkpoint{epoch:04}_caps_net.pth'))

        # save the logs for the epoch.
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        with (Path(args.cp_dir) / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, capsule_net, device, args):
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
        global_crops = global_crops.to(device=device, dtype=torch.float32)
        local_crops = local_crops.to(device=device, dtype=torch.float32)

        # reshape to absorb the batch size in the num crops
        global_crops = global_crops.reshape(-1, args.num_points, 3)
        local_crops = local_crops.reshape(-1, args.num_points, 3)

        # apply the pre-trained capsulenet model first
        global_crops = global_crops.transpose(2, 1)
        local_crops = local_crops.transpose(2, 1)
        global_caps = capsule_net(global_crops).reshape(-1, 64 * 64)
        local_caps = capsule_net(local_crops).reshape(-1, 64 * 64)

        # reshape back to include batch size
        global_caps = global_caps.reshape(args.global_crops_number, args.batch_size_per_gpu, -1)
        local_caps = local_caps.reshape(args.local_crops_number, args.batch_size_per_gpu, -1)

        # apply the DINO head.
        all_caps = torch.cat([global_caps, local_caps], dim=0)
        teacher_output = teacher(global_caps)
        student_output = student(all_caps)
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
        batch_center = batch_center / len(teacher_output)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def get_args():
    parser = OptionParser()
    parser.add_option('--epochs', dest='epochs', default=100, type='int', help='number of epochs')
    parser.add_option('--warmup_epochs', dest='warmup_epochs', default=10, type='int')
    parser.add_option('--warmup_teacher_temp_epochs', dest='warmup_teacher_temp_epochs', default=0, type='int')
    parser.add_option('--saveckp_freq', dest='saveckp_freq', default=20, type='int')
    parser.add_option('--data_dir', dest='data_dir',
                      default='../../data/matterport3d/mesh_regions')
    parser.add_option('--scene_dir', dest='scene_dir', default='../../data/matterport3d/scenes')
    parser.add_option('--cp_dir', dest='cp_dir',
                      default='../../results/matterport3d/LearningBased/3D_DINO_exact_regions')
    parser.add_option('--gpu', action='store_true', dest='gpu', default=True, help='use cuda')

    parser.add_option('--num_points', dest='num_points', default=4096, type='int')
    parser.add_option('--best_capsule_net', dest='best_capsule_net',
                      default='../../results/matterport3d/LearningBased/3D_DINO_exact_regions/best_capsule_net.pth')

    parser.add_option('--lr', dest='lr', default=0.0005, type='float')
    parser.add_option('--min_lr', dest='min_lr', default=1e-6, type='float')
    parser.add_option('--weight_decay', dest='weight_decay', default=0.04, type='float')
    parser.add_option('--weight_decay_end', dest='weight_decay_end', default=0.4, type='float')
    parser.add_option('--momentum_teacher', dest='momentum_teacher', default=0.996, type='float')
    parser.add_option('--warmup_teacher_temp', dest='warmup_teacher_temp', default=0.04, type='float')
    parser.add_option('--teacher_temp', dest='teacher_temp', default=0.04, type='float')
    parser.add_option('--clip_grad', dest='clip_grad', default=3.0, type='float')
    parser.add_option('--freeze_last_layer', dest='freeze_last_layer', default=1, type=int)

    parser.add_option('--out_dim', dest='out_dim', default=50000, type='int')
    parser.add_option('--num_layers', dest='num_layers', default=3, type='int')
    parser.add_option('--global_crops_number', dest='global_crops_number', default=2, type='int')
    parser.add_option('--local_crops_number', dest='local_crops_number', default=2, type='int')
    parser.add_option('--batch_size_per_gpu', dest='batch_size_per_gpu', default=2, type='int')

    (options, args) = parser.parse_args()
    return options


def main():
    # get the arguments
    args = get_args()

    # create a directory for checkpoints
    if not os.path.exists(args.cp_dir):
        os.makedirs(args.cp_dir)

    # Set the right device for all the models
    device = torch.device('cpu')
    if args.gpu:
        device = torch.device('cuda')

    # time the training
    train_net(device, args)


if __name__ == '__main__':
    main()
