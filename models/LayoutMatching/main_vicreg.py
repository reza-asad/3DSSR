import argparse
from argparse import Namespace
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.distributed as dist

from distributed import init_distributed_mode
from query_target_scene_dataset import Scene
from models.LearningBased.transformer_models import PointTransformerSeg
import models.LearningBased.utils as utils
from models.LearningBased.projection_models import DINOHead
from classifier_subscene import Backbone, Classifier
from scripts.helper import load_from_json


def get_arguments():
    parser = argparse.ArgumentParser(description="Self-supervised Layout Matching", add_help=False)

    # Data
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--accepted_cats_path', default='../../data/{}/accepted_cats.json')
    parser.add_argument('--metadata_path', dest='metadata_path', default='../../data/{}/metadata.csv')
    parser.add_argument('--pc_dir', default='../../data/{}/pc_regions')
    parser.add_argument('--scene_dir', default='../../results/{}/scenes')
    parser.add_argument('--cp_dir', default='../../results/{}/LayoutMatching/')
    parser.add_argument('--results_folder_name', default='exp')
    parser.add_argument('--results_folder_name_pret', default='exp_supervise_pret')
    parser.add_argument('--checkpoint', default='checkpoint.pth')
    parser.add_argument('--pre_training_checkpoint', default='CP_best.pth')
    parser.add_argument("--log_freq_time", type=int, default=60, help='Print logs to the stats.txt file')

    # Model
    parser.add_argument('--num_point', default=4096, type=int)
    parser.add_argument('--nblocks', default=2, type=int)
    parser.add_argument('--nneighbor', default=16, type=int)
    parser.add_argument('--input_dim', default=3, type=int)
    parser.add_argument('--transformer_dim', default=32, type=int)
    parser.add_argument('--out_dim', default=2000, type=int)
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag)
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag)
    parser.add_argument('--max_coord_box', default=3.65, type=float, help='3.65 for MP3D')
    parser.add_argument('--max_coord_scene', default=13.07, type=float, help='13.07 for MP3D')
    parser.add_argument('--crop_bounds', type=float, nargs='+', default=(0.9, 0.9))
    parser.add_argument('--max_query_size', default=5, type=int)
    parser.add_argument('--max_sample_size', default=5, type=int)
    parser.add_argument('--random_subscene', default=False, type=utils.bool_flag)
    parser.add_argument('--freeze_backbone', default=True, type=utils.bool_flag)

    # Optim
    parser.add_argument("--epochs", type=int, default=100, help='Number of epochs')
    parser.add_argument("--base_lr", type=float, default=0.2,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--wd", type=float, default=1e-6, help='Weight decay')

    # Loss
    parser.add_argument("--sim_coeff", type=float, default=25.0, help='Invariance regularization loss coefficient')
    parser.add_argument("--std_coeff", type=float, default=25.0, help='Variance regularization loss coefficient')
    parser.add_argument("--cov_coeff", type=float, default=1.0, help='Covariance regularization loss coefficient')

    # Running
    # TODO: chnage bakc to 6
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    # Distributed
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    return parser


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.base_lr
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def exclude_bias_and_norm(p):
    return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def load_model(args, layout_args):
    # build the shape backbone.
    shape_backbone = PointTransformerSeg(args)

    # build layout matching backbone.
    layout_backbone = PointTransformerSeg(layout_args)

    # combine the backbones and attach a projection module to that.
    backbone = Backbone(shape_backbone=shape_backbone, layout_backbone=layout_backbone, num_point=args.num_point,
                        num_objects=args.max_sample_size + args.max_query_size)
    projection = DINOHead(in_dim=args.transformer_dim, out_dim=args.num_class, use_bn=args.use_bn_in_head,
                          norm_last_layer=args.norm_last_layer)
    model = Classifier(backbone=backbone, head=projection)
    model = torch.nn.DataParallel(model).cuda()

    # load the pre-trained classification weights
    checkpoint_path = os.path.join(args.cp_dir, args.results_folder_name_pret, args.pre_training_checkpoint)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # replace the projection head from classifier with one that will learn layout matching.
    model.module.head = DINOHead(in_dim=args.transformer_dim, out_dim=args.out_dim, use_bn=args.use_bn_in_head,
                                 norm_last_layer=args.norm_last_layer)

    # freeze the backbone weights if necessary but let the projection module learn new weights.
    if args.freeze_backbone:
        model.module.backbone.requires_grad_(False)
    else:
        model.module.backbone.requires_grad_(True)
    model.module.head.requires_grad_(True)

    return model


def main(args):
    torch.backends.cudnn.benchmark = True
    adjust_paths(args, exceptions=['dist_url'])
    init_distributed_mode(args)
    print(args)
    gpu = torch.device(args.device)

    # prepare the accepted categories for training.
    accepted_cats = load_from_json(args.accepted_cats_path)
    accepted_cats = sorted(accepted_cats)
    args.cat_to_idx = {cat: i for i, cat in enumerate(accepted_cats)}
    args.num_class = len(args.cat_to_idx)

    if args.rank == 0:
        output_dir = os.path.join(args.cp_dir, args.results_folder_name)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except FileExistsError:
                pass
        stats_file = open(os.path.join(output_dir, "stats.txt"), "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)

    dataset = Scene(scene_dir=args.scene_dir, pc_dir=args.pc_dir, metadata_path=args.metadata_path,
                    accepted_cats_path=args.accepted_cats_path, max_coord_box=args.max_coord_box,
                    max_coord_scene=args.max_coord_scene, num_points=args.num_point, crop_bounds=args.crop_bounds,
                    max_query_size=args.max_query_size, max_sample_size=args.max_sample_size,
                    random_subscene=args.random_subscene)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
        collate_fn=collate_fn
    )

    layout_args = Namespace()
    exceptions = {'input_dim': 35, 'nblocks': 0}
    for k, v in vars(args).items():
        if k in exceptions:
            vars(layout_args)[k] = exceptions[k]
        else:
            vars(layout_args)[k] = v

    # load the backbone
    model = load_model(args, layout_args)
    model = model.cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    optimizer = LARS(
        model.parameters(),
        lr=0,
        weight_decay=args.wd,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )

    checkpoint_path = os.path.join(args.cp_dir, args.results_folder_name, args.checkpoint)
    if os.path.exists(checkpoint_path):
        if args.rank == 0:
            print("resuming from checkpoint")
        ckpt = torch.load(os.path.join(checkpoint_path), map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    start_time = last_logging = time.time()
    # scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, data in enumerate(loader, start=epoch * len(loader)):
            # load data
            pc_t = data['pc_t'].squeeze(dim=0)
            centroid_t = data['centroid_t'].squeeze(dim=0)
            match_label_t = data['match_label_t'].squeeze(dim=0)

            pc_q = data['pc_q'].squeeze(dim=0)
            centroid_q = data['centroid_q'].squeeze(dim=0)
            match_label_q = data['match_label_q'].squeeze(dim=0)

            # print(pc_t.shape, centroid_t.shape, match_label_t.shape)
            # print(pc_q.shape, centroid_q.shape, match_label_q.shape)

            # move data to the right device
            pc_t = pc_t.to(dtype=torch.float32).cuda()
            centroid_t = centroid_t.to(dtype=torch.float32).cuda()
            match_label_t = match_label_t.to(dtype=torch.bool).cuda()

            pc_q = pc_q.to(dtype=torch.float32).cuda()
            centroid_q = centroid_q.to(dtype=torch.float32).cuda()
            match_label_q = match_label_q.to(dtype=torch.bool).cuda()

            # apply the model on query subscene and target subscenes.
            pc_q_t = torch.cat([pc_q, pc_t], dim=0)
            centroid_q_t = torch.cat([centroid_q, centroid_t], dim=0)
            q_t = model(pc_q_t, centroid_q_t)
            q = q_t[:args.max_query_size, :]
            t = q_t[args.max_query_size:, :]

            # filter the output to matching objects.
            q = q[match_label_q, :]
            t = t[match_label_t, :]

            # representation loss.
            repr_loss = F.mse_loss(q, t)

            # std loss
            q = q - q.mean(dim=0)
            t = t - t.mean(dim=0)
            std_q = torch.sqrt(q.var(dim=0) + 0.0001)
            std_t = torch.sqrt(t.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_q)) / 2 + torch.mean(F.relu(1 - std_t)) / 2

            # cov loss.
            batch_size = len(q)
            cov_q = (q.T @ q) / (batch_size - 1)
            cov_t = (t.T @ t) / (batch_size - 1)
            cov_loss = off_diagonal(cov_q).pow_(2).sum().div(args.out_dim) + \
                       off_diagonal(cov_t).pow_(2).sum().div(args.out_dim)

            # combined loss.
            loss = (args.sim_coeff * repr_loss + args.std_coeff * std_loss + args.cov_coeff * cov_loss)

            lr = adjust_learning_rate(args, optimizer, loader, step)

            optimizer.zero_grad()
            loss.backward()

            # with torch.cuda.amp.autocast():
            #     loss = model.forward(pc_t, centroid_t, match_label_t, pc_q, centroid_q, match_label_q)
            #     # print(loss.item())
            #     # t=t
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            current_time = time.time()
            if args.rank == 0 and current_time - last_logging > args.log_freq_time:
                stats = dict(
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    time=int(current_time - start_time),
                    lr=lr,
                )
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
                last_logging = current_time
        if args.rank == 0:
            print('Finished epoch {}'.format(epoch))
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            torch.save(state, checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('VICReg training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
