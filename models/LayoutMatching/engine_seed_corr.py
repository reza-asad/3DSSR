# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
import datetime
import logging
import math
import time
import sys
from scipy.optimize import linear_sum_assignment

from torch.distributed.distributed_c10d import reduce
from utils.ap_calculator import APCalculator
from utils.misc import SmoothedValue
from utils.dist import (
    all_gather_dict,
    all_reduce_average,
    is_primary,
    reduce_dict,
    barrier,
)


def compute_learning_rate(args, curr_epoch_normalized):
    assert curr_epoch_normalized <= 1.0 and curr_epoch_normalized >= 0.0
    if (
        curr_epoch_normalized <= (args.warm_lr_epochs / args.max_epoch)
        and args.warm_lr_epochs > 0
    ):
        # Linear Warmup
        curr_lr = args.warm_lr + curr_epoch_normalized * args.max_epoch * (
            (args.base_lr - args.warm_lr) / args.warm_lr_epochs
        )
    else:
        # Cosine Learning Rate Schedule
        curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
            1 + math.cos(math.pi * curr_epoch_normalized)
        )
    return curr_lr


def adjust_learning_rate(args, optimizer, curr_epoch):
    curr_lr = compute_learning_rate(args, curr_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = curr_lr
    return curr_lr


def find_correspondence_accuracy(batch_logits):
    batch_size, npos_pairs, _ = batch_logits.shape
    cost_matrix = 1 / (batch_logits.detach().cpu().numpy() + 1e-8)
    num_correct, num_total = 0, 0
    for i in range(batch_size):
        row_ind, col_ind = linear_sum_assignment(cost_matrix[i, :, :])
        num_correct += (row_ind == col_ind).sum()
        num_total += npos_pairs

    return num_correct, num_total


def train_one_epoch(
    args,
    curr_epoch,
    model,
    optimizer,
    criterion,
    dataset_loader,
    logger,
):

    curr_iter = curr_epoch * len(dataset_loader)
    max_iters = args.max_epoch * len(dataset_loader)
    net_device = next(model.parameters()).device

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)

    model.train()
    barrier()

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
        for key in batch_data_label:
            if key != 'scan_name':
                batch_data_label[key] = batch_data_label[key].to(net_device)

        # Forward pass
        optimizer.zero_grad()

        # TODO: encode the point cloud with mask first.
        masked_inputs = {"point_clouds": batch_data_label["point_clouds_with_mask"]}
        enc_xyz, enc_features, enc_inds_q = model(masked_inputs)

        # TODO: take furthest points sampled from the subscene as well as their contextual features.
        q_seed_points_info = model(inputs=None, encode=False, masked_pc=masked_inputs['point_clouds'], enc_xyz=enc_xyz,
                                   enc_features=enc_features, enc_inds=enc_inds_q,
                                   instance_labels=batch_data_label["instance_labels"])
        # batch_size x num_points x 3 and batch_size x num_points x 256

        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }
        # TODO: find the seed points, their features and indices on the target scene.
        enc_xyz, enc_features, enc_inds_t = model(inputs)

        # during training the target seed point sampling is done on the subscene.
        t_seed_points_info = model(inputs=None, encode=False, masked_pc=masked_inputs['point_clouds'], enc_xyz=enc_xyz,
                                   enc_features=enc_features, enc_inds=enc_inds_t,
                                   instance_labels=batch_data_label["instance_labels"])

        # compute loss.
        loss, _ = criterion(q_seed_points_info, t_seed_points_info, evaluation=False)
        loss_reduced = all_reduce_average(loss)

        if not math.isfinite(loss_reduced.item()):
            logging.info(f"Loss in not finite. Training will be stopped.")
            sys.exit(1)

        loss.backward()
        if args.clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()

        time_delta.update(time.time() - curr_time)
        loss_avg.update(loss_reduced.item())

        # logging
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            eta_seconds = (max_iters - curr_iter) * time_delta.avg
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            print(
                f"Epoch [{curr_epoch}/{args.max_epoch}]; Iter [{curr_iter}/{max_iters}]; Loss {loss_avg.avg:0.2f}; LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; ETA {eta_str}; Mem {mem_mb:0.2f}MB"
            )
            logger.log_scalars({'loss': loss}, curr_iter, prefix="Train_details/")

            train_dict = {}
            train_dict["lr"] = curr_lr
            train_dict["memory"] = mem_mb
            train_dict["loss"] = loss_avg.avg
            train_dict["batch_time"] = time_delta.avg
            logger.log_scalars(train_dict, curr_iter, prefix="Train/")

        curr_iter += 1
        barrier()


@torch.no_grad()
def evaluate(
    args,
    curr_epoch,
    model,
    criterion,
    dataset_loader,
    logger,
    curr_train_iter,
):

    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    model.eval()
    barrier()
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""

    num_correct_epoch, num_total_epoch = 0, 0
    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        for key in batch_data_label:
            if key != 'scan_name':
                batch_data_label[key] = batch_data_label[key].to(net_device)

        # TODO: encode the point cloud with mask first.
        masked_inputs = {"point_clouds": batch_data_label["point_clouds_with_mask"]}
        enc_xyz, enc_features, enc_inds_q = model(masked_inputs)

        # TODO: take furthest points sampled from the subscene as well as their contextual features.
        q_seed_points_info = model(inputs=None, encode=False, masked_pc=masked_inputs['point_clouds'], enc_xyz=enc_xyz,
                                   enc_features=enc_features, enc_inds=enc_inds_q,
                                   instance_labels=batch_data_label["instance_labels"])
        # batch_size x num_points x 3 and batch_size x num_points x 256

        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }
        # TODO: find the seed points, their features and indices on the target scene.
        enc_xyz, enc_features, enc_inds_t = model(inputs)

        # during training the target seed point sampling is done on the subscene.
        t_seed_points_info = model(inputs=None, encode=False, masked_pc=masked_inputs['point_clouds'], enc_xyz=enc_xyz,
                                   enc_features=enc_features, enc_inds=enc_inds_t,
                                   instance_labels=batch_data_label["instance_labels"])

        # compute loss.
        loss, batch_logits = criterion(q_seed_points_info, t_seed_points_info, evaluation=True)
        loss_reduced = all_reduce_average(loss)
        loss_avg.update(loss_reduced.item())
        loss_str = f"Loss {loss_avg.avg:0.2f};"

        # apply the hungarian algorithm on the logits and compute correspondence accuracy.
        num_correct, num_total = find_correspondence_accuracy(batch_logits)
        num_correct_epoch += num_correct
        num_total_epoch += num_total

        time_delta.update(time.time() - curr_time)
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; {loss_str} Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )

            test_dict = {}
            test_dict["memory"] = mem_mb
            test_dict["batch_time"] = time_delta.avg
            test_dict["loss"] = loss_avg.avg
        curr_iter += 1
        barrier()

    if is_primary():
        logger.log_scalars(test_dict, curr_train_iter, prefix="Test/")

    accuracy_dict = {'accuracy': num_correct_epoch / float(num_total_epoch) * 100}

    return accuracy_dict, loss_avg.global_avg
