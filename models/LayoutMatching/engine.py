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
import utils.pc_util as pc_util


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


def rotate_pc(pc, thetas):
    B, N, D = pc.shape
    pc_rot = torch.zeros_like(pc)
    rotation_matrix = torch.zeros((B, 3, 3), dtype=torch.float32, device=pc.device)
    for i in range(B):
        # build rotation.
        theta = thetas[i].item()
        rotation = np.asarray([[np.cos(theta), -np.sin(theta), 0],
                               [np.sin(theta), np.cos(theta), 0],
                               [0, 0, 1]])
        rotation_matrix[i, ...] = torch.from_numpy(rotation)

    # apply rotation.
    rotation_matrix = rotation_matrix.permute(0, 2, 1)
    pc_rot[:, :, :3] = torch.bmm(pc[:, :, :3], rotation_matrix)

    # retain the mask as before.
    pc_rot[:, :, 3] = pc[:, :, 3]

    return pc_rot


def find_alignment(pc_q, pc_t, alignment_model):
    # encode query and target scenes first.
    inputs_q = {'point_clouds': pc_q}
    inputs_t = {'point_clouds': pc_t}
    enc_xyz_q, enc_features_q, _ = alignment_model(inputs_q, encoder_only=True)
    enc_xyz_t, enc_features_t, _ = alignment_model(inputs_t, encoder_only=True)

    # find predicted sines and cosines for the rotation angle.
    feature_inputs = {"enc_features_q": enc_features_q, "enc_features_t": enc_features_t}
    output = alignment_model(feature_inputs, encoder_only=False)
    output_np = output.detach().cpu().numpy()
    pred_thetas_np = np.arctan2(output_np[:, 0], output_np[:, 1])
    pred_thetas = torch.from_numpy(pred_thetas_np).to(output.device)

    return pred_thetas


def train_one_epoch(
    args,
    curr_epoch,
    model,
    optimizer,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
):

    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        exact_eval=False,
    )

    curr_iter = curr_epoch * len(dataset_loader)
    max_iters = args.max_epoch * len(dataset_loader)
    net_device = next(model.parameters()).device

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)

    model.train()
    # TODO: ensuring the corr model is in evaluation mode.
    if args.ngpus > 1:
        model.module.alignment_model.eval()
    else:
        model.alignment_model.eval()
    barrier()

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
        for key in batch_data_label:
            if key != 'scan_name':
                batch_data_label[key] = batch_data_label[key].to(net_device)

        # Forward pass
        optimizer.zero_grad()

        # TODO: during training use gt rot angle to align the query and target scenes.
        pc_q_rot = rotate_pc(batch_data_label["point_clouds_with_mask"], batch_data_label["rot_angle"])

        # TODO: encode the query point cloud with mask.
        masked_inputs = {"point_clouds": pc_q_rot}
        enc_xyz_q, enc_features_q, enc_inds_q = model(masked_inputs, encoder_only=True)

        subscene_inputs = {
            "enc_xyz": enc_xyz_q,
            "enc_features": enc_features_q,
        }
        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }

        # TODO: decoding is conditioned on the subscene where target and query are aligned.
        outputs = model(inputs, subscene_inputs)

        # Compute loss
        loss, loss_dict = criterion(outputs, batch_data_label)
        loss_reduced = all_reduce_average(loss)
        loss_dict_reduced = reduce_dict(loss_dict)

        if not math.isfinite(loss_reduced.item()):
            logging.info(f"Loss in not finite. Training will be stopped.")
            sys.exit(1)

        loss.backward()
        if args.clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()

        if curr_iter % args.log_metrics_every == 0:
            # This step is slow. AP is computed approximately and locally during training.
            # It will gather outputs and ground truth across all ranks.
            # It is memory intensive as point_cloud ground truth is a large tensor.
            # If GPU memory is not an issue, uncomment the following lines.
            # outputs["outputs"] = all_gather_dict(outputs["outputs"])
            # batch_data_label = all_gather_dict(batch_data_label)
            ap_calculator.step_meter(outputs, batch_data_label)

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
            logger.log_scalars(loss_dict_reduced, curr_iter, prefix="Train_details/")

            train_dict = {}
            train_dict["lr"] = curr_lr
            train_dict["memory"] = mem_mb
            train_dict["loss"] = loss_avg.avg
            train_dict["batch_time"] = time_delta.avg
            logger.log_scalars(train_dict, curr_iter, prefix="Train/")

        curr_iter += 1
        barrier()

    return ap_calculator


@torch.no_grad()
def evaluate(
    args,
    curr_epoch,
    model,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
    curr_train_iter,
):

    # ap calculator is exact for evaluation. This is slower than the ap calculator used during training.
    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        exact_eval=True,
    )

    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    model.eval()
    barrier()
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        for key in batch_data_label:
            if key != 'scan_name':
                batch_data_label[key] = batch_data_label[key].to(net_device)

        # TODO: during evaluation predict the rotation angle to align the query and target scenes.
        if args.ngpus > 1:
            pred_thetas = find_alignment(batch_data_label["point_clouds_with_mask"], batch_data_label["point_clouds"],
                                         model.module.alignment_model)
        else:
            pred_thetas = find_alignment(batch_data_label["point_clouds_with_mask"], batch_data_label["point_clouds"],
                                         model.alignment_model)

        # TODO: rotate the query scene using the predicted angle.
        pc_q_rot = rotate_pc(batch_data_label["point_clouds_with_mask"], pred_thetas)

        # TODO: encode the query point cloud with mask.
        masked_inputs = {"point_clouds": pc_q_rot}
        enc_xyz_q, enc_features_q, enc_inds_q = model(masked_inputs, encoder_only=True)

        subscene_inputs = {
            "enc_xyz": enc_xyz_q,
            "enc_features": enc_features_q,
        }
        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }

        # TODO: decoding is conditioned on the subscene where target and query are aligned.
        outputs = model(inputs, subscene_inputs)

        # Compute loss
        loss_str = ""
        if criterion is not None:
            loss, loss_dict = criterion(outputs, batch_data_label)

            loss_reduced = all_reduce_average(loss)
            loss_dict_reduced = reduce_dict(loss_dict)
            loss_avg.update(loss_reduced.item())
            loss_str = f"Loss {loss_avg.avg:0.2f};"

        # Memory intensive as it gathers point cloud GT tensor across all ranks
        outputs["outputs"] = all_gather_dict(outputs["outputs"])
        batch_data_label = all_gather_dict(batch_data_label)
        ap_calculator.step_meter(outputs, batch_data_label)
        time_delta.update(time.time() - curr_time)
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; {loss_str} Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )

            test_dict = {}
            test_dict["memory"] = mem_mb
            test_dict["batch_time"] = time_delta.avg
            if criterion is not None:
                test_dict["loss"] = loss_avg.avg
        curr_iter += 1
        barrier()
    if is_primary():
        if criterion is not None:
            logger.log_scalars(
                loss_dict_reduced, curr_train_iter, prefix="Test_details/"
            )
        logger.log_scalars(test_dict, curr_train_iter, prefix="Test/")

    return ap_calculator
