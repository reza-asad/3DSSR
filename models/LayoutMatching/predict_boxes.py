# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import os
import sys

import numpy as np
import torch
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader, DistributedSampler

# 3DETR codebase specific imports
from datasets import build_dataset
from engine import evaluate
from models import build_model
from utils.misc import my_worker_init_fn
from utils.logger import Logger
import json


def make_args_parser():
    parser = argparse.ArgumentParser("Saving Detected Boxes", add_help=False)

    ##### Model #####
    parser.add_argument(
        "--model_name",
        default="3detr",
        type=str,
        help="Name of the model",
        choices=["3detr"],
    )
    ### Encoder
    parser.add_argument(
        "--enc_type", default="vanilla", choices=["masked", "maskedv2", "vanilla"]
    )
    # Below options are only valid for vanilla encoder
    parser.add_argument("--num_mask_feats", default=1, type=int)
    parser.add_argument("--enc_nlayers", default=3, type=int)
    parser.add_argument("--enc_dim", default=256, type=int)
    parser.add_argument("--enc_ffn_dim", default=128, type=int)
    parser.add_argument("--enc_dropout", default=0.1, type=float)
    parser.add_argument("--enc_nhead", default=4, type=int)
    parser.add_argument("--enc_pos_embed", default=None, type=str)
    parser.add_argument("--enc_activation", default="relu", type=str)

    ### Decoder
    parser.add_argument("--dec_nlayers", default=8, type=int)
    parser.add_argument("--dec_dim", default=256, type=int)
    parser.add_argument("--dec_ffn_dim", default=256, type=int)
    parser.add_argument("--dec_dropout", default=0.1, type=float)
    parser.add_argument("--dec_nhead", default=4, type=int)

    ### MLP heads for predicting bounding boxes
    parser.add_argument("--mlp_dropout", default=0.3, type=float)
    parser.add_argument(
        "--nsemcls",
        default=-1,
        type=int,
        help="Number of semantic object classes. Can be inferred from dataset",
    )

    ### Other model params
    parser.add_argument("--preenc_npoints", default=2048, type=int)
    parser.add_argument("--n_rot", default=4, type=int)
    parser.add_argument("--norm_2d", default=False, action="store_true")
    parser.add_argument(
        "--pos_embed", default="fourier", type=str, choices=["fourier", "sine"]
    )
    parser.add_argument("--nqueries", default=256, type=int)
    parser.add_argument("--use_color", default=False, action="store_true")
    parser.add_argument("--aggressive_rot", default=False, action="store_true")

    ##### Dataset #####
    parser.add_argument(
        "--dataset_name", required=True, type=str, choices=["scannet", "sunrgbd"]
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default=None,
        help="Root directory containing the dataset files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument(
        "--meta_data_dir",
        type=str,
        default=None,
        help="Root directory containing the metadata files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument("--dataset_num_workers", default=4, type=int)
    parser.add_argument("--batchsize_per_gpu", default=8, type=int)

    ##### Testing #####
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--test_ckpt", default=None, type=str)
    parser.add_argument("--output_path", default='../../results/{}/LayoutMatching/rendered_results', type=str)
    parser.add_argument("--experiment_name", default='', type=str)

    ##### I/O #####
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--log_metrics_every", default=20, type=int)

    return parser


def test_model(args, model, model_no_ddp, dataset_config, dataloaders):
    if args.test_ckpt is None or not os.path.isfile(args.test_ckpt):
        f"Please specify a test checkpoint using --test_ckpt. Found invalid value {args.test_ckpt}"
        sys.exit(1)

    sd = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
    model_no_ddp.load_state_dict(sd["model"])
    logger = Logger()
    criterion = None  # do not compute loss for speed-up; Comment out to see test loss
    epoch = -1
    curr_iter = 0
    # for visualization set per class proposal off.
    ap_calculator = evaluate(
        args,
        epoch,
        model,
        criterion,
        dataset_config,
        dataloaders["test"],
        logger,
        curr_iter,
        per_class_proposal=False
    )
    metrics = ap_calculator.compute_metrics()
    metric_str = ap_calculator.metrics_to_str(metrics)
    print("==" * 10)
    print(f"Test model; Metrics {metric_str}")
    print("==" * 10)

    # load all the scan names
    scan_names = []
    for batch_idx, batch_data_label in enumerate(dataloaders["test"]):
        scan_name_batch = batch_data_label['scan_name']
        scan_names += scan_name_batch

    # load the predicted and gt boxes.
    predictions = ap_calculator.pred_map_cls
    gt = ap_calculator.gt_map_cls

    # iterate through each scan name and record the ground truth and predictions.
    query_predictions = {'query': [], 'predictions': [], 'scene_names': []}
    for idx, scan_name in enumerate(scan_names):
        # add scene name
        query_predictions['scene_names'].append(scan_name)

        # add gt query.
        gt_scene = gt[idx]
        curr_query = []
        for obj_idx, gt_obj in enumerate(gt_scene):
            gt_obj_info = {'id': obj_idx, 'class_id': gt_obj[0], 'box': gt_obj[1].astype(float).tolist()}
            curr_query.append(gt_obj_info)
        query_predictions['query'].append(curr_query)

        # add predictions for the query.
        pred_scene = predictions[idx]
        curr_pred = []
        for obj_idx, pred_obj in enumerate(pred_scene):
            pred_obj_info = {'id': obj_idx, 'class_id': pred_obj[0], 'box': pred_obj[1].astype(float).tolist(),
                             'score': float(pred_obj[2])}
            curr_pred.append(pred_obj_info)
        query_predictions['predictions'].append(curr_pred)

    # save the results.
    output_path = os.path.join(args.output_path, 'query_predictions.json')
    with open(output_path, 'w') as f:
        json.dump(query_predictions, f, indent=4)


def main():
    print(f"Called with args: {args}")
    torch.cuda.set_device(0)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # make the output dir if it does not exist.
    args.output_path = os.path.join(args.output_path, args.experiment_name)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    datasets, dataset_config = build_dataset(args)
    model, _ = build_model(args, dataset_config)
    model = model.cuda(0)
    model_no_ddp = model

    dataloaders = {}
    dataset_splits = ["test"]
    for split in dataset_splits:
        sampler = torch.utils.data.SequentialSampler(datasets[split])

        dataloaders[split] = DataLoader(
            datasets[split],
            sampler=sampler,
            batch_size=args.batchsize_per_gpu,
            num_workers=args.dataset_num_workers,
            worker_init_fn=my_worker_init_fn,
            collate_fn=collate_fn
        )
        dataloaders[split + "_sampler"] = sampler

    test_model(args, model, model_no_ddp, dataset_config, dataloaders)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def adjust_paths(exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset_name)
            vars(args)[k] = os.path.join(base_dir, v)


if __name__ == "__main__":
    parser = make_args_parser()
    args = parser.parse_args()
    adjust_paths([])
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()
