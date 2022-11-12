# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import os
import sys

import json
import trimesh
import numpy as np
import torch
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader, DistributedSampler

# 3DETR codebase specific imports
from datasets import build_dataset
from engine import evaluate_real_query
from models import build_model
from utils.misc import my_worker_init_fn
from utils.logger import Logger
from scripts.helper import load_from_json, write_to_json
from scripts.box import Box


def make_args_parser():
    parser = argparse.ArgumentParser("Saving Detected Boxes", add_help=False)

    ##### Model #####
    parser.add_argument(
        "--model_name",
        default="3dssr",
        type=str,
        help="Name of the model",
        choices=["3detr", "3dssr"],
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
    parser.add_argument("--npos_pairs", default=256, type=int)
    parser.add_argument("--tempreature", default=0.4, type=float)
    parser.add_argument("--crop_factor", default=0.1, type=float)
    parser.add_argument("--use_color", default=False, action="store_true")

    ##### Dataset #####
    parser.add_argument(
        "--dataset_name", required=True, type=str, choices=["scannet", "sunrgbd", "matterport3d", "matterport3d_real"]
    )
    parser.add_argument("--test_split", default='real', type=str, choices=["test", "val"])
    parser.add_argument('--accepted_cats_path', default='../../data/{}/accepted_cats.json')
    parser.add_argument('--scene_dir', default='../../results/{}/scenes')
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
    parser.add_argument("--aggressive_rot", default=False, action="store_true")
    parser.add_argument("--augment_eval", default=False, action="store_true")

    ##### Testing #####
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--test_ckpt", default=None, type=str)
    parser.add_argument("--corr_model_ckpt", default=None, type=str)
    parser.add_argument("--output_path", default='../../results/{}/LayoutMatching/', type=str)
    parser.add_argument('--query_dir', default='../../queries/{}/')
    parser.add_argument('--query_input_file_name', default='query_dict_top10.json')
    parser.add_argument('--results_folder_name', default='full_3dssr_real_query')
    parser.add_argument("--experiment_name", default='3detr_pre_rank', type=str)

    ##### I/O #####
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--log_metrics_every", default=20, type=int)
    parser.add_argument("--ngpus", default=1, type=int)

    return parser


def format_bbox(box_corners):
    box_corners[..., 1] *= -1
    box_corners[..., [0, 1, 2]] = box_corners[..., [0, 2, 1]]

    # find the centroid of the box.
    centroid = np.mean(box_corners, axis=0)

    # compute scale
    d1 = np.max(box_corners[:, 0]) - np.min(box_corners[:, 0])
    d2 = np.max(box_corners[:, 1]) - np.min(box_corners[:, 1])
    d3 = np.max(box_corners[:, 2]) - np.min(box_corners[:, 2])
    scale = [d1, d2, d3]

    bbox = Box.from_transformation(np.eye(3), centroid, scale)

    return bbox.vertices


def test_model(args, model, model_no_ddp):
    # load the model
    if args.test_ckpt is None or not os.path.isfile(args.test_ckpt):
        f"Please specify a test checkpoint using --test_ckpt. Found invalid value {args.test_ckpt}"
        sys.exit(1)

    sd = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
    model_no_ddp.load_state_dict(sd["model"])
    logger = Logger()
    criterion = None  # do not compute loss for speed-up; Comment out to see test loss
    epoch = -1
    curr_iter = 0

    # for each query build dataset, dataloader and detect 3d subscenes.
    query_results = {}
    for idx, query in enumerate(query_dict.keys()):
        print('Processing {}/{} queries: {}.'.format(idx+1, len(query_dict), query))
        # fill in the query results with the query itself first
        query_results[query] = query_dict[query]

        # apply the trained model on the real query.
        args.query_info = query_dict[query]
        datasets, dataset_config = build_dataset(args)
        dataloaders = build_data_loader(datasets, dataset_config)
        ap_calculator, scan_names = evaluate_real_query(
            args,
            epoch,
            model,
            criterion,
            dataset_config,
            dataloaders[args.test_split],
            logger,
            curr_iter,
            per_class_proposal=False
        )

        # load all predictions
        predictions = ap_calculator.pred_map_cls

        # record and sort the retrieved target subscenes for the query.
        target_subscenes = []
        for i, scan_name in enumerate(scan_names):
            # load the predictions for the current scene.
            predictions_scene = predictions[i]
            categories, boxes, scores = [], [], []
            for j, predictions_obj in enumerate(predictions_scene):
                categories.append(class_id_to_cat[predictions_obj[0]])
                bbox = format_bbox(predictions_obj[1].astype(float))
                boxes.append(bbox.tolist())
                scores.append(float(predictions_obj[2]))

            # the template for retrieved target subscens.
            target_subscene = {'scene_name': scan_name, 'boxes': boxes, 'cats': categories, 'scores': scores}
            target_subscenes.append(target_subscene)

        query_results[query]['target_subscenes'] = target_subscenes

    # save the results.
    write_to_json(query_results, query_dict_output_path)


def build_data_loader(datasets, dataset_config):
    dataloaders = {}
    dataset_splits = [args.test_split]
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

    return dataloaders


def main():
    print(f"Called with args: {args}")
    torch.cuda.set_device(0)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    datasets, dataset_config = build_dataset(args)
    model, _ = build_model(args, dataset_config)
    model = model.cuda(0)
    model_no_ddp = model

    # load the pre-trained weights for the seed corr model
    corr_model = torch.load(args.corr_model_ckpt, map_location=torch.device("cpu"))
    model_no_ddp.corr_model.load_state_dict(corr_model["model"])

    test_model(args, model, model_no_ddp)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def adjust_paths(exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset_name.split('_')[0])
            vars(args)[k] = os.path.join(base_dir, v)


if __name__ == "__main__":
    parser = make_args_parser()
    args = parser.parse_args()
    args.query_info = None
    args.scene_dir = os.path.join(args.scene_dir, args.test_split)
    adjust_paths([])

    # load the query dict.
    args.query_dir = os.path.join(args.query_dir, args.test_split)
    query_dict_input_path = os.path.join(args.query_dir, args.query_input_file_name)
    query_dict = load_from_json(query_dict_input_path)

    # create an output path for the retrieval results.
    output_path = os.path.join(args.output_path, args.results_folder_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    query_output_file_name = args.query_input_file_name.split('.')[0] + '_{}_{}.json'.format(args.test_split,
                                                                                             args.experiment_name)
    query_dict_output_path = os.path.join(output_path, query_output_file_name)

    # find a mapping from class_ids to their category.
    cats = load_from_json(args.accepted_cats_path)
    class_id_to_cat = dict(zip(range(len(cats)), sorted(cats)))

    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()
