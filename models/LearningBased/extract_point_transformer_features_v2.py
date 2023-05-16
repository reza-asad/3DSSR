# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse

import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from scripts.helper import load_from_json, write_to_json
import utils
from region_dataset import Region
from models.LearningBased.region_dataset_normalized_crop import Region as RegionNorm
from transformer_models import PointTransformerSeg
from projection_models import DINOHead


def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    if args.crop_normalized:
        dataset = ReturnIndexDatasetNorm(args.pc_dir, args.scene_dir, args.metadata_path, max_coord=args.max_coord,
                                         num_local_crops=0, num_global_crops=0, mode=args.mode,
                                         cat_to_idx=args.cat_to_idx, num_points=args.num_point,
                                         file_name_to_idx=args.file_name_to_idx, theta=args.theta)
    else:
        dataset = ReturnIndexDataset(args.pc_dir, args.scene_dir, args.metadata_path, num_local_crops=0,
                                     num_global_crops=0, mode=args.mode, cat_to_idx=args.cat_to_idx,
                                     num_points=args.num_point, file_name_to_idx=args.file_name_to_idx)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset)} {args.mode} imgs.")

    # ============ building network ... ============
    if args.classifier_type == 'supervised':
        backbone = PointTransformerSeg(args)
        head = DINOHead(in_dim=args.transformer_dim, out_dim=args.num_class, use_bn=args.use_bn_in_head,
                        norm_last_layer=args.norm_last_layer)
        classifier = utils.DINO(backbone, head, num_local_crops=0, num_global_crops=1, network_type='teacher')
        classifier = torch.nn.DataParallel(classifier).cuda()
        checkpoint = torch.load(os.path.join(args.cp_dir, args.pretrained_weights_name))
        classifier.load_state_dict(checkpoint['model_state_dict'])
        model = classifier.module.backbone
    elif args.classifier_type == 'DINO':
        pretrained_weights_dir = os.path.join(args.cp_dir, args.pretrained_weights_name)
        backbone = PointTransformerSeg(args)
        head = DINOHead(in_dim=args.transformer_dim, out_dim=args.out_dim, use_bn=args.use_bn_in_head)
        DINO = utils.DINO(backbone, head, num_local_crops=0, num_global_crops=1, network_type='teacher')
        DINO.cuda()
        utils.load_pretrained_weights(DINO, pretrained_weights_dir, args.checkpoint_key)
        model = DINO.backbone
    else:
        raise NotImplementedError('model {} is not recognized'.format(args.classifier_type))

    model.eval()

    # ============ extract features ... ============
    print(f"Extracting features for {args.mode} set...")
    features, labels, file_names = extract_features(model, data_loader, args.use_cuda)
    if utils.get_rank() == 0:
        features = nn.functional.normalize(features, dim=1, p=2)

    # save features and labels
    if dist.get_rank() == 0:
        torch.save(features.cpu(), os.path.join(args.features_dir, f"{args.mode}feat.pth"))
        torch.save(labels.cpu(), os.path.join(args.features_dir, f"{args.mode}labels.pth"))
        torch.save(file_names.cpu(), os.path.join(args.features_dir, f"{args.mode}_file_names.pth"))


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    labels_combined = None
    file_names_combined = None
    for data, index in metric_logger.log_every(data_loader, 10):
        file_names = data['file_name'].long().cuda()
        samples = data['crops'][0]
        samples = samples.cuda(non_blocking=use_cuda)

        labels = data['labels'].long().cuda()

        index = index.cuda(non_blocking=use_cuda)

        output = model(samples)
        feats = output.mean(1)

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            features = features.cuda(non_blocking=use_cuda)
            print(f"Storing features into tensor of shape {features.shape}")

            labels_combined = torch.zeros(len(data_loader.dataset), 1)
            labels_combined = labels_combined.long().cuda(non_blocking=use_cuda)
            print(f"Storing labels into tensor of shape {labels_combined.shape}")

            file_names_combined = torch.zeros(len(data_loader.dataset), 1)
            file_names_combined = file_names_combined.long().cuda(non_blocking=True)
            print(f"Storing file_names into tensor of shape {file_names_combined.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        labels_all = torch.empty(
            dist.get_world_size(),
            labels.size(0),
            labels.size(1),
            dtype=labels.dtype,
            device=labels.device,
        )
        file_names_all = torch.empty(
            dist.get_world_size(),
            file_names.size(0),
            file_names.size(1),
            dtype=file_names.dtype,
            device=file_names.device,
        )

        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        output_l_labels = list(labels_all.unbind(0))
        output_all_reduce_labels = torch.distributed.all_gather(output_l_labels, labels, async_op=True)
        output_all_reduce_labels.wait()

        output_l_file_names = list(file_names_all.unbind(0))
        output_all_reduce_file_names = torch.distributed.all_gather(output_l_file_names, file_names, async_op=True)
        output_all_reduce_file_names.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            features.index_copy_(0, index_all, torch.cat(output_l))
            labels_combined.index_copy_(0, index_all, torch.cat(output_l_labels))
            file_names_combined.index_copy_(0, index_all, torch.cat(output_l_file_names))

    return features, labels_combined, file_names_combined


class ReturnIndexDataset(Region):
    def __getitem__(self, idx):
        data = super(ReturnIndexDataset, self).__getitem__(idx)
        return data, idx


class ReturnIndexDatasetNorm(RegionNorm):
    def __getitem__(self, idx):
        data = super(ReturnIndexDatasetNorm, self).__getitem__(idx)
        return data, idx


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on Matterport3D')
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--mode', default='test', help='train|val|test')
    parser.add_argument('--accepted_cats_path', default='../../data/{}/accepted_cats.json')
    parser.add_argument('--metadata_path', default='../../data/{}/metadata.csv')
    parser.add_argument('--pc_dir', default='../../data/{}/pc_regions')
    parser.add_argument('--cp_dir', default='../../results/{}/LearningBased/')
    parser.add_argument('--results_folder_name', dest='results_folder_name', default='3D_DINO_regions_non_equal_full_top10_seg')
    parser.add_argument('--scene_dir', default='../../data/{}/scenes')
    parser.add_argument('--features_dir_name', default='features', type=str)

    # point transformer arguments
    parser.add_argument('--classifier_type', default='DINO', help='supervised | DINO')
    parser.add_argument('--num_point', default=4096, type=int)
    parser.add_argument('--theta', default=0, type=int)
    parser.add_argument('--nblocks', default=2, type=int)
    parser.add_argument('--nneighbor', default=16, type=int)
    parser.add_argument('--input_dim', default=3, type=int)
    parser.add_argument('--transformer_dim', default=32, type=int)
    parser.add_argument('--crop_normalized', default=True, type=utils.bool_flag)
    parser.add_argument('--max_coord', default=3.65, type=float, help='3.65 for MP3D')
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag)
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag)
    parser.add_argument('--out_dim', dest='out_dim', default=2000, type=int)
    parser.add_argument('--num_class', dest='num_class', default=28, type=int)

    parser.add_argument('--batch_size_per_gpu', default=8, type=int, help='Per-GPU batch-size')
    parser.add_argument('--pretrained_weights_name', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--num_workers', default=6, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    args = parser.parse_args()
    adjust_paths(args, exceptions=['dist_url'])

    # reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    # find a mapping from the accepted categories into indices
    accepted_cats = load_from_json(args.accepted_cats_path)
    accepted_cats = sorted(accepted_cats)
    cat_to_idx = {cat: i for i, cat in enumerate(accepted_cats)}

    # load the metadata.
    df = pd.read_csv(args.metadata_path)

    # category based filtering and parameters.
    args.cat_to_idx = None
    if 'mpcat40' in df.keys():
        args.cat_to_idx = cat_to_idx
        is_accepted = df['mpcat40'].apply(lambda x: x in args.cat_to_idx)
        df = df.loc[is_accepted]

    # find a mapping from the region files to their indices.
    df = df.loc[(df['split'] == 'train') | (df['split'] == args.mode)]
    file_names = df[['room_name', 'objectId']].apply(lambda x: '-'.join([str(x[0]), str(x[1]) + '.npy']), axis=1).tolist()
    file_names = sorted(file_names)
    file_name_to_idx = {file_name: i for i, file_name in enumerate(file_names)}
    args.file_name_to_idx = file_name_to_idx

    # prepare the checkpoint dir
    args.cp_dir = os.path.join(args.cp_dir, args.results_folder_name)

    # create a directory to load or dump the features.
    if args.theta != 0:
        args.features_dir_name = args.features_dir_name + '_{}'.format(args.theta)
    args.features_dir = os.path.join(args.cp_dir, args.features_dir_name)
    if not os.path.exists(args.features_dir):
        try:
            os.makedirs(args.features_dir)
        except FileExistsError:
            pass

    # extract features and labels
    extract_feature_pipeline(args)

    dist.barrier()

