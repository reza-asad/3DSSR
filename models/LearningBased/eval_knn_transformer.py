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


def compute_macro_average(predictions, test_labels, cat_to_idx):
    # find a mapping from the label indices to the actual category
    idx_to_cat = {idx: cat for cat, idx in cat_to_idx.items()}

    # compute accuracy per category
    accuracy_per_cat = {}
    label_indices = cat_to_idx.values()
    for idx in label_indices:
        is_idx = test_labels == idx
        num_correct = torch.sum(test_labels[is_idx, ...] == predictions[is_idx, ...]).item()
        num_total = torch.sum(is_idx).item()
        accuracy_per_cat[idx_to_cat[idx]] = num_correct / num_total * 100
    print('Accuracy per category: {}'.format(accuracy_per_cat))

    # compute the macro average
    print('Macro average is {}'.format(np.mean(list(accuracy_per_cat.values()))))


def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    if args.crop_normalized:
        dataset_train = ReturnIndexDatasetNorm(args.pc_dir, args.scene_dir, args.metadata_path,
                                               max_coord=args.max_coord, num_local_crops=0, num_global_crops=0,
                                               mode='train', cat_to_idx=args.cat_to_idx, num_points=args.num_point,
                                               file_name_to_idx=args.file_name_to_idx)
        dataset_val = ReturnIndexDatasetNorm(args.pc_dir, args.scene_dir, args.metadata_path, max_coord=args.max_coord,
                                             num_local_crops=0, num_global_crops=0, mode=args.mode,
                                             cat_to_idx=args.cat_to_idx, num_points=args.num_point,
                                             file_name_to_idx=args.file_name_to_idx)
    else:
        dataset_train = ReturnIndexDataset(args.pc_dir, args.scene_dir, args.metadata_path, num_local_crops=0,
                                           num_global_crops=0, mode='train', cat_to_idx=args.cat_to_idx,
                                           num_points=args.num_point, file_name_to_idx=args.file_name_to_idx)
        dataset_val = ReturnIndexDataset(args.pc_dir, args.scene_dir, args.metadata_path, num_local_crops=0,
                                         num_global_crops=0, mode=args.mode, cat_to_idx=args.cat_to_idx,
                                         num_points=args.num_point, file_name_to_idx=args.file_name_to_idx)

    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} {args.mode} imgs.")

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
        raise ValueError('model {} is not recognized'.format(args.classifier_type))

    model.eval()

    # ============ extract features ... ============
    if args.load_features_train:
        train_features = torch.load(os.path.join(args.features_dir, "trainfeat.pth"))
        train_labels = torch.load(os.path.join(args.features_dir, "trainlabels.pth"))
        train_file_names = torch.load(os.path.join(args.features_dir, "train_file_names.pth"))
    else:
        print("Extracting features for train set...")
        train_features, train_labels, train_file_names = extract_features(model, data_loader_train, args.use_cuda)

    if args.load_features_test:
        test_features = torch.load(os.path.join(args.features_dir, "{}feat.pth".format(args.mode)))
        test_labels = torch.load(os.path.join(args.features_dir, "{}labels.pth".format(args.mode)))
        test_file_names = torch.load(os.path.join(args.features_dir, "{}_file_names.pth".format(args.mode)))
    else:
        print("Extracting features for {} set...".format(args.mode))
        test_features, test_labels, test_file_names = extract_features(model, data_loader_val, args.use_cuda)

    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    # save features and labels
    if dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(args.features_dir, "trainfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.features_dir, "trainlabels.pth"))
        torch.save(train_file_names.cpu(), os.path.join(args.features_dir, "train_file_names.pth"))

        torch.save(test_features.cpu(), os.path.join(args.features_dir, "{}feat.pth".format(args.mode)))
        torch.save(test_labels.cpu(), os.path.join(args.features_dir, "{}labels.pth".format(args.mode)))
        torch.save(test_file_names.cpu(), os.path.join(args.features_dir, "{}_file_names.pth".format(args.mode)))

    return train_features, test_features, train_labels, test_labels, train_file_names, test_file_names


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


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes):
    top1, total = 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    all_predictions = []
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)
        all_predictions.append(predictions[:, 0:1])

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        total += targets.size(0)

    top1 = top1 * 100.0 / total
    all_predictions = torch.cat(all_predictions, dim=0)

    return top1, all_predictions


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
    parser.add_argument('--mode', default='val')
    parser.add_argument('--accepted_cats_path', default='../../data/{}/accepted_cats_top10.json')
    parser.add_argument('--metadata_path', default='../../data/{}/metadata_non_equal_full_top10.csv')
    parser.add_argument('--pc_dir', dest='pc_dir', default='../../data/{}/pc_regions')
    parser.add_argument('--cp_dir', default='../../results/{}/LearningBased/')
    parser.add_argument('--results_folder_name', dest='results_folder_name', default='3D_DINO_regions_non_equal_full_top10_seg')
    parser.add_argument('--scene_dir', default='../../data/{}/scenes')
    parser.add_argument('--output_file_name', default='predicted_labels_knn.json', type=str)
    parser.add_argument('--features_dir_name', default='features', type=str)

    # point transformer arguments
    parser.add_argument('--classifier_type', default='DINO', help='supervised | DINO')
    parser.add_argument('--num_point', default=4096, type=int)
    parser.add_argument('--nblocks', default=2, type=int)
    parser.add_argument('--nneighbor', default=16, type=int)
    parser.add_argument('--input_dim', default=3, type=int)
    parser.add_argument('--transformer_dim', default=32, type=int)
    parser.add_argument('--crop_normalized', default=True, type=utils.bool_flag)
    parser.add_argument('--max_coord', default=3.65, type=float, help='14.30 for MP3D| 5.02 for shapenetsem')
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag)
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag)
    parser.add_argument('--out_dim', dest='out_dim', default=2000, type=int)

    parser.add_argument('--batch_size_per_gpu', default=8, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--final_nbb', default=20, type=int)
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights_name', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--load_features_train', default=False, type=utils.bool_flag,
        help="if true train features are loaded")
    parser.add_argument('--load_features_test', default=False, type=utils.bool_flag,
        help="if true test features are derived")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
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
    args.cat_to_idx = cat_to_idx
    args.num_class = len(cat_to_idx)

    # find a mapping from the region files to their indices.
    df = pd.read_csv(args.metadata_path)
    is_accepted = df['mpcat40'].apply(lambda x: x in args.cat_to_idx)
    df = df.loc[is_accepted]
    df = df.loc[(df['split'] == 'train') | (df['split'] == args.mode)]
    file_names = df[['room_name', 'objectId']].apply(lambda x: '-'.join([str(x[0]), str(x[1]) + '.npy']), axis=1).tolist()
    file_names = sorted(file_names)
    file_name_to_idx = {file_name: i for i, file_name in enumerate(file_names)}
    args.file_name_to_idx = file_name_to_idx

    # prepare the checkpoint dir
    args.cp_dir = os.path.join(args.cp_dir, args.results_folder_name)

    # create a directory to load or dump the features.
    args.features_dir = os.path.join(args.cp_dir, args.features_dir_name)
    if not os.path.exists(args.features_dir):
        try:
            os.makedirs(args.features_dir)
        except FileExistsError:
            pass

    # extract or load features and labels
    train_features, test_features, train_labels, test_labels, train_file_names, test_file_names = \
        extract_feature_pipeline(args)

    if utils.get_rank() == 0:
        if args.use_cuda:
            train_features = train_features.cuda()
            test_features = test_features.cuda()
            train_labels = train_labels.cuda()
            test_labels = test_labels.cuda()

        print("Features are ready!\nStart the k-NN classification.")
        final_predictions = None
        for k in args.nb_knn:
            top1, predictions = knn_classifier(train_features, train_labels, test_features, test_labels, k,
                                               args.temperature, num_classes=args.num_class)
            print(f"{k}-NN classifier result: Top1: {top1}")
            if k == args.final_nbb:
                final_predictions = predictions

        # compute the macro average
        compute_macro_average(final_predictions, test_labels, cat_to_idx)

        # save predictions.
        idx_to_file_name = {idx: file_name for file_name, idx in file_name_to_idx.items()}
        file_name_to_predictions = {}
        for i, prediction in enumerate(final_predictions):
            file_name = idx_to_file_name[test_file_names[i].item()]
            file_name_to_predictions[file_name] = prediction[0].item()
        args.output_file_name = args.output_file_name.split('.')[0] + '_{}.json'.format(args.mode)
        write_to_json(file_name_to_predictions, os.path.join(args.cp_dir, args.features_dir_name,
                                                             args.output_file_name))

    dist.barrier()

