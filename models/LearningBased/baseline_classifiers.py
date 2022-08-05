import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn

import utils
from scripts.helper import load_from_json, write_to_json
from train_point_transformer import compute_accuracy
from eval_knn_transformer import ReturnIndexDataset, ReturnIndexDatasetNorm, extract_features, knn_classifier
from transformer_models import Backbone


def most_frequent_cls(args, accepted_cats, cat_to_idx, topk_cat_indices, topk_cat_frequencies):
    # read all the objects in test or val and store them and their labels.
    df_metadata = pd.read_csv(args.metadata_path)
    df_metadata = df_metadata.loc[df_metadata['split'] == args.mode]
    df_metadata = df_metadata.loc[df_metadata['mpcat40'].apply(lambda x: x in accepted_cats)]
    labels = np.asarray(df_metadata['mpcat40'].apply(lambda cat: cat_to_idx[cat]))

    # assign each object randomly to the topk most frequent classes.
    predictions = np.random.choice(topk_cat_indices, len(labels), p=topk_cat_frequencies)

    # compute the micro average at top1
    top1 = np.sum(predictions == labels) / len(labels)

    return top1


def create_data_loader(args):
    # ============ preparing data ... ============
    if args.crop_normalized:
        dataset_train = ReturnIndexDatasetNorm(args.pc_dir, args.scene_dir, max_coord=args.max_coord, num_local_crops=0,
                                               num_global_crops=0, mode='train', cat_to_idx=args.cat_to_idx,
                                               num_points=args.num_point)
        dataset_val = ReturnIndexDatasetNorm(args.pc_dir, args.scene_dir, max_coord=args.max_coord, num_local_crops=0,
                                             num_global_crops=0, mode=args.mode, cat_to_idx=args.cat_to_idx,
                                             num_points=args.num_point)
    else:
        dataset_train = ReturnIndexDataset(args.pc_dir, args.scene_dir, num_local_crops=0, num_global_crops=0,
                                           mode='train', cat_to_idx=args.cat_to_idx, num_points=args.num_point)
        dataset_val = ReturnIndexDataset(args.pc_dir, args.scene_dir, num_local_crops=0, num_global_crops=0,
                                         mode=args.mode, cat_to_idx=args.cat_to_idx, num_points=args.num_point)

    tr_sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    val_sampler = torch.utils.data.DistributedSampler(dataset_val, shuffle=False)

    # create train and test dataloaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=tr_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=val_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    return data_loader_train, data_loader_val


def apply_point_transformer(args, data_loader_train, data_loader_val):
    # initialize a point transformer model
    model = Backbone(args)
    model.cuda()
    model.eval()

    # iterate through the data and apply the transformer.
    print("Extracting features for train set...")
    train_features, train_labels = extract_features(model, data_loader_train)
    print("Extracting features for {} set...".format(args.mode))
    test_features, test_labels = extract_features(model, data_loader_val)

    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

        if args.use_cuda:
            train_features = train_features.cuda()
            test_features = test_features.cuda()
            train_labels = train_labels.cuda()
            test_labels = test_labels.cuda()

        print("Features are ready!\nStart the k-NN classification.")

        # apply knn to the extracted features and compute accuracy
        top1, _ = knn_classifier(train_features, train_labels, test_features, test_labels, args.nb_knn,
                                 args.temperature)

    dist.barrier()

    return top1


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def get_args():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # paths
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--accepted_cats_path', default='../../data/{}/accepted_cats.json')
    parser.add_argument('--cat_to_freq_path', dest='cat_to_freq_path',
                        default='../../data/{}/accepted_cats_to_frequency.json')
    parser.add_argument('--pc_dir', default='../../data/{}/objects_pc')
    parser.add_argument('--scene_dir', default='../../data/{}/scenes')
    parser.add_argument('--metadata_path', default='../../data/{}/metadata.csv')

    # parameters
    parser.add_argument('--mode', dest='mode', default='val')
    parser.add_argument('--topk', dest='topk', default=10)
    parser.add_argument('--nb_knn', default=20)
    parser.add_argument('--temperature', default=0.07, type=float, help='Temperature used in the voting coefficient')
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag)
    parser.add_argument('--num_runs', dest='num_runs', default=10)
    parser.add_argument('--classifier_type', dest='classifier_type', default='point_transformer',
                        help='random|point_transformer')
    parser.add_argument('--batch_size_per_gpu', default=12, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
            distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')

    # transformer parameters
    parser.add_argument('--num_point', default=4096, type=int)
    parser.add_argument('--nblocks', default=4, type=int)
    parser.add_argument('--nneighbor', default=16, type=int)
    parser.add_argument('--input_dim', default=3, type=int)
    parser.add_argument('--transformer_dim', default=64, type=int)
    parser.add_argument('--crop_normalized', default=False, type=utils.bool_flag)
    parser.add_argument('--max_coord', default=15.24, type=float, help='15.24 for MP3D| 5.02 for shapenetsem')

    return parser


def main():
    # get the arguments
    parser = argparse.ArgumentParser('DINO', parents=[get_args()])
    args = parser.parse_args()
    adjust_paths(args, exceptions=['dist_url'])
    utils.init_distributed_mode(args)
    cudnn.benchmark = True

    # map each category to an index
    accepted_cats = load_from_json(args.accepted_cats_path)
    accepted_cats = sorted(accepted_cats)
    cat_to_idx = {cat: i for i, cat in enumerate(accepted_cats)}
    args.cat_to_idx = cat_to_idx

    # find the topk most frequent classes and their prob distribution
    cat_to_freq = load_from_json(args.cat_to_freq_path)
    topk_cats = [e[0] for e in sorted(cat_to_freq.items(), reverse=True, key=lambda x: x[1])[:args.topk]]
    topk_cat_indices = [cat_to_idx[cat] for cat in topk_cats]
    topk_cat_frequencies = [cat_to_freq[cat] for cat in topk_cats]
    sum_freq = np.sum(topk_cat_frequencies)
    topk_cat_frequencies = [freq/sum_freq for freq in topk_cat_frequencies]

    # create a data loader if necessary.
    if args.classifier_type in ['point_transformer']:
        data_loader_train, data_loader_val = create_data_loader(args)

    # apply the classifier N times
    all_top1s = []
    for i in range(args.num_runs):
        if args.classifier_type == 'random':
            top1 = most_frequent_cls(args, accepted_cats, cat_to_idx, topk_cat_indices, topk_cat_frequencies)
        elif args.classifier_type == 'point_transformer':
            top1 = apply_point_transformer(args, data_loader_train, data_loader_val)
        else:
            raise Exception('Did not recognize classifier {}'.format(args.classifier_type))
        all_top1s.append(top1)
        print(top1)

    print(all_top1s)
    print('Top1 ACC for {} is {}'.format(args.classifier_type, np.mean(all_top1s)))


if __name__ == '__main__':
    main()



