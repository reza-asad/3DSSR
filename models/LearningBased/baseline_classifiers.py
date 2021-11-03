import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
from eval_knn_transformer import extract_features, vanilla_knn_classifier

import utils
from scripts.helper import load_from_json, write_to_json
from train_linear_classifier import compute_accuracy
from region_dataset_varying import Region
from transformer_models import Backbone


def most_frequent_cls(args, accepted_cats, cat_to_idx, topk_cat_indices, topk_cat_frequencies):
    # read all the objects in test or val and store them and their labels.
    df_metadata = pd.read_csv(args.metadata_path)
    df_metadata = df_metadata.loc[df_metadata['split'] == args.mode]
    df_metadata = df_metadata.loc[df_metadata['mpcat40'].apply(lambda x: x in accepted_cats)]
    labels = np.asarray(df_metadata['mpcat40'].apply(lambda cat: cat_to_idx[cat]))

    # assign each object randomly to the topk most frequent classes.
    predictions = np.random.choice(topk_cat_indices, len(labels), p=topk_cat_frequencies)

    return predictions, labels


def create_data_loader(args):
    # create train and test datasets
    dataset_train = Region(args.mesh_dir, args.scene_dir, args.metadata_path, args.accepted_cats_path,
                           num_local_crops=0, num_global_crops=0, mode='train', num_files=None,
                           cat_to_idx=args.cat_to_idx, num_points=args.num_point)
    dataset_val = Region(args.mesh_dir, args.scene_dir, args.metadata_path, args.accepted_cats_path,
                         num_local_crops=0, num_global_crops=0, mode=args.mode, num_files=None,
                         cat_to_idx=args.cat_to_idx, num_points=args.num_point)
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

    # convert to numpy
    train_features = train_features.cpu().detach().numpy()
    test_features = test_features.cpu().detach().numpy()
    train_labels = train_labels.cpu().detach().numpy()
    test_labels = test_labels.cpu().detach().numpy()

    # apply knn to the extracted features and compute accuracy
    predictions = vanilla_knn_classifier(train_features, train_labels, test_features, args.nb_knn)

    return predictions, test_labels


def get_args():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # paths
    parser.add_argument('--accepted_cats_path', default='../../data/matterport3d/accepted_cats.json')
    parser.add_argument('--cat_to_freq_path', dest='cat_to_freq_path',
                        default='../../data/matterport3d/accepted_cats_to_frequency.json')
    parser.add_argument('--mesh_dir', default='../../data/matterport3d/mesh_regions')
    parser.add_argument('--scene_dir', default='../../data/matterport3d/scenes')
    parser.add_argument('--metadata_path', default='../../data/matterport3d/metadata.csv')
    parser.add_argument('--cp_dir', default='../../results/matterport3d/LearningBased/')
    parser.add_argument('--results_folder_name', dest='results_folder_name',
                        default='region_classification_transformer_random_init')

    # parameters
    parser.add_argument('--mode', dest='mode', default='val')
    parser.add_argument('--topk', dest='topk', default=10)
    parser.add_argument('--nb_knn', default=10)
    parser.add_argument('--num_runs', dest='num_runs', default=10)
    parser.add_argument('--classifier_type', dest='classifier_type', default='point_transformer',
                        help='random|point_transformer')
    parser.add_argument('--batch_size_per_gpu', default=200, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
            distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')

    # transformer parameters
    parser.add_argument('--num_point', default=1024, type=int)
    parser.add_argument('--nblocks', default=4, type=int)
    parser.add_argument('--nneighbor', default=16, type=int)
    parser.add_argument('--input_dim', default=3, type=int)
    parser.add_argument('--transformer_dim', default=64, type=int)

    return parser


def main():
    # get the arguments
    parser = argparse.ArgumentParser('DINO', parents=[get_args()])
    args = parser.parse_args()
    utils.init_distributed_mode(args)
    cudnn.benchmark = True

    # create an output dir for the results
    args.cp_dir = os.path.join(args.cp_dir, args.results_folder_name)
    if not os.path.exists(args.cp_dir):
        os.makedirs(args.cp_dir)

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
    per_class_accuracies = []
    for i in range(args.num_runs):
        if args.classifier_type == 'random':
            predictions, labels = most_frequent_cls(args, accepted_cats, cat_to_idx, topk_cat_indices,
                                                    topk_cat_frequencies)
        elif args.classifier_type == 'point_transformer':
            predictions, labels = apply_point_transformer(args, data_loader_train, data_loader_val)
        else:
            raise Exception('Did not recognize classifier {}'.format(args.classifier_type))

        # find per class accuracy and record the experiment
        per_class_accuracy = {cat: (0, 0) for cat in cat_to_idx.keys()}
        compute_accuracy(per_class_accuracy, predictions, labels, cat_to_idx)
        per_class_accuracy_final = {}
        for c, (num_correct, num_total) in per_class_accuracy.items():
            per_class_accuracy_final[c] = float(num_correct) / num_total
        per_class_accuracies.append(per_class_accuracy_final)

    # take average over the N runs
    final_accuracy = {}
    for cat in cat_to_idx.keys():
        accuracies = [per_class_accuracy_final[cat] for per_class_accuracy_final in per_class_accuracies]
        final_accuracy[cat] = np.mean(accuracies)

    # save the per class accuracy over N runs.
    write_to_json(final_accuracy, os.path.join(args.cp_dir, 'per_class_accuracy.json'))


if __name__ == '__main__':
    main()




