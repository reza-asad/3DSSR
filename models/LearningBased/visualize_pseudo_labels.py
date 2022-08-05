import os
import argparse

import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from scripts.helper import load_from_json


def filter_topk(cats, pseudo_labels, accepted_cats_topk):
    filtered_cats = []
    filtered_pseudo_labels = []
    for i, cat  in enumerate(cats):
        if cat in accepted_cats_topk:
            filtered_cats.append(cat)
            filtered_pseudo_labels.append(pseudo_labels[i])

    return filtered_cats, filtered_pseudo_labels


def plot_num_pseudo_per_cat(cat_to_pseudo_labels):
    # find the number of unique pseudo labels per cat.
    cat_to_num_pseudo_labels = {}
    for cat, pseudo_labels in cat_to_pseudo_labels.items():
        cat_to_num_pseudo_labels[cat] = len(pseudo_labels)

    # convert data to dataframe
    df = pd.DataFrame({'categories': cat_to_num_pseudo_labels.keys(),
                       'num_pseudo_labels': cat_to_num_pseudo_labels.values()})

    plt.figure(figsize=(20, 10))
    sns.set(style="whitegrid")
    ax = sns.barplot(x="categories", y="num_pseudo_labels", data=df)
    plt.show()


def plot_jaccard_sim(cat_to_pseudo_labels):
    # create data frame with index and columns cats.
    df = pd.DataFrame(columns=cat_to_pseudo_labels.keys(), index=cat_to_pseudo_labels.keys(), dtype=float)

    # compute jaccard similarity and populate the corresponding cell in the frame.
    for cat_a, pseudo_labels_a in cat_to_pseudo_labels.items():
        for cat_b, pseudo_labels_b in cat_to_pseudo_labels.items():
            jaccard_sim = len(pseudo_labels_a.intersection(pseudo_labels_b)) / len(pseudo_labels_a.union(pseudo_labels_b))
            df.loc[cat_a, cat_b] = jaccard_sim

    # plot the correlation.
    plt.figure(figsize=(20, 20))
    sns.heatmap(df)
    plt.show()


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def get_args():
    parser = argparse.ArgumentParser('Visualize Pseudo Labels', add_help=False)
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--accepted_cats_path', default='../../data/{}/accepted_cats.json')
    parser.add_argument('--accepted_cats_topk_path', default='../../data/{}/accepted_cats_top10.json')
    parser.add_argument('--metadata_path', dest='metadata_path', default='../../data/{}/metadata.csv')
    parser.add_argument('--cp_dir', default='../../results/{}/LearningBased/')
    parser.add_argument('--results_folder_name', default='3D_DINO_full')
    parser.add_argument('--features_dir_name', default='features_50', type=str)
    parser.add_argument('--pseudo_labels_file_name', default='pseudo_labels.pth', type=str)
    parser.add_argument('--train_labels_file_name', default='labels.pth', type=str)
    parser.add_argument('--filter_topk', action='store_true', default=True,
                        help='If True top 10 most frequent furniture like objects are selected.')

    return parser


def main():
    # get the arguments
    parser = argparse.ArgumentParser('Visualize Pseudo Labels', parents=[get_args()])
    args = parser.parse_args()
    args.pseudo_labels_file_name = '{}_{}'.format(args.mode, args.pseudo_labels_file_name)
    args.train_labels_file_name = '{}{}'.format(args.mode, args.train_labels_file_name)
    args.cp_dir = os.path.join(args.cp_dir, args.results_folder_name, args.features_dir_name)
    adjust_paths(args, exceptions=['dist_url'])

    # load the train and pseudo labels
    pseudo_labels = torch.load(os.path.join(args.cp_dir, args.pseudo_labels_file_name))
    labels = torch.load(os.path.join(args.cp_dir, args.train_labels_file_name))
    pseudo_labels = pseudo_labels.cpu().detach().numpy().squeeze(axis=1)
    labels = labels.cpu().detach().numpy()

    # map each label idx to a mpcat40 category.
    accepted_cats = load_from_json(args.accepted_cats_path)
    accepted_cats = sorted(accepted_cats)
    idx_to_cat = {i: cat for i, cat in enumerate(accepted_cats)}
    cats = [idx_to_cat[idx] for idx in labels[:, 0]]

    print('{} different categories'.format(len(set(cats))))
    print('{} different pseudo labels'.format(len(set(pseudo_labels))))
    print('*'*50)

    # restrict categories to the
    if args.filter_topk:
        accepted_cats_topk = load_from_json(args.accepted_cats_topk_path)
        cats, pseudo_labels = filter_topk(cats, pseudo_labels, accepted_cats_topk)

    print('{} different categories after filtration'.format(len(set(cats))))
    print('{} different pseudo labels after filtration'.format(len(set(pseudo_labels))))

    # map cats to pseudo labels.
    cat_to_pseudo_labels = {}
    for i, cat in enumerate(cats):
        if cat not in cat_to_pseudo_labels:
            cat_to_pseudo_labels[cat] = {pseudo_labels[i]}
        else:
            cat_to_pseudo_labels[cat].add(pseudo_labels[i])

    # number of unique pseudo labels per cat.
    # plot_num_pseudo_per_cat(cat_to_pseudo_labels)

    # jaccard similarity between the categories based on common pseudo labels
    # plot_jaccard_sim(cat_to_pseudo_labels)


if __name__ == '__main__':
    main()
