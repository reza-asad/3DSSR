import os
import argparse

import extract_rank_subscenes
import cluster_embeddings
from scripts.helper import load_from_json


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def get_args():
    parser = argparse.ArgumentParser('Extracting and Ranking 3D Subscenes', add_help=False)

    parser.add_argument('--model_name', default='dino_50_full_config',
                        help='choose one from 3dssr_model_configs.json')
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--mode', dest='mode', default='test', help='val or test')
    parser.add_argument('--accepted_cats_path', default='../../data/{}/accepted_cats.json')
    parser.add_argument('--metadata_path_queries', default='../../data/{}/metadata.csv')
    parser.add_argument('--metadata_path', default='../../data/{}/metadata.csv')
    parser.add_argument('--scene_dir_queries', default='../../results/{}/scenes')
    parser.add_argument('--scene_dir', default='../../results/{}/scenes')
    parser.add_argument('--query_dir', default='../../queries/{}/')
    parser.add_argument('--query_input_file_name', default='query_dict_top10.json')
    parser.add_argument('--include_embedding_sim', action='store_true', default=True)
    parser.add_argument('--with_cat_predictions', action='store_true', default=False)
    parser.add_argument('--with_cluster_predictions', action='store_true', default=False)
    parser.add_argument('--with_rotations', action='store_true', default=False)
    parser.add_argument('--grid_res', type=float, default=0)
    parser.add_argument('--iou_threshold', type=float, default=0)
    parser.add_argument('--k_max', type=int, default=15)
    parser.add_argument('--predicted_labels_file_name', default='predicted_labels.json', type=str)
    parser.add_argument('--predicted_clusters_file_name', default='predicted_clusters.json', type=str)
    parser.add_argument('--model_config_filename', default='3dssr_model_configs.json')
    parser.add_argument('--topk', dest='topk', default=100, help='Number of most similar subscenes to be returned.')

    return parser


def main():
    # read the args
    parser = argparse.ArgumentParser('Extracting and Ranking 3D Subscenes', parents=[get_args()])
    args = parser.parse_args()

    # determine the correct config for the pretraining strategy
    configs = load_from_json(args.model_config_filename)[args.model_name]

    # add the pretraining configs and apply 3dssr and adjust paths.
    for k, v in configs.items():
        vars(args)[k] = v
    adjust_paths(args, exceptions=[])

    # set the correct mode for required fields
    args.query_dir = os.path.join(args.query_dir, args.mode)
    args.predicted_labels_file_name = args.predicted_labels_file_name.split('.')[0] + '_{}.json'.format(args.mode)
    args.predicted_clusters_file_name = args.predicted_clusters_file_name.split('.')[0] + '_{}.json'.format(args.mode)

    # apply clustering on the training features if necessary.
    if args.with_cluster_predictions:
        cluster_embeddings.cluster(args)

    # find similarity threshold
    args.sim_threshold = cluster_embeddings.find_sim_threshold(args)
    print('Using a Similarity Threshold of {}'.format(args.sim_threshold))

    # apply 3dssr
    extract_rank_subscenes.apply_3dssr(args)


if __name__ == '__main__':
    main()
