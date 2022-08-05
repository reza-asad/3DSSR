import os
import argparse

import extract_rank_subscenes
from scripts.helper import load_from_json


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def get_args():
    parser = argparse.ArgumentParser('Extracting and Ranking 3D Subscenes', add_help=False)

    parser.add_argument('--model_name', default='supervised_pret', help='choose one from 3dssr_model_configs.json')
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--mode', dest='mode', default='val', help='val or test')
    parser.add_argument('--accepted_cats_path', default='../../data/{}/accepted_cats.json')
    parser.add_argument('--metadata_path', default='../../data/{}/metadata.csv')
    parser.add_argument('--scene_dir', default='../../results/{}/scenes')
    parser.add_argument('--query_dir', default='../../queries/{}/')
    parser.add_argument('--query_input_file_name', default='query_dict_top10.json')
    parser.add_argument('--predicted_labels_path', default='../../results/{}/LearningBased/'
                                                           'region_classification_transformer_full/features/'
                                                           'predicted_labels_knn.json')
    parser.add_argument('--model_config_filename', default='3dssr_model_configs.json')

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

    # apply 3dssr
    args.query_dir = os.path.join(args.query_dir, args.mode)
    # extract_rank_subscenes.apply_3dssr(args)


if __name__ == '__main__':
    main()
