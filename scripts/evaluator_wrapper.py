import os
import argparse
from time import time
import pandas as pd

from scripts.evaluator import evaluate_3dssr
from scripts.evaluator_full_3dssr import evaluate_full_3dssr
from scripts.helper import load_from_json


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def get_args():
    parser = argparse.ArgumentParser('Evaluating 3D Subscene Retrieval', add_help=False)
    parser.add_argument('--model_names', default=['full_3dssr'],
                        type=str, nargs='+', help='choose one from 3dssr_model_configs.json')
    parser.add_argument('--cat_threshold', default=None, help='Threshold for categorizing the predicted boxes or None.')
    parser.add_argument('--bidirectional', action='store_true', default=True,
                        help='If True distance is computed in both directions and added.')
    parser.add_argument('--mode', default='test', help='val | test')
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--scene_dir_queries', default='../results/{}/scenes')
    parser.add_argument('--scene_dir', default='../results/{}/scenes',
                        help='scenes_top10 | predicted_boxes_large/scenes_predicted_nms_final')
    parser.add_argument('--query_input_file_name', default='query_dict_top10.json')
    parser.add_argument('--pc_dir_queries', default='../data/{}/pc_regions')
    parser.add_argument('--pc_dir', default='../data/{}/pc_regions_predicted',
                        help='pc_regions | pc_regions_predicted')
    parser.add_argument('--cd_path', default='../data/{}/cd_thresholds.json')
    parser.add_argument('--num_points', default=4096, type=int, help='number of points randomly sampled form the pc.')
    parser.add_argument('--remove_model', action='store_true', default=False,
                        help='If True the model and its corresponding experiment are removed from the evaluation table.')
    parser.add_argument('--ablations', action='store_true', default=False,
                        help='If True the evaluation results are stored in the ablation folder.')
    parser.add_argument('--topk', default=10, type=int, help='number of top results for mAP computations.')
    parser.add_argument('--model_config_filename', default='3dssr_model_configs.json')
    parser.add_argument('--metadata_path', default='../data/{}/metadata.csv')
    parser.add_argument('--fine_cat_field', default=None, help='wnsynsetkey | raw_category.')

    return parser


def main():
    # read the args
    parser = argparse.ArgumentParser('Evaluating 3D Subscene Retrieval', parents=[get_args()])
    args = parser.parse_args()
    adjust_paths(args, exceptions=[])

    # set the correct mode for required fields
    args.pc_dir = os.path.join(args.pc_dir, args.mode)
    args.pc_dir_queries = os.path.join(args.pc_dir_queries, args.mode)
    args.cd_path = args.cd_path.split('.json')[0]
    if args.bidirectional:
        args.cd_path += '_{}_{}.json'.format('bidirectional', args.mode)
    else:
        args.cd_path += '_{}.json'.format(args.mode)

    # load the metadata and index by key.
    df_metadata = pd.read_csv(args.metadata_path)
    df_metadata['key'] = df_metadata.apply(lambda x: '-'.join([x['room_name'], str(x['objectId'])]), axis=1)
    df_metadata.set_index('key', inplace=True)
    args.df_metadata = df_metadata

    # determine the correct config for the pretraining strategy
    for model_name in args.model_names:
        config = load_from_json(args.model_config_filename)[model_name]

        # add the pretraining configs and apply 3dssr and adjust paths.
        for k, v in config.items():
            vars(args)[k] = v
        adjust_paths(args, exceptions=[])

        # evaluate 3dssr
        if model_name == 'full_3dssr':
            evaluate_full_3dssr(args)
        else:
            evaluate_3dssr(args)


if __name__ == '__main__':
    t = time()
    main()
    duration = time() - t
    print('Evaluation took {} minutes'.format(round(duration / 60, 2)))


