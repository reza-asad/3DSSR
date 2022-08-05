import os
import argparse
import pandas as pd
import numpy as np
import torch
from chamferdist import ChamferDistance

from scripts.helper import load_from_json, write_to_json


def compute_cd_thresholds(args, q_file_name, file_names, cd_thresholds):
    # load the query pc.
    pc_q = np.load(os.path.join(args.pc_dir, q_file_name))

    # sample N points
    np.random.seed(0)
    sampled_indices = np.random.choice(range(len(pc_q)), args.num_points, replace=False)
    pc_q = np.expand_dims(pc_q[sampled_indices, :], axis=0)
    pc_q = torch.from_numpy(pc_q).cuda()

    distances = []
    chamferDist = ChamferDistance()
    for i in range(len(file_names)):
        # load the pc.
        pc = np.load(os.path.join(args.pc_dir, file_names[i]))

        # sample N points
        np.random.seed(0)
        sampled_indices = np.random.choice(range(len(pc)), args.num_points, replace=False)
        pc = np.expand_dims(pc[sampled_indices, :], axis=0)
        pc = torch.from_numpy(pc).cuda()

        # compute chamfer distance.
        dist = chamferDist(pc_q, pc, bidirectional=True).item()
        distances.append(dist)

    # sort the distances and pick similarity thresholds at each cut-off percentage.
    cd_thresholds[q_file_name] = {threshold: None for threshold in args.thresholds}
    distances = sorted(distances)
    for threshold in args.thresholds:
        idx = int(np.round(len(distances) * threshold / 100))
        cd_thresholds[q_file_name][threshold] = distances[idx-1]


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def get_args():
    parser = argparse.ArgumentParser('Finding CD thresholds', add_help=False)
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--mode', dest='mode', default='val', help='val | test')
    parser.add_argument('--accepted_cats_path', default='../../data/{}/accepted_cats.json')
    parser.add_argument('--metadata_path', default='../../data/{}/metadata.csv')
    parser.add_argument('--pc_dir', default='../../data/{}/pc_objects')
    parser.add_argument('--cd_path', default='../../data/{}/cd_norm_constant.json')
    parser.add_argument('--query_dir', default='../../queries/{}/')
    parser.add_argument('--query_input_file_name', default='query_dict_top10.json')
    parser.add_argument('--num_points', default=4096, type=int, help='number of points randomly sampled form the pc.')
    parser.add_argument('--thresholds', default=[5, 10, 20, 50], nargs='+', type=int,
                        help='percentage thresholds for CD distance')
    return parser


def main():
    # read the args
    parser = argparse.ArgumentParser('Pre-compute DF', parents=[get_args()])
    args = parser.parse_args()
    adjust_paths(args, exceptions=[])
    args.pc_dir = os.path.join(args.pc_dir, args.mode)
    args.query_dir = os.path.join(args.query_dir, args.mode)
    args.cd_path = args.cd_path.split('.json')[0] + '_{}.json'.format(args.mode)

    # load accepted categories and the metadata.
    accepted_cats = load_from_json(args.accepted_cats_path)
    df_metadata = pd.read_csv(args.metadata_path)
    df_metadata = df_metadata[df_metadata['split'] == args.mode]
    is_accepted = df_metadata['mpcat40'].apply(lambda x: x in accepted_cats)
    df_metadata = df_metadata[is_accepted]
    df_metadata['key'] = df_metadata[['room_name', 'objectId']].apply(lambda x: '-'.join([str(x[0]), str(x[1])]) + '.npy', axis=1)

    # load the query dict.
    query_dict_input_path = os.path.join(args.query_dir, args.query_input_file_name)
    query_dict = load_from_json(query_dict_input_path)

    # for each query category find the CD distance threshold for that category at various percentages.
    cd_thresholds = {}
    for i, (query, query_info) in enumerate(query_dict.items()):
        print('*'*50)
        print('Processing Query {}/{}'.format(i+1, len(query_dict)))
        query_objects = [query_info['example']['query']] + query_info['example']['context_objects']
        query_scene_name = query_info['example']['scene_name']
        for j, query_obj in enumerate(query_objects):
            print('Processing Object {}/{}'.format(j + 1, len(query_objects)))

            # find the category of the query object.
            q_file_name = '{}-{}.npy'.format(query_scene_name.split('.')[0], query_obj)
            if q_file_name in cd_thresholds:
                continue
            q_cat = df_metadata.loc[df_metadata['key'] == q_file_name, 'mpcat40'].values[0]

            # find all file_names with the same category as the query object.
            file_names = df_metadata.loc[df_metadata['mpcat40'] == q_cat, 'key'].values.tolist()

            # compute chamfer distance thresholds at various percentages.
            compute_cd_thresholds(args, q_file_name, file_names, cd_thresholds)

    # save the CD similarity thresholds.
    write_to_json(cd_thresholds, args.cd_path)


if __name__ == '__main__':
    main()
