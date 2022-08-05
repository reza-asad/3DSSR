import os
import argparse
import pandas as pd
import numpy as np
import torch
from chamferdist import ChamferDistance

from scripts.helper import load_from_json, write_to_json


def compute_cd_thresholds(args, file_names, cd_thresholds, cat):
    # compute all pair-wise distances.
    distances = []
    chamferDist = ChamferDistance()
    for i in range(len(file_names)):
        for j in range(i+1, len(file_names)):
            # load the point clouds.
            file_name1 = file_names[i]
            file_name2 = file_names[j]
            pc1 = np.load(os.path.join(args.pc_dir, file_name1))
            pc2 = np.load(os.path.join(args.pc_dir, file_name2))

            # sample N points
            np.random.seed(0)
            sampled_indices = np.random.choice(range(len(pc1)), args.num_points, replace=False)
            pc1 = np.expand_dims(pc1[sampled_indices, :], axis=0)
            pc1 = torch.from_numpy(pc1).cuda()
            np.random.seed(0)
            sampled_indices = np.random.choice(range(len(pc2)), args.num_points, replace=False)
            pc2 = np.expand_dims(pc2[sampled_indices, :], axis=0)
            pc2 = torch.from_numpy(pc2).cuda()

            # compute chamfer distance.
            if args.bidirectional:
                dist_bidirectional = chamferDist(pc1, pc2, bidirectional=True)
                dist = dist_bidirectional.detach().cpu().item()
            else:
                dist_forward = chamferDist(pc1, pc2)
                dist = dist_forward.detach().cpu().item()

            distances.append(dist)

    # sort the distances and pick similarity thresholds at each cut-off percentage.
    cd_thresholds[cat] = {threshold: None for threshold in args.thresholds}
    distances = sorted(distances)
    for threshold in args.thresholds:
        idx = int(np.round(len(distances) * threshold / 100))
        cd_thresholds[cat][threshold] = distances[idx-1]


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def get_args():
    parser = argparse.ArgumentParser('Finding CD thresholds', add_help=False)
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--mode', dest='mode', default='test', help='val or test')
    parser.add_argument('--accepted_cats_path', default='../data/{}/accepted_cats_top10.json')
    parser.add_argument('--metadata_path', default='../data/{}/metadata_non_equal_full_top10.csv')
    parser.add_argument('--pc_dir', default='../data/{}/pc_regions')
    parser.add_argument('--cd_path', default='../data/{}/cd_thresholds_bidirectional.json')
    parser.add_argument('--bidirectional', action='store_true', default=True,
                        help='If True distance is computed in both directions and added.')
    parser.add_argument('--num_points', default=4096, type=int, help='number of points randomly sampled form the pc.')
    parser.add_argument('--thresholds', default=[5, 10, 20, 40], nargs='+', type=int,
                        help='percentage thresholds for CD similarity')
    return parser


def main():
    # read the args
    parser = argparse.ArgumentParser('Pre-compute DF', parents=[get_args()])
    args = parser.parse_args()
    adjust_paths(args, exceptions=[])
    args.pc_dir = os.path.join(args.pc_dir, args.mode)
    args.cd_path = args.cd_path.split('.json')[0] + '_{}.json'.format(args.mode)

    # load accepted categories and the metadata.
    accepted_cats = load_from_json(args.accepted_cats_path)
    df_metadata = pd.read_csv(args.metadata_path)
    df_metadata = df_metadata[df_metadata['split'] == args.mode]

    # for each accepted category find the CD similarity threshold at various percentages.
    cd_thresholds = {}
    for i, cat in enumerate(accepted_cats):
        print('Iteration {}/{} for Category {}'.format(i+1, len(accepted_cats), cat))

        # find all file_names with cat category.
        df_metadata_cat = df_metadata.loc[df_metadata['mpcat40'] == cat]
        file_names = df_metadata_cat[['room_name', 'objectId']].apply(lambda x: '-'.join([str(x[0]), str(x[1]) +
                                                                                          '.npy']), axis=1).tolist()

        # compute CD similarity thresholds at various percentages.
        compute_cd_thresholds(args, file_names, cd_thresholds, cat)

    # save the CD similarity thresholds.
    write_to_json(cd_thresholds, args.cd_path)


if __name__ == '__main__':
    main()
