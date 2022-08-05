import os
import argparse
import numpy as np
import pandas as pd
import trimesh

from scripts.helper import load_from_json


def find_mean_scale(file_names):
    # load each mesh and record the scales.
    scales = []
    for file_name in file_names:
        mesh = trimesh.load(os.path.join(args.models_dir, file_name))
        scales.append([mesh.bounding_box.extents * args.expansion_factor])

    # take the mean scale.
    scales = np.concatenate(scales, axis=0)
    mean_scale = np.mean(scales, axis=0)

    return mean_scale


def get_args():
    parser = argparse.ArgumentParser('Extract Regions', add_help=False)
    parser.add_argument('--metadata_path', dest='metadata_path', default='../data/{}/metadata.csv')
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--accepted_cats_path', default='../data/{}/accepted_cats.json')
    parser.add_argument('--accepted_cats_10_path', default='../data/{}/accepted_cats_top10.json')
    parser.add_argument('--models_dir', default='../data/{}/models')
    parser.add_argument('--results_dir', default='../data/{}')
    parser.add_argument('--output_file_name', default='{}_means.npz')
    parser.add_argument('--expansion_factor', default=1.5, type=float)

    return parser


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def main():
    # load the all accepted cats and the top 10
    accepted_cats = load_from_json(os.path.join(args.accepted_cats_path))
    # accepted_cats_top10 = load_from_json(os.path.join(args.accepted_cats_10_path))
    # accepted_cats_top10 = sorted(list(accepted_cats_top10))

    # load the metadata and filter based on accepted cats and train data.
    df_metadata = pd.read_csv(args.metadata_path)
    df_metadata = df_metadata.loc[df_metadata['split'] == 'train']
    is_accepted = df_metadata['mpcat40'].apply(lambda x: x in accepted_cats)
    df_metadata = df_metadata.loc[is_accepted]
    df_metadata['key'] = df_metadata[['room_name', 'objectId']].apply(lambda x: '-'.join([str(x[0]), str(x[1]) +
                                                                                          '.ply']), axis=1)

    # mean_size_arr = np.zeros((len(accepted_cats_top10) + 1, 3), dtype=float)
    mean_size_arr = np.zeros((len(accepted_cats), 3), dtype=float)
    for i, cat in enumerate(accepted_cats):
        # find all object meshes with category cat
        file_names = df_metadata.loc[df_metadata['mpcat40'] == cat, 'key'].values
        mean_scale = find_mean_scale(file_names)
        mean_size_arr[i, :] = mean_scale
        print('Mean scale for {} is {}'.format(cat, mean_scale))

    # add the mean scale for the other category.
    # file_names = []
    # for cat in accepted_cats:
    #     if cat not in accepted_cats_top10:
    #         file_names += df_metadata.loc[df_metadata['mpcat40'] == cat, 'key'].values.tolist()
    # mean_scale = find_mean_scale(file_names)
    # mean_size_arr[10, :] = mean_scale
    # print('Mean scale for {} is {}'.format('other', mean_scale))

    # save the mean_size_arr
    output_path = os.path.join(args.results_dir, args.output_file_name.format(args.dataset).replace('3d', ''))
    np.savez(output_path, mean_size_arr)


if __name__ == '__main__':
    # read the args
    parser_ = argparse.ArgumentParser('Extract Regions', parents=[get_args()])
    args = parser_.parse_args()
    adjust_paths(args, exceptions=[])

    main()
