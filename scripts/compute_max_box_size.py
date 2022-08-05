import os
import argparse
import numpy as np
import pandas as pd

from scripts.helper import load_from_json
import trimesh


def get_args():
    parser = argparse.ArgumentParser('Extract Regions', add_help=False)
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--metadata_path', dest='metadata_path', default='../data/{}/metadata.csv')
    parser.add_argument('--accepted_cats_path', default='../data/{}/accepted_cats.json')
    parser.add_argument('--mesh_dir', default='/media/reza/Large/matterport3d/rooms')
    parser.add_argument('--box_type', default='scene', help='scene | object')

    return parser


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def compute_max_obj_box_size():
    # load accepted cats.
    accepted_cats = load_from_json(os.path.join(args.accepted_cats_path))

    # filter metadata and create keys.
    df_metadata = pd.read_csv(args.metadata_path)
    df_metadata = df_metadata.loc[df_metadata['split'] == 'train']
    is_accepted = df_metadata['mpcat40'].apply(lambda x: x in accepted_cats)
    df_metadata = df_metadata.loc[is_accepted]
    df_metadata['key'] = df_metadata[['room_name', 'objectId']].apply(lambda x: '-'.join([str(x[0]), str(x[1]) +
                                                                                          '.ply']), axis=1)

    # find mean box size.
    max_dims = []
    args.mesh_region_dir = os.path.join(args.mesh_region_dir, 'train')
    for region_name in df_metadata['key'].values:
        # load the region and compute the extent
        mesh = trimesh.load(os.path.join(args.mesh_region_dir, region_name))
        curr_max_dim = np.max(mesh.extents)
        max_dims.append(curr_max_dim)

    max_dims = np.asarray(sorted(max_dims, reverse=True))
    print('Max scale is" {}'.format(np.quantile(max_dims / np.max(max_dims), 0.95) * np.max(max_dims)))


def compute_max_scene_box_size():
    scene_names = os.listdir(args.mesh_dir)
    max_dims = []
    for scene_name in scene_names:
        if len(os.listdir(os.path.join(args.mesh_dir, scene_name))) == 0:
            continue

        # load the mesh
        mesh = trimesh.load(os.path.join(args.mesh_dir, scene_name, '{}.annotated.ply'.format(scene_name)))
        curr_max_dim = np.max(mesh.extents)
        max_dims.append(curr_max_dim)

    max_dims = np.asarray(sorted(max_dims, reverse=True))
    print('Max scale is" {}'.format(np.quantile(max_dims / np.max(max_dims), 0.95) * np.max(max_dims)))


def main():
    if args.box_type == 'object':
        compute_max_obj_box_size()
    elif args.box_type == 'scene':
        compute_max_scene_box_size()
    else:
        raise NotImplemented('The box type {} is not implemented.'.format(args.box_type))


if __name__ == '__main__':
    # read the args
    parser_ = argparse.ArgumentParser('Extract Regions', parents=[get_args()])
    args = parser_.parse_args()
    adjust_paths(args, exceptions=[])

    main()
