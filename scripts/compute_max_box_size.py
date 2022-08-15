import os
import argparse
import numpy as np
import pandas as pd

from scripts.helper import load_from_json
import trimesh


def compute_max_obj_box_size():
    # filter metadata and create keys.
    df_metadata = pd.read_csv(args.metadata_path)
    df_metadata = df_metadata.loc[df_metadata['split'] == 'train']
    df_metadata['key'] = df_metadata[['room_name', 'objectId']].apply(lambda x: '-'.join([str(x[0]), str(x[1]) +
                                                                                          '.npy']), axis=1)

    # find mean box size.
    max_dims = []
    args.pc_dir = os.path.join(args.pc_dir, 'train')
    for region_name in df_metadata['key'].values:
        # load the region and compute the extent
        pc = trimesh.points.PointCloud(np.load(os.path.join(args.pc_dir, region_name)))
        curr_max_dim = np.max(pc.extents)
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


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def get_args():
    parser = argparse.ArgumentParser('Extract Regions', add_help=False)
    parser.add_argument('--dataset', default='scannet')
    parser.add_argument('--metadata_path', dest='metadata_path', default='../data/{}/metadata.csv')
    parser.add_argument('--pc_dir', default='../data/{}/pc_regions')
    parser.add_argument('--box_type', default='object', help='scene | object')

    return parser


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
