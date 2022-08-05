import os
import argparse
import shutil

import pandas as pd
import numpy as np
import torch
import trimesh
from PIL import Image
from torch.utils.data import DataLoader

from subscene_dataset import SubScene
from subscene_dataset_no_centroid import SubScene as SubSceneNoCentroid
from scripts.helper import load_from_json
import models.LearningBased.utils as utils
from scripts.renderer import Render
from scripts.helper import create_img_table


def build_colored_pc_subscene(subscene_pc, translations=None):
    # for each pc in the subscene find its color and then translate to the scene's frame.
    num_points = subscene_pc.shape[1]
    subscene_colors = np.zeros((num_points * len(subscene_pc), 4), dtype=int)
    subscene_pc_transformed = np.zeros((num_points * len(subscene_pc), 3), dtype=np.float32)
    for i in range(len(subscene_pc)):
        # record the color in the local frame
        pc = subscene_pc[i, ...]
        radii = np.linalg.norm(pc, axis=1)
        colors = trimesh.visual.interpolate(radii, color_map='viridis')
        subscene_colors[i * num_points: (i+1) * num_points, :] = colors

        # translate to the global frame.
        if translations is not None:
            rotation = np.eye(3)
            subscene_pc_transformed[i * num_points: (i+1) * num_points, :] = transform_pc(subscene_pc[i, ...],
                                                                                          translations[i, ...],
                                                                                          rotation)
        else:
            subscene_pc_transformed[i * num_points: (i + 1) * num_points, :] = subscene_pc[i, ...]

    return subscene_pc_transformed, subscene_colors


def render_single_pc(r, pc, colors):
    # setup the camera pose and extract the dimensions of the room
    pc_vis = trimesh.Trimesh.scene(trimesh.points.PointCloud(pc, colors=colors))
    room_dimension = pc_vis.extents
    camera_pose, _ = pc_vis.graph[pc_vis.camera.name]

    # render the pc
    img, _ = r.pyrender_render(pc, resolution=resolution, camera_pose=camera_pose, room_dimension=room_dimension,
                               points=True, colors=colors, with_obbox=False, obbox=None, adjust_camera_height=False,
                               point_size=1.5, with_height_offset=False)

    return img


def transform_pc(pc, translation, rotation):
    transformation = np.eye(4)
    transformation[:3, 3] = translation
    transformation[:3, :3] = rotation

    new_pc = np.ones((len(pc), 4), dtype=np.float32)
    new_pc[:, 0] = pc[:, 0]
    new_pc[:, 1] = pc[:, 1]
    new_pc[:, 2] = pc[:, 2]

    new_pc = np.dot(transformation, new_pc.T).T
    new_pc = new_pc[:, :3]

    return new_pc


def render_subscene(args):
    # create dataset
    dataset = SubScene(scene_dir=args.scene_dir, pc_dir=args.pc_dir, metadata_path=args.metadata_path,
                       accepted_cats_path=args.accepted_cats_path, max_coord_box=args.max_coord_box,
                       max_coord_scene=args.max_coord_scene, num_points=args.num_point, num_objects=args.num_objects,
                       mode='train', file_name_to_idx=args.file_name_to_idx, with_transform=args.with_transform,
                       random_subscene=args.random_subscene, batch_one=False, global_frame=args.global_frame)

    # create the dataloader
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=True)

    # iterate through the data to collect all the 3D data that is going to be rendered.
    subscene_pcs, subscene_centroids = [], []
    for i, data in enumerate(loader):
        # load pc and centroid.
        pc = data['pc'].numpy()
        centroid = data['centroid'].numpy()

        # denormalize the data and record it.
        pc = pc * args.max_coord_box
        centroid = centroid * args.max_coord_scene
        subscene_pcs.append(pc)
        subscene_centroids.append(centroid)

        if (i+1) >= args.topk:
            break
    subscene_pcs = np.concatenate(subscene_pcs, axis=0)
    subscene_centroids = np.concatenate(subscene_centroids, axis=0)

    # initialize the renderer.
    r = Render(rendering_kwargs)

    img_dir = os.path.join(args.results_dir, 'imgs')
    os.mkdir(img_dir)

    # render each subscene and save the images.
    for i, subscene_pc in enumerate(subscene_pcs):
        # translate the subscene according to the centroids.
        translations = subscene_centroids[i, ...]
        subscene_pc, colors = build_colored_pc_subscene(subscene_pc, translations)

        # render the subscene and save the image.
        img = render_single_pc(r, subscene_pc, colors)
        img = Image.fromarray(img)
        img_path = os.path.join(img_dir, 'img_{}.png'.format(i+1))
        img.save(img_path)

    # create image table.
    imgs = os.listdir(img_dir)
    imgs = sorted(imgs, key=lambda x: int(x.split('.')[0].split('_')[-1]))
    create_img_table(img_dir, 'imgs', imgs, 'img_table.html', topk=args.topk, ncols=3, captions=[],
                     with_query_scene=False, evaluation_plot=None, query_img=None, query_caption=None)


def render_local_subscene(args):
    # create dataset
    dataset = SubSceneNoCentroid(scene_dir=args.scene_dir, pc_dir=args.pc_dir, metadata_path=args.metadata_path,
                                 accepted_cats_path=args.accepted_cats_path, max_coord_scene=args.max_coord_scene,
                                 num_points=args.num_point, num_objects=args.num_objects, mode='train',
                                 file_name_to_idx=args.file_name_to_idx, with_transform=args.with_transform,
                                 random_subscene=args.random_subscene, batch_one=False)

    # create the dataloader
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=True)

    # iterate through the data to collect all the 3D data that is going to be rendered.
    subscene_pcs = []
    for i, data in enumerate(loader):
        # load pc and centroid.
        pc = data['pc'].numpy()

        # denormalize the data and record it.
        pc = pc * args.max_coord_scene
        subscene_pcs.append(pc)

        if (i+1) >= args.topk:
            break
    subscene_pcs = np.concatenate(subscene_pcs, axis=0)

    # initialize the renderer.
    r = Render(rendering_kwargs)

    img_dir = os.path.join(args.results_dir, 'imgs')
    os.mkdir(img_dir)

    # render each subscene and save the images.
    for i, subscene_pc in enumerate(subscene_pcs):
        # translate the subscene according to the centroids.
        subscene_pc, colors = build_colored_pc_subscene(subscene_pc, translations=None)

        # render the subscene and save the image.
        img = render_single_pc(r, subscene_pc, colors)
        img = Image.fromarray(img)
        img_path = os.path.join(img_dir, 'img_{}.png'.format(i+1))
        img.save(img_path)

    # create image table.
    imgs = os.listdir(img_dir)
    imgs = sorted(imgs, key=lambda x: int(x.split('.')[0].split('_')[-1]))
    create_img_table(img_dir, 'imgs', imgs, 'img_table.html', topk=args.topk, ncols=3, captions=[],
                     with_query_scene=False, evaluation_plot=None, query_img=None, query_caption=None)


def get_args():
    parser = argparse.ArgumentParser('Rendering MultiCrop', add_help=False)

    parser.add_argument('--rendering_type', default='local_subscenes', help='subscene | two_subscenes | local_subscenes')
    parser.add_argument('--with_transform', default=False, type=utils.bool_flag)
    parser.add_argument('--random_subscene', default=False, type=utils.bool_flag)
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--accepted_cats_path', default='../../data/{}/accepted_cats.json')
    parser.add_argument('--metadata_path',  default='../../data/{}/metadata.csv')
    parser.add_argument('--pc_dir', default='../../data/{}/pc_regions')
    parser.add_argument('--scene_dir', default='../../results/{}/scenes')
    parser.add_argument('--results_dir', default='../../results/{}/subscene_rendering/')
    parser.add_argument('--results_folder_name', default='exp_supervise_pret_aug_local_no_centroids')
    parser.add_argument('--num_objects', default=5, type=int)
    parser.add_argument('--num_point', default=4096, type=int)
    parser.add_argument('--max_coord_box', default=3.65, type=float, help='3.65 for MP3D')
    parser.add_argument('--max_coord_scene', default=13.07, type=float, help='13.07 for MP3D')
    parser.add_argument('--global_frame', default=True, type=utils.bool_flag)
    parser.add_argument('--topk', default=50, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    return parser


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def main():
    # get the arguments
    parser = argparse.ArgumentParser('Rendering Subscenes', parents=[get_args()])
    args = parser.parse_args()
    adjust_paths(args, exceptions=[])

    # create a results folder name and the results directory
    args.results_dir = os.path.join(args.results_dir, args.results_folder_name)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    else:
        shutil.rmtree(args.results_dir)
        os.makedirs(args.results_dir)

    # find a mapping from the region files to their indices.
    accepted_cats = load_from_json(args.accepted_cats_path)
    accepted_cats = sorted(accepted_cats)
    df = pd.read_csv(args.metadata_path)
    is_accepted = df['mpcat40'].apply(lambda x: x in accepted_cats)
    df = df.loc[is_accepted]
    df = df.loc[(df['split'] == 'train') | (df['split'] == 'val')]
    file_names = df[['room_name', 'objectId']].apply(lambda x: '-'.join([str(x[0]), str(x[1]) + '.npy']), axis=1).tolist()
    file_names = sorted(file_names)
    file_name_to_idx = {file_name: i for i, file_name in enumerate(file_names)}
    args.file_name_to_idx = file_name_to_idx

    if args.rendering_type == 'subscene':
        render_subscene(args)
    elif args.rendering_type == 'local_subscenes':
        render_local_subscene(args)


if __name__ == '__main__':
    # set up rendering parameters
    resolution = (512, 512)
    rendering_kwargs = {'fov': np.pi / 4, 'light_directional_intensity': 0.01, 'light_point_intensity_center': 0.0,
                        'wall_thickness': 5}
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    main()
