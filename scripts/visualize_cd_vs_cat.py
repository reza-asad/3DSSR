import os
import argparse
import shutil
import pandas as pd
import numpy as np
import trimesh
from PIL import Image
import torch
from chamferdist import ChamferDistance

from scripts.helper import load_from_json, write_to_json, create_img_table_scrollable
from render_scene_functions import render_single_mesh


def chamfer_distance(args, file_name_q, file_name_t):
    # load the point clouds.
    pc_q = np.load(os.path.join(args.pc_dir, file_name_q))
    pc_t = np.load(os.path.join(args.pc_dir, file_name_t))

    # sample N points
    np.random.seed(0)
    sampled_indices = np.random.choice(range(len(pc_q)), args.num_points, replace=False)
    pc_q = np.expand_dims(pc_q[sampled_indices, :], axis=0)
    pc_q = torch.from_numpy(pc_q).cuda()
    np.random.seed(0)
    sampled_indices = np.random.choice(range(len(pc_t)), args.num_points, replace=False)
    pc_t = np.expand_dims(pc_t[sampled_indices, :], axis=0)
    pc_t = torch.from_numpy(pc_t).cuda()

    # compute chamfer distance.
    chamferDist = ChamferDistance()
    if args.bidirectional:
        dist_bidirectional = chamferDist(pc_q, pc_t, bidirectional=True)
        dist = dist_bidirectional.detach().cpu().item()
    else:
        dist_forward = chamferDist(pc_q, pc_t)
        dist = dist_forward.detach().cpu().item()

    return dist


def sample_query_objects(args, df_metadata, query_dict):
    all_query_objects = {}
    sampled_query_objects = {}
    for query, query_info in query_dict.items():
        scene_name = query_info['example']['scene_name'].split('.')[0]
        q = query_info['example']['query']
        q_context = query_info['example']['context_objects'] + [q]
        # add the objects for each category
        for obj in q_context:
            # find the category
            key = '{}-{}'.format(scene_name, obj)
            obj_cat = df_metadata.loc[key, 'mpcat40']
            if obj_cat not in all_query_objects:
                all_query_objects[obj_cat] = [key]
            else:
                all_query_objects[obj_cat].append(key)

    # sample one query object per category.
    for obj_cat, obj_list in all_query_objects.items():
        np.random.seed(args.seed)
        sampled_query_objects[obj_cat] = np.random.choice(obj_list, 1).tolist()[0]

    return sampled_query_objects


def find_matching_target_objects(args, query_objects):
    # examine all target objects to see if they satisfy the CD threshold but have different category than the query obj.
    all_target_objects = {}
    scene_names = os.listdir(args.scene_dir)
    for i, scene_name in enumerate(scene_names):
        print('Processing scene {}/{}'.format(i+1, len(scene_names)))

        # load the scene.
        scene = load_from_json(os.path.join(args.scene_dir, scene_name))

        # examine if each object is a candidate CD match for each query object but categories are different.
        for t, t_info in scene.items():
            t_cat = t_info['category'][0]
            file_name_t = '{}-{}.npy'.format(scene_name.split('.')[0], t)

            # test each sampled query object.
            for q_cat, file_name_q in query_objects.items():
                # ensure the target and query scenes are different.
                q_scene_name = file_name_q.split('-')[0]
                t_scene_name = file_name_t.split('-')[0]
                if t_scene_name == q_scene_name:
                    continue

                # ensure categories are different.
                if q_cat != t_cat:
                    file_name_q = file_name_q + '.npy'
                    cd_dist = chamfer_distance(args, file_name_q, file_name_t)
                    cd_threshold = args.cd_thresholds[q_cat][args.cd_threshold]
                    if cd_dist < cd_threshold:
                        if q_cat not in all_target_objects:
                            all_target_objects[q_cat] = [(file_name_t.split('.')[0], cd_dist, cd_threshold, t_cat)]
                        else:
                            all_target_objects[q_cat].append((file_name_t.split('.')[0], cd_dist, cd_threshold, t_cat))

    return all_target_objects


def render_query_targets(args, query_objects, target_objects):
    # render the query and target objects.
    for q_cat, file_name_q in query_objects.items():
        # create a directory fot the category
        imgs_path = os.path.join(args.rendering_path, q_cat, 'imgs')
        if not os.path.exists(imgs_path):
            os.makedirs(imgs_path)

        # load the mesh_region corresponding to the query
        mesh = trimesh.load(os.path.join(args.mesh_dir, file_name_q+'.ply'))

        # render the query img.
        query_img = render_single_mesh(mesh, resolution, rendering_kwargs, region=True)
        query_img = Image.fromarray(query_img)

        # save the result.
        q_img_path = os.path.join(imgs_path, 'query_{}.png'.format(file_name_q))
        query_img.save(q_img_path)

        # render the target images.
        if q_cat in target_objects:
            t_list = target_objects[q_cat]

            # render sampled target objects for each query object.
            for (file_name_t, _, _, _) in t_list:
                # load the mesh_region corresponding to the target
                mesh = trimesh.load(os.path.join(args.mesh_dir, file_name_t + '.ply'))

                # render the target image.
                target_img = render_single_mesh(mesh, resolution, rendering_kwargs, region=True)
                target_img = Image.fromarray(target_img)

                # save the result.
                t_img_path = os.path.join(imgs_path, file_name_t+'.png')
                target_img.save(t_img_path)


def build_img_table(args, query_objects, sampled_target_objects):
    for q_cat, file_name_q in query_objects.items():
        # find the query img name
        query_img = 'query_{}.png'.format(file_name_q)
        imgs_path = os.path.join(args.rendering_path, q_cat, 'imgs')

        # build the caption for the query.
        query_caption = '{}'.format(q_cat)

        # build the caption for the target images.
        img_names, captions = [], []
        if q_cat in sampled_target_objects:
            t_list = sampled_target_objects[q_cat]
            for (file_name_t, cd_dist, cd_threshold, t_cat) in t_list:
                caption = '<br />\n'
                caption += '{} <br />\n'.format(t_cat)
                caption += 'CD_dist: {} <br />\n'.format(np.round(cd_dist, 2))
                caption += 'CD_threshold: {} <br />\n'.format(np.round(cd_threshold, 2))
                captions.append(caption)
                img_names.append(file_name_t+'.png')

            # create img table for the category
            create_img_table_scrollable(imgs_path, 'imgs', img_names, query_img=query_img, html_file_name='img_table.html',
                                        topk=len(img_names), ncols=2, captions=captions, query_caption=query_caption)


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def get_args():
    parser = argparse.ArgumentParser('CAT VS CD', add_help=False)
    parser.add_argument('--action', default='img_table', help='extract | render | img_table')
    parser.add_argument('--bidirectional', action='store_true', default=True,
                        help='If True distance is computed in both directions and added.')
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--mode', dest='mode', default='test', help='val or test')
    parser.add_argument('--metadata_path', default='../data/{}/metadata_non_equal_full_top10.csv')
    parser.add_argument('--query_dir', default='../queries/{}/')
    parser.add_argument('--query_input_file_name', default='query_dict_top10.json')
    parser.add_argument('--scene_dir', default='../results/{}/scenes_top10')
    parser.add_argument('--mesh_dir', default='../data/{}/mesh_regions')
    parser.add_argument('--pc_dir', default='../data/{}/pc_regions')
    parser.add_argument('--cd_path', default='../data/{}/cd_thresholds.json')
    parser.add_argument('--results_dir', default='../results/{}/CatVsCDBiDirectional')
    parser.add_argument('--num_points', default=4096, type=int, help='number of points randomly sampled form the pc.')
    parser.add_argument('--cd_threshold', default='20', help='percentage threshold for CD similarity')
    parser.add_argument('--seed', default=0, type=int, help='seed for sampling query and target objects.')
    parser.add_argument('--topk', default=50, type=int, help='number of target objects sampled per query object.')

    return parser


def main():
    # read the args
    parser = argparse.ArgumentParser('CAT VS CD', parents=[get_args()])
    args = parser.parse_args()
    adjust_paths(args, exceptions=[])
    args.query_dir = os.path.join(args.query_dir, args.mode)
    args.scene_dir = os.path.join(args.scene_dir, args.mode)
    args.pc_dir = os.path.join(args.pc_dir, args.mode)
    args.mesh_dir = os.path.join(args.mesh_dir, args.mode)
    cd_path = args.cd_path.split('.json')[0]
    if args.bidirectional:
        cd_path += '_{}_{}.json'.format('bidirectional', args.mode)
    else:
        cd_path += '_{}.json'.format(args.mode)
    args.cd_thresholds = load_from_json(cd_path)
    args.rendering_path = os.path.join(args.results_dir, 'rendered_results_{}'.format(args.cd_threshold))

    # load the metadata and index it by obj keys.
    df_metadata = pd.read_csv(args.metadata_path)
    df_metadata = df_metadata[df_metadata['split'] == args.mode]
    df_metadata['key'] = df_metadata.apply(lambda x: '-'.join([x['room_name'], str(x['objectId'])]), axis=1)
    df_metadata.set_index('key', inplace=True)

    # load the query dict.
    query_dict_input_path = os.path.join(args.query_dir, args.query_input_file_name)
    query_dict = load_from_json(query_dict_input_path)

    # remove the rendering path if action is render and the folder already exists.
    if args.action == 'render':
        if os.path.exists(args.rendering_path):
            shutil.rmtree(args.rendering_path)

    # create the rendering path
    if not os.path.exists(args.rendering_path):
        os.makedirs(args.rendering_path)

    if args.action == 'extract':
        # For each category sample a query object.
        query_objects = sample_query_objects(args, df_metadata, query_dict)

        # for each query object sample target objects with CD passing threshold and cat being different.
        target_objects = find_matching_target_objects(args, query_objects)

        # save the query and target objects.
        write_to_json(query_objects, os.path.join(args.results_dir, 'sampled_query_objects_{}.json'.format(args.cd_threshold)))
        write_to_json(target_objects, os.path.join(args.results_dir, 'matching_target_objects_{}.json'.format(args.cd_threshold)))
    else:
        # load the sampled query and target objects.
        query_objects = load_from_json(os.path.join(args.results_dir, 'sampled_query_objects_{}.json'.format(args.cd_threshold)))
        target_objects = load_from_json(os.path.join(args.results_dir, 'matching_target_objects_{}.json'.format(args.cd_threshold)))

        # sample target objects per category.
        sampled_target_objects = {}
        for obj_cat, obj_list in target_objects.items():
            np.random.seed(args.seed)
            np.random.shuffle(obj_list)
            sampled_target_objects[obj_cat] = obj_list[:args.topk]

        if args.action == 'render':
            render_query_targets(args, query_objects, sampled_target_objects)
        elif args.action == 'img_table':
            build_img_table(args, query_objects, sampled_target_objects)


if __name__ == '__main__':
    resolution = (512, 512)
    rendering_kwargs = {'fov': np.pi / 4, 'light_directional_intensity': 0.01, 'light_point_intensity_center': 0.0,
                        'wall_thickness': 5}

    main()
