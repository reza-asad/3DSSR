import os
import argparse
import shutil

import numpy as np
import pandas as pd
import trimesh
from PIL import Image

from scripts.helper import load_from_json
from scripts.render_scene_functions import render_single_pc


def create_img_table(table_dir, html_file_name='img_table.html'):
    def insert_into_table(file, img_path, caption=None):
        file.write('\n')
        file.write('<td align="center" valign="center">\n')
        file.write('<img src="{}" />\n'.format(img_path))
        file.write('<br />\n')
        # add caption
        file.write('<br />\n')
        if caption is not None:
            file.write(caption)
            file.write('<br />\n')
        file.write('</td>\n')
        file.write('\n')

    html_file_path = os.path.join(table_dir, html_file_name)
    with open(html_file_path, 'w+') as f:
        # add the table
        f.write('<table width="500" border="0" cellpadding="5">\n')

        # for each scene insert its image+bbox and the pc inside the box
        img_dir = os.path.join(table_dir, 'imgs')
        img_names = [e for e in os.listdir(img_dir) if 'obj' in e]
        img_names = sorted([(int(e.split('-')[1]), e) for e in img_names])
        for img_name in img_names:
            # insert the region into the table
            f.write('<tr>\n')
            img_path_obj = os.path.join('imgs', img_name[1])
            insert_into_table(f, img_path_obj)
            img_path_scene_obj = img_path_obj.replace('obj', 'scene')
            insert_into_table(f, img_path_scene_obj)
            img_path_scene_raw = img_path_obj.replace('obj', 'scene_raw')
            insert_into_table(f, img_path_scene_raw)
            f.write('</tr>\n')

        # end the table
        f.write('</table>\n')


def build_colored_pc(pc):
    radii = np.linalg.norm(pc, axis=1)
    colors = trimesh.visual.interpolate(radii, color_map='viridis')
    pc_vis = trimesh.points.PointCloud(pc, colors=colors)

    return pc_vis


def find_examples_per_cat(args, df_metadata, accepted_cats):
    examples_per_cat = {}
    for cat in accepted_cats:
        df_sampled = df_metadata.loc[df_metadata['mpcat40'] == cat]
        if len(df_sampled) < args.num_examples:
            print('Not enough examples for {}'.format(cat))
            continue
        df_sampled = df_sampled.sample(n=args.num_examples, random_state=1)
        examples_per_cat[cat] = df_sampled.copy()

    return examples_per_cat


def render_detection_input(args, examples_per_cat):
    # for each category render images representing the scene+box and the mesh inside box.
    for cat, df in examples_per_cat.items():
        # create the rendering folder for that category.
        rendering_path = os.path.join(args.results_dir, cat, 'imgs')
        if not os.path.exists(rendering_path):
            os.makedirs(rendering_path)
        print(cat)
        keys = df.index.values
        for i, key in enumerate(keys):
            # find the id of the current obj.
            if args.use_nyu40:
                curr_id = df.loc[key, 'NYU40']
            else:
                curr_id = args.cat2class[cat]

            # load the bbox from the prepared detection data.
            scene_name = df.loc[key, 'room_name']
            bbox_file_name = scene_name + '_bbox.npy'
            bbox_path = os.path.join(args.detection_data_dir, bbox_file_name)
            scene_boxes = np.load(bbox_path)
            for j, id_ in enumerate(scene_boxes[:, -1]):
                if int(id_) == curr_id:
                    bbox = scene_boxes[j, :-1]
                    break

            # convert the bbox to a format that can be rendered.
            transformation = np.eye(4)
            transformation[:3, 3] = bbox[:3]
            bbox_vis = trimesh.creation.box(bbox[3:], transform=transformation)
            bbox_vis.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#0000FF")

            # create the scene + bbox rendering.
            scene_vertices_file_name = scene_name + '_vert.npy'
            scene_vertices = np.load(os.path.join(args.detection_data_dir, scene_vertices_file_name))
            # scene_pc = build_colored_pc(scene_vertices[:, :3])
            # scene_pc_box = trimesh.Scene([scene_pc, bbox_vis])
            # scene_pc_box.show()

            # render the scene
            img = render_single_pc(scene_vertices[:, :3], resolution, rendering_kwargs, with_obbox=False)
            img_path = os.path.join(rendering_path, '{}-{}-scene_raw.png'.format(scene_name, i+1))
            Image.fromarray(img).save(img_path)

            # render the scene + bbox
            img = render_single_pc(scene_vertices[:, :3], resolution, rendering_kwargs, with_obbox=True, obbox=bbox_vis)
            img_path = os.path.join(rendering_path, '{}-{}-scene.png'.format(scene_name, i+1))
            Image.fromarray(img).save(img_path)

            # find the pc inside the box for rendering
            scale = np.expand_dims(bbox[3:], axis=0)
            is_inside = np.abs(scene_vertices[:, :3] - bbox[:3]) < scale
            is_inside = np.sum(is_inside, axis=1) == 3
            # obj_pc = build_colored_pc(scene_vertices[is_inside, :3])
            # obj_pc.show()

            # render the pc inside the box.
            img = render_single_pc(scene_vertices[is_inside, :3], resolution, rendering_kwargs, with_obbox=False)
            img_path = os.path.join(rendering_path, '{}-{}-obj.png'.format(scene_name, i+1))
            Image.fromarray(img).save(img_path)


def get_args():
    parser = argparse.ArgumentParser('Rendering Detection Data', add_help=False)

    # path parameters
    parser.add_argument('--action', default='img_table', help='render | img_table')
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--metadata_path', default='../data/{}/metadata_non_equal_full_top10.csv')
    parser.add_argument('--accepted_cats_path', default='../data/{}/accepted_cats_top10.json')
    parser.add_argument('--scene_dir', default='../data/{}/scenes')
    parser.add_argument('--results_dir', default='/media/reza/Large/{}')
    parser.add_argument('--results_folder_name', default='detection_input_rendered')
    parser.add_argument('--num_examples', default=15, type=int)
    parser.add_argument('--colormap_path', default='../data/{}/color_map.json')
    parser.add_argument('--use_nyu40', action='store_true', default=False)

    return parser


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def main():
    # get the arguments
    parser = argparse.ArgumentParser('Rendering MultiCrop', parents=[get_args()])
    args = parser.parse_args()
    adjust_paths(args, exceptions=[])
    dataset_name = args.dataset if '3d' not in args.dataset else args.dataset[:-2]
    args.detection_data_dir = os.path.join(args.results_dir, '{}_train_detection_data'.format(dataset_name))
    args.scene_dir = os.path.join(args.scene_dir, 'all')
    args.colormap = load_from_json(args.colormap_path)
    args.results_dir = os.path.join(args.results_dir, args.results_folder_name)

    # load metadata and accepted cats.
    accepted_cats = load_from_json(args.accepted_cats_path)
    df_metadata = pd.read_csv(args.metadata_path)
    df_metadata = df_metadata.loc[(df_metadata['split'] == 'train') | (df_metadata['split'] == 'val')]
    if not args.use_nyu40:
        cats = np.unique(df_metadata['mpcat40'].values)
        args.cat2class = dict(zip(sorted(cats), np.arange(len(cats))))

    # index the metadata by key for fast retrieval.
    df_metadata['key'] = df_metadata.apply(lambda x: '-'.join([x['room_name'], str(x['objectId'])]), axis=1)
    df_metadata.set_index('key', inplace=True)

    # create a results folder name and the results directory
    if args.action == 'render':
        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
        else:
            shutil.rmtree(args.results_dir)

        # find a number of examples per category.
        examples_per_cat = find_examples_per_cat(args, df_metadata, accepted_cats)

        # render input detection data.
        render_detection_input(args, examples_per_cat)
    elif args.action == 'img_table':
        cats = os.listdir(args.results_dir)
        for cat in cats:
            table_dir = os.path.join(args.results_dir, cat)
            create_img_table(table_dir, )
    else:
        raise NotImplementedError('Action not recognized')


if __name__ == '__main__':
    # set up rendering parameters
    resolution = (512, 512)
    rendering_kwargs = {'fov': np.pi / 4, 'light_directional_intensity': 0.01, 'light_point_intensity_center': 0.0,
                        'wall_thickness': 5}
    main()
