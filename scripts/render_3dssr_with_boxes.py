import argparse
import os
import gc
import trimesh
import numpy as np
from PIL import Image

from scripts.helper import load_from_json, render_scene_subscene, create_img_table_scrollable
from scripts.renderer import Render
from scripts.box import Box


def color_box(room_mesh, room_mesh_vertices, t_box):
    dist_per_dim = np.abs(room_mesh_vertices - t_box.translation)
    is_inside = dist_per_dim < (t_box.scale / 2)
    is_inside = np.sum(is_inside, axis=1) == 3

    # add opacity based on distance to the center of the box
    radii = np.linalg.norm(dist_per_dim[is_inside, ...], axis=1)
    colors = trimesh.visual.interpolate(radii, color_map='viridis')
    room_mesh_copy = room_mesh.copy()
    room_mesh_copy.visual.vertex_colors[is_inside, :] = colors

    return room_mesh_copy, is_inside


def render_query_img(img_dir, results_info):
    scene_name = results_info['example']['scene_name']
    query_graph = load_from_json(os.path.join(args.scene_dir_queries, scene_name))
    q = results_info['example']['query']
    q_context = set(results_info['example']['context_objects'] + [q])

    # render the image
    faded_nodes = [obj for obj in query_graph.keys() if obj not in q_context]

    img_path = os.path.join(img_dir, 'query_{}-{}.png'.format(scene_name.split('.')[0], q))
    render_scene_subscene(graph=query_graph, objects=query_graph.keys(), highlighted_object=[q],
                          faded_nodes=faded_nodes, path=img_path, model_dir=args.models_dir, colormap=args.colormap,
                          with_height_offset=False)


def render_scene_subscene_boxes(room_path, target_subscene, img_path, rendering_kwargs=None):
    # setup default rendering conditions such as lighting
    if rendering_kwargs is None:
        rendering_kwargs = {'fov': np.pi/4, 'light_directional_intensity': 0.01, 'light_point_intensity_center': 0.0,
                            'wall_thickness': 5}

    # render the entire scene first.
    imgs = []
    r = Render(rendering_kwargs)
    scene, camera_pose, room_dimension, room_mesh_copies = prepare_scene_for_rendering(room_path, target_subscene,
                                                                                       crop=True)
    img, _ = r.pyrender_render(scene, resolution=resolution, camera_pose=camera_pose,
                               room_dimension=room_dimension, with_height_offset=False)
    imgs.append(img)

    # render the subscenes, each showing a box.
    for room_mesh_copy in room_mesh_copies:
        scene = trimesh.Scene(room_mesh_copy)
        img, _ = r.pyrender_render(scene, resolution=resolution, camera_pose=camera_pose,
                                   room_dimension=room_dimension, with_height_offset=False)
        imgs.append(img)

    # put the scene image in the top left corner and the subscene in the bottom right corner.
    num_crop_imgs = len(imgs)
    width, height = img.shape[:-1]
    between_crop_space = 50
    new_img = Image.new('RGB', (width + between_crop_space, int(np.ceil(height / num_crop_imgs))), color=(255, 255, 255))

    # divide the image bottom corner of the new image to showcase each crop
    img_subscene_width = width // num_crop_imgs
    img_subscene_shape = (img_subscene_width, height // num_crop_imgs)
    width_offset = 0
    for img in imgs:
        img_subscene = Image.fromarray(img)
        img_subscene = img_subscene.resize(img_subscene_shape)
        new_img.paste(img_subscene, (width_offset, 0))
        width_offset += img_subscene_width
        width_offset += (between_crop_space // (num_crop_imgs - 1))

    new_img.save(img_path)


def prepare_scene_for_rendering(room_path, target_subscene, crop=False):
    default_color = '#aec7e8'
    cropped_scene = []
    room_mesh_copies = []

    # load the room mesh with default color.
    room_mesh = trimesh.load_mesh(room_path)
    room_mesh_all = room_mesh.copy()
    room_mesh.visual.vertex_colors = trimesh.visual.color.hex_to_rgba(default_color)
    room_mesh_all.visual.vertex_colors = trimesh.visual.color.hex_to_rgba(default_color)
    room_mesh_vertices = room_mesh.vertices

    # load the target scene and the target box
    scene_name = target_subscene['scene_name']
    target_scene = load_from_json(os.path.join(args.scene_dir, scene_name))
    target_obj = target_subscene['target']
    t_vertices = np.asarray(target_scene[target_obj]['aabb'], dtype=np.float64)
    t_box = Box(t_vertices)

    # color the target vertices
    room_mesh_all, _ = color_box(room_mesh_all, room_mesh_vertices, t_box)
    room_mesh_copy, is_inside = color_box(room_mesh, room_mesh_vertices, t_box)
    room_mesh_copies.append(room_mesh_copy)
    # room_mesh_copy.show()

    # add the subscene pc for rendering.
    if crop:
        pc = trimesh.points.PointCloud(room_mesh_vertices[is_inside, ...])
        cropped_scene.append(pc)

    # color the other corresponding objects.
    for candidate, _ in target_subscene['correspondence'].items():
        t_c_vertices = np.asarray(target_scene[candidate]['aabb'], dtype=np.float64)
        t_c_box = Box(t_c_vertices)

        # color the candidate vertices
        room_mesh_all, _ = color_box(room_mesh_all, room_mesh_vertices, t_c_box)
        room_mesh_copy, is_inside = color_box(room_mesh, room_mesh_vertices, t_c_box)
        room_mesh_copies.append(room_mesh_copy)
        # room_mesh_copy.show()

        # add the subscene pc for rendering.
        if crop:
            pc = trimesh.points.PointCloud(room_mesh_vertices[is_inside, ...])
            cropped_scene.append(pc)

    # extract the room dimension and the camera pose
    scene = trimesh.Scene(room_mesh_all)
    room_dimension = scene.extents
    camera_pose, _ = scene.graph[scene.camera.name]

    # if the scene is cropped, camera pose is rotated by theta and room dimension is extracted from the subscene.
    if crop:
        cropped_scene = trimesh.Scene(cropped_scene)
        room_dimension = cropped_scene.extents
        camera_pose, _ = cropped_scene.graph[cropped_scene.camera.name]

    del room_mesh
    gc.collect()

    return scene, camera_pose, room_dimension, room_mesh_copies


def main():
    # for each query render the top k results.
    for query, results_info in query_results.items():
        # if query == 'chair-45':
        #     print(results_info['target_subscenes'][0])

        if query in args.query_list or 'all' in args.query_list:
            # create the image dir
            img_dir = os.path.join(args.rendering_path, query, 'imgs')
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)

            if args.action == 'render':
                # render the query img.
                render_query_img(img_dir, results_info)

                # render the target images.
                target_subscenes = results_info['target_subscenes'][: args.topk]
                for i, target_subscene in enumerate(target_subscenes):
                    scene_name = target_subscene['scene_name'].split('.')[0]
                    t = target_subscene['target']
                    room_path = os.path.join(args.room_dir, scene_name, '{}.annotated.ply'.format(scene_name))
                    img_path = os.path.join(img_dir, 'top_{}_{}-{}.png'.format(i+1, scene_name, t))
                    render_scene_subscene_boxes(room_path, target_subscene, img_path)

            # create an image table for the query.
            imgs = os.listdir(img_dir)
            query_img = [img for img in imgs if 'query' in img][0]
            imgs.remove(query_img)
            imgs_and_ranks = [(img, int(img.split('.')[0].split('_')[1])) for img in imgs]
            imgs_and_ranks = sorted(imgs_and_ranks, key=lambda x: x[1])
            imgs = list(list(zip(*imgs_and_ranks))[0])
            create_img_table_scrollable(img_dir, 'imgs', imgs, 'img_table.html', query_img, ncols=1)


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def get_args():
    parser = argparse.ArgumentParser('Render Results', add_help=False)
    parser.add_argument('--ranking_strategy', default='dino_with_boxes_full_config', help='choose one from 3dssr_model_configs.json')
    parser.add_argument('--action', default='img_table', help='render | img_table')
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--mode', default='test', help='val | test')
    parser.add_argument('--scene_dir_queries', default='../data/{}/scenes')
    parser.add_argument('--scene_dir', default='../results/{}/predicted_boxes_large/scenes_predicted_nms_final')
    parser.add_argument('--room_dir', default='/media/reza/Large/{}/rooms')
    parser.add_argument('--models_dir', default='../data/{}/models')
    parser.add_argument('--mesh_regions_dir', default='../data/{}/mesh_regions_predicted_nms')
    parser.add_argument('--rendering_path', default='../results/{}/rendered_results_with_boxes')
    parser.add_argument('--colormap_path', default='../data/{}/color_map.json')
    parser.add_argument('--query_input_file_name', default='query_dict_top10.json')
    parser.add_argument('--query_list', dest='query_list', default=["all"], type=str, nargs='+',
                        help='Name of the queries to render. If ["all"] is chosen all queries will be rendered')
    parser.add_argument('--topk', dest='topk', default=10, type=int, help='Number of images rendered for each query.')
    parser.add_argument('--model_config_filename', default='3dssr_model_configs.json')

    return parser


if __name__ == '__main__':
    # read the args
    parser_ = argparse.ArgumentParser('Extract Regions', parents=[get_args()])
    args = parser_.parse_args()
    args.scene_dir = os.path.join(args.scene_dir, args.mode)
    args.scene_dir_queries = os.path.join(args.scene_dir_queries, args.mode)
    resolution = (1024, 1024)

    # load the query results for the model
    model_config = load_from_json(args.model_config_filename)[args.ranking_strategy]
    for k, v in model_config.items():
        vars(args)[k] = v
    adjust_paths(args, exceptions=[])
    query_results_dir = os.path.join(args.cp_dir.format(args.model_name), args.results_folder_name)
    query_results_file_name = args.query_input_file_name.split('.')[0] + '_{}_{}.json'.format(args.mode,
                                                                                              args.experiment_name)
    query_results = load_from_json(os.path.join(query_results_dir, query_results_file_name))

    # load the colormap for query images.
    args.colormap = load_from_json(args.colormap_path)

    main()
