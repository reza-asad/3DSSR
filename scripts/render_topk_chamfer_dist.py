import os
import argparse
import pandas as pd
import numpy as np
import trimesh
from PIL import Image
import torch
from chamferdist import ChamferDistance

from scripts.helper import load_from_json, create_img_table
from scripts.renderer import Render


def load_pc(args, file_name):
    pc = np.load(os.path.join(args.pc_dir, file_name))

    # sample N points
    np.random.seed(0)
    sampled_indices = np.random.choice(range(len(pc)), args.num_points, replace=False)
    pc = np.expand_dims(pc[sampled_indices, :], axis=0)
    pc = torch.from_numpy(pc).cuda()

    return pc


def find_top_cd(args, file_names, thresholds):
    # load the pc for the query aabb
    pc_q = load_pc(args, args.query_file_name + '.npy')

    # compute CD between the query and all aabbs with the same category as query.
    chamferDist = ChamferDistance()
    threshold_file_names = {t: [np.inf, None] for t in thresholds.keys()}
    threshold_file_names['100'] = [0, None]
    for i, file_name in enumerate(file_names):
        print('Iternation {}/{}'.format(i+1, len(file_names)))

        # load the pc for the target aabb.
        pc_t = load_pc(args, file_name + '.npy')

        # compute chamfer distance.
        dist_forward = chamferDist(pc_q, pc_t)
        dist = dist_forward.detach().cpu().item()

        # find the closest candidate for each threshold
        for k, threshold in thresholds.items():
            if dist <= threshold and (np.abs(dist - threshold) < np.abs(threshold_file_names[k][0] - threshold)):
                threshold_file_names[k][0] = dist
                threshold_file_names[k][1] = file_name
        if dist > threshold_file_names['100'][0]:
            threshold_file_names['100'][0] = dist
            threshold_file_names['100'][1] = file_name

    return threshold_file_names


def render_topk_CD(args, topk, threshold, file_name, output_dir):
    # load the mesh and add color
    mesh = trimesh.load(os.path.join(args.mesh_region_dir, file_name+'.ply'))
    mesh.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#83c2bc")
    mesh = trimesh.Trimesh.scene(trimesh.load(mesh))

    # set up camera pose and room dimensions.
    room_dimension = mesh.extents
    camera_pose, _ = mesh.graph[mesh.camera.name]

    # determine the output path
    if threshold is not None:
        path = os.path.join(output_dir, '{}_{}_'.format(topk, np.round(threshold, 2)) + file_name + '.png'.format())
    else:
        path = os.path.join(output_dir, '{}_'.format(topk) + file_name + '.png'.format())

    # render
    r = Render(rendering_kwargs)
    img, _ = r.pyrender_render(mesh, resolution=resolution, camera_pose=camera_pose, room_dimension=room_dimension)
    img = Image.fromarray(img)
    img.save(path)


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def get_args():
    parser = argparse.ArgumentParser('Render Topk CD thresholds', add_help=False)
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--mode', default='test', help='val | test')
    parser.add_argument('--metadata_path', default='../data/{}/metadata_non_equal_full_top10.csv')
    parser.add_argument('--scene_dir', default='../results/{}/scenes_top10')
    parser.add_argument('--model_dir', default='../data/{}/models')
    parser.add_argument('--rendering_dir', default='../results/{}/topk_CD_rendered')
    parser.add_argument('--query_file_name', default='yqstnuAEVhm_room12-3')
    parser.add_argument('--pc_dir', default='../data/{}/pc_regions')
    parser.add_argument('--mesh_region_dir', default='../data/{}/mesh_regions')
    parser.add_argument('--cd_path', default='../data/{}/cd_thresholds.json')
    parser.add_argument('--num_points', default=4096, type=int, help='number of points randomly sampled form the pc.')

    return parser


def main():
    # read the args
    parser = argparse.ArgumentParser('Pre-compute DF', parents=[get_args()])
    args = parser.parse_args()
    adjust_paths(args, exceptions=[])
    args.pc_dir = os.path.join(args.pc_dir, args.mode)
    args.mesh_region_dir = os.path.join(args.mesh_region_dir, args.mode)
    args.scene_dir = os.path.join(args.scene_dir, args.mode)
    args.rendering_dir = os.path.join(args.rendering_dir, args.mode)
    args.cd_path = args.cd_path.split('.json')[0] + '_{}.json'.format(args.mode)

    # load accepted categories and the metadata.
    df_metadata = pd.read_csv(args.metadata_path)
    df_metadata = df_metadata[df_metadata['split'] == args.mode]

    # find all file_names with same category as the query obj.
    df_metadata['file_name'] = df_metadata[['room_name', 'objectId']].apply(lambda x: '-'.join([str(x[0]), str(x[1])]),
                                                                            axis=1)
    q_cat = df_metadata.loc[df_metadata['file_name'] == args.query_file_name, 'mpcat40'].values[0]
    file_names = df_metadata.loc[df_metadata['mpcat40'] == q_cat, 'file_name'].tolist()

    # load the chamfer distance (CD) thresholds for query cat.
    cd_sim = load_from_json(args.cd_path)
    thresholds = cd_sim[q_cat]

    # find the aabbs with similar category as the query aabb at various CD thresholds.
    file_names.remove(args.query_file_name)
    threshold_file_names = find_top_cd(args, file_names, thresholds)

    # create the output dir
    output_dir = os.path.join(args.rendering_dir, args.query_file_name, 'imgs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # rend the point cloud within each aabb query + targets
    render_topk_CD(args, 'query', None, args.query_file_name, output_dir)
    for topk, (threshold, file_name) in threshold_file_names.items():
        if file_name is not None:
            render_topk_CD(args, topk, threshold, file_name, output_dir)

    # create an image table
    imgs = os.listdir(output_dir)
    query_img = [img for img in imgs if 'query' in img]
    imgs.remove(query_img[0])
    imgs = sorted(imgs, key=lambda x: int(x.split('_')[0]))
    captions = [img for img in imgs]
    create_img_table(output_dir, 'imgs', imgs, 'img_table.html', topk=5, ncols=5, captions=captions,
                     with_query_scene=True, evaluation_plot=None, query_img=query_img[0], query_caption=None)


if __name__ == '__main__':
    rendering_kwargs = {'fov': np.pi / 4, 'light_directional_intensity': 0.01, 'light_point_intensity_center': 0.0,
                        'wall_thickness': 5}
    resolution = (512, 512)
    main()
