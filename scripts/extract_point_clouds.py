import os
import argparse
import numpy as np
import trimesh
from PIL import Image

from scripts.helper import load_from_json, sample_mesh, visualize_pc, render_single_pc, create_img_table


def derive_pc(region_names):
    for region_name in region_names:
        visited = set(os.listdir(args.pc_dir))
        file_name_pc = region_name.split('.')[0] + '.npy'
        if file_name_pc not in visited:
            # create the path to the mesh region
            mesh_region_path = os.path.join(args.mesh_dir, region_name)
            mesh_region = trimesh.load(os.path.join(mesh_region_path))

            # centralize the mesh.
            mesh_region.vertices -= np.mean(mesh_region.vertices, axis=0)

            # find the number of points to be sampled.
            pc, _ = sample_mesh(mesh_region, num_points=args.num_points)
            # mesh_region.show()
            # visualize_pc(pc)
            # t=y

            output_path = os.path.join(args.pc_dir, file_name_pc)
            np.save(output_path, pc)

            visited.add(file_name_pc)


def sample_pc_and_render():
    np.random.seed(0)
    pc_file_names = np.random.choice(os.listdir(args.pc_dir), args.num_imgs, replace=False)
    imgs, captions = [], []
    for pc_file_name in pc_file_names:
        # render the image and save it.
        pc = np.load(os.path.join(args.pc_dir, pc_file_name))
        img = render_single_pc(pc, resolution, rendering_kwargs, region=True)
        img = Image.fromarray(img)
        img_name = pc_file_name.split('.')[0] + '.png'
        img.save(os.path.join(args.rendering_dir, img_name))
        imgs.append(img_name)

        # use category as the caption.
        scene_name, obj_id = pc_file_name.split('.')[0].split('-')
        scene = load_from_json(os.path.join(args.scene_dir, '{}.json'.format(scene_name)))
        caption = scene[obj_id]['category'][0]
        captions.append(caption)

    # create img table.
    create_img_table(args.rendering_dir, 'imgs', imgs, 'img_table.html', ncols=3, captions=captions,
                     topk=args.num_imgs)


def get_args():
    parser = argparse.ArgumentParser('Extract Regions', add_help=False)
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--mode', default='test', help='train | val')
    parser.add_argument('--action', default='extract', help='extract | render')
    parser.add_argument('--scene_dir', default='../data/{}/scenes')
    parser.add_argument('--mesh_dir', default='../data/{}/mesh_regions_predicted')
    parser.add_argument('--pc_dir', default='../data/{}/pc_regions_predicted')
    parser.add_argument('--rendering_dir', default='../data/{}/pc_regions_rendered')
    parser.add_argument('--num_points', default=40960, type=int, help='number of points to sample from the mesh')
    parser.add_argument('--num_imgs', default=50, type=int, help='number of imgs to render')
    parser.add_argument('--seed', default=0, type=int, help='use different seed for parallel runs')
    parser.add_argument('--num_chunks', default=1, type=int, help='number of chunks for parallel run')
    parser.add_argument('--chunk_idx', default=0, type=int, help='chunk id for parallel run')

    return parser


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def main():
    if args.action == 'extract':
        np.random.seed(args.seed)
        np.random.shuffle(region_file_names)
        chunk_size = int(np.ceil(len(region_file_names) / args.num_chunks))
        derive_pc(region_file_names[args.chunk_idx * chunk_size: (args.chunk_idx + 1) * chunk_size])
    elif args.action == 'render':
        args.rendering_dir = os.path.join(args.rendering_dir, 'imgs')
        if not os.path.exists(args.rendering_dir):
            os.makedirs(args.rendering_dir)

        sample_pc_and_render()


if __name__ == '__main__':
    # rendering arguments.
    resolution = (512, 512)
    rendering_kwargs = {'fov': np.pi / 4, 'light_directional_intensity': 0.01, 'light_point_intensity_center': 0.0,
                        'wall_thickness': 5}
    # read the args
    parser_ = argparse.ArgumentParser('Extract Regions', parents=[get_args()])
    args = parser_.parse_args()
    adjust_paths(args, exceptions=[])

    args.mesh_dir = os.path.join(args.mesh_dir, args.mode)
    args.pc_dir = os.path.join(args.pc_dir, args.mode)
    args.scene_dir = os.path.join(args.scene_dir, args.mode)

    # find scene names.
    region_file_names = os.listdir(args.mesh_dir)

    # create the output pc dir if needed.
    if not os.path.exists(args.pc_dir):
        try:
            os.makedirs(args.pc_dir)
        except FileExistsError:
            pass

    main()
    # To run in parallel you can use the command:
    # parallel -j5 "python3 -u extract_point_clouds.py --mode {1} --seed {2} --num_chunks {3} --chunk_idx {4}" ::: val ::: 0 ::: 5 ::: 0 1 2 3 4
