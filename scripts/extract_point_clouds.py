import os
import argparse
import numpy as np
import trimesh

from scripts.helper import sample_mesh, visualize_pc


class PointCloud:
    def __init__(self, model_name):
        self.model_name = model_name.split('.')[0] + '.ply'

    def sample(self, num_points=40960, centralize=False):
        # sample points on the mesh
        mesh_path = os.path.join(mesh_regions_dir, self.model_name)
        mesh = trimesh.load(mesh_path, process=False)
        try:
            mesh.area_faces
        except AttributeError:
            return None

        # bring the pc to the center if necessary
        if centralize:
            mesh.vertices -= np.mean(mesh.vertices, axis=0)

        # find the number of points to be sampled.
        pc, _ = sample_mesh(mesh, num_points=num_points)

        return pc


def derive_pc(args, region_names):
    for region_name in region_names:
        visited = set(os.listdir(results_dir))
        region_name = region_name.split('.')[0] + '.npy'
        if region_name not in visited:
            pc_object = PointCloud(region_name)

            # sample point clouds
            pc = pc_object.sample(num_points=args.num_points, centralize=False)
            if pc is not None:
                # visualize_pc(pc)
                # t=y
                # save point clouds and labels
                output_path = os.path.join(results_dir, region_name)
                np.save(output_path, pc)

                visited.add(region_name)


def build_colored_pc(pc):
    radii = np.linalg.norm(pc, axis=1)
    colors = trimesh.visual.interpolate(radii, color_map='viridis')
    pc_vis = trimesh.points.PointCloud(pc, colors=colors)

    return pc_vis


def build_cube(center, pc_extents, coverage_percent):
    # find the scale of the cube
    scale = pc_extents * coverage_percent
    # build the cube
    cube = {'extents': scale, 'center': center}

    return cube


def sample_cube_centers(pc, pc_extents):
    denominator = 4
    is_center = []
    while np.sum(is_center) == 0:
        indices = np.arange(len(pc))
        is_center = np.abs(pc) <= (1/denominator * pc_extents / 2)
        is_center = np.sum(is_center, axis=1) == 3
        denominator -= 1

    return indices[is_center]


def find_sampled_crop(pc, cube, num_points):
    # take points inside the cube.
    is_inside = np.abs(pc - cube['center']) <= (cube['extents'] / 2.0)
    is_inside = np.sum(is_inside, axis=1) == 3
    crop = pc[is_inside, :]

    sampled_crop = None
    if len(crop) >= num_points:
        sampled_indices = np.random.choice(range(len(crop)), num_points, replace=False)
        sampled_crop = crop[sampled_indices, :]

    return sampled_crop


def sample_crop_pc(pc_file_names, crop_bounds, num_points, num_tries=200):
    # for each object sample points and extract a crop.
    coverage_percent = np.random.uniform(crop_bounds[0], crop_bounds[1], 3)
    for pc_file_name in pc_file_names:
        # load the pc and sample the center of the cube near the center of the object.
        input_path = os.path.join(pc_dir, pc_file_name)
        pc = np.load(input_path)
        pc_extents = trimesh.points.PointCloud(pc).extents
        sampled_indices = sample_cube_centers(pc, pc_extents)

        # try at least x times to get enough points.
        num_curr_tries = np.minimum(num_tries, len(sampled_indices))
        sampled_indices = np.random.choice(sampled_indices, num_curr_tries, replace=False)

        # iterate through the points and exit once you find enough points in the crop.
        sampled_crop = None
        for sampled_index in sampled_indices:
            center = pc[sampled_index, :]
            cube = build_cube(center, pc_extents, coverage_percent)

            # find the points inside the cube crop
            sampled_crop = find_sampled_crop(pc, cube, num_points)

            # if you find enough points, you found the crop so exit.
            if sampled_crop is not None:
                break

        # return the entire crop if sampling was not successful.
        if sampled_crop is None:
            print('Cropping not successful. Taking the entire pc')
            sampled_indices = np.random.choice(range(len(pc)), num_points, replace=False)
            sampled_crop = pc[sampled_indices, :]

        # build_colored_pc(pc).show()
        # build_colored_pc(sampled_crop).show()

        # save the crop
        output_path = os.path.join(results_dir, pc_file_name)
        np.save(output_path, sampled_crop)


def get_args():
    parser = argparse.ArgumentParser('Extract Regions', add_help=False)
    parser.add_argument('--action', default='sample_crop', help='extract | sample_crop')
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--mode', default='train', help='train | val | test')
    parser.add_argument('--mesh_regions_dir', default='../data/{}/mesh_regions')
    parser.add_argument('--pc_dir', default='../data/{}/pc_regions')
    parser.add_argument('--results_dir', default='../data/{}/pc_region_crops')
    parser.add_argument('--results_folder_name', default='crops')
    parser.add_argument('--num_points', default=4096, type=int, help='number of points randomly sampled form the pc.')
    parser.add_argument('--crop_bounds', type=float, nargs='+', default=(0.9, 0.9))
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
    region_names = os.listdir(mesh_regions_dir)
    np.random.seed(args.seed)
    if args.action == 'extract':
        np.random.shuffle(region_names)
        chunk_size = int(np.ceil(len(region_names) / args.num_chunks))
        derive_pc(args, region_names=region_names[args.chunk_idx * chunk_size: (args.chunk_idx + 1) * chunk_size])
    elif args.action == 'sample_crop':
        pc_file_names = os.listdir(pc_dir)
        sample_crop_pc(pc_file_names, crop_bounds=args.crop_bounds, num_points=args.num_points, num_tries=200)
        

if __name__ == '__main__':
    # read the args
    parser_ = argparse.ArgumentParser('Extract Regions', parents=[get_args()])
    args = parser_.parse_args()
    adjust_paths(args, exceptions=[])

    # load metadata
    mesh_regions_dir = os.path.join(args.mesh_regions_dir, args.mode)
    pc_dir = os.path.join(args.pc_dir, args.mode)
    results_dir = os.path.join(args.results_dir, args.results_folder_name, args.mode)
    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
        except FileExistsError:
            pass

    main()
    # To run in parallel you can use the command:
    # parallel -j5 "python3 -u extract_point_clouds.py --mode {1} --seed {2} --num_chunks {3} --chunk_idx {4}" ::: test ::: 0 ::: 5 ::: 0 1 2 3 4
    # parallel -j5 "python3 -u extract_point_clouds.py --mode {1} --results_folder_name {2} --seed {3} --num_chunks {4} --chunk_idx {5}" ::: test ::: crops_1 ::: 0 ::: 5 ::: 0 1 2 3 4
