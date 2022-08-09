import os
import argparse
import numpy as np
import trimesh

from scripts.helper import sample_mesh, visualize_pc


def derive_pc(room_names):
    for room_name in room_names:
        visited = set(os.listdir(pc_dir))
        room_name_pc = room_name + '.npy'
        if room_name_pc not in visited:
            # create the path to the room mesh
            room_dir = os.path.join(args.mesh_dir, room_name)
            if len(os.listdir(room_dir)) > 0:
                room_name_ply = room_name + '.annotated.ply'
                room_mesh = trimesh.load(os.path.join(room_dir, room_name_ply), process=False)

                # centralize the mesh.
                room_mesh.vertices -= np.mean(room_mesh.vertices, axis=0)

                # find the number of points to be sampled from the mesh.
                num_points = int(args.sampling_factor * np.sqrt(room_mesh.area))

                # find the number of points to be sampled.
                pc, _ = sample_mesh(room_mesh, num_points=num_points)
                # visualize_pc(pc)
                # t=y

                output_path = os.path.join(pc_dir, room_name_pc)
                np.save(output_path, pc)

            visited.add(room_name_pc)


def get_args():
    parser = argparse.ArgumentParser('Extract Regions', add_help=False)
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--mode', default='val', help='train | val | test')
    parser.add_argument('--scene_dir', default='../data/{}/scenes')
    parser.add_argument('--mesh_dir', default='/media/reza/Large/{}/rooms/')
    parser.add_argument('--pc_dir', default='../data/{}/pc_rooms')
    parser.add_argument('--results_folder_name', default='crops')
    parser.add_argument('--seed', default=0, type=int, help='use different seed for parallel runs')
    parser.add_argument('--sampling_factor', default=5000, type=int, help='used for finding numebr of points')
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
    np.random.seed(args.seed)
    np.random.shuffle(scene_names)
    chunk_size = int(np.ceil(len(scene_names) / args.num_chunks))
    derive_pc(scene_names[args.chunk_idx * chunk_size: (args.chunk_idx + 1) * chunk_size])


if __name__ == '__main__':
    # read the args
    parser_ = argparse.ArgumentParser('Extract Regions', parents=[get_args()])
    args = parser_.parse_args()
    adjust_paths(args, exceptions=[])

    # find scene names.
    scene_names = [e.split('.')[0] for e in os.listdir(os.path.join(args.scene_dir, args.mode))]

    # create the output pc dir if needed.
    pc_dir = os.path.join(args.pc_dir, args.mode)
    if not os.path.exists(pc_dir):
        try:
            os.makedirs(pc_dir)
        except FileExistsError:
            pass

    main()
    # To run in parallel you can use the command:
    # parallel -j5 "python3 -u extract_point_clouds.py --mode {1} --seed {2} --num_chunks {3} --chunk_idx {4}" ::: val ::: 0 ::: 5 ::: 0 1 2 3 4
