import os
import argparse
import numpy as np
import trimesh

from scripts.helper import load_from_json


def extract_mesh_region(obj_info, vertices, faces, face_normals, max_dist):
    # find the centroid of the object and the distance from each vertex to the centroid.
    centroid = np.asarray(obj_info['aabb'])[0:3]
    distances = np.abs(vertices - centroid)

    # find the vertices in the region.
    is_close = np.sum(distances < max_dist, axis=1) == 3
    filtered_vertices = vertices[is_close]

    # find a map that sends the indices of the filter vertices in the old mesh to the vertex indices in the
    # filtered mesh.
    old_indices_filtered = np.arange(len(vertices))[is_close]
    old_to_new_vertex_map = dict(zip(old_indices_filtered, np.arange(len(filtered_vertices))))

    # filter the room mesh faces and normals to only contain the filtered vertices.
    filtered_faces, filtered_face_normals = [], []
    for i, face in enumerate(faces):
        if np.sum([True if v in old_to_new_vertex_map else False for v in face]) == 3:
            new_face = [old_to_new_vertex_map[v] for v in face]
            filtered_faces.append(new_face)
            filtered_face_normals.append(face_normals[i, :])

    # translate the vertices so that the center of the object is at the origin.
    filtered_vertices = filtered_vertices - centroid

    # build the mesh
    mesh = trimesh.Trimesh(vertices=filtered_vertices, faces=filtered_faces, face_normals=filtered_face_normals)

    return mesh


def derive_mesh_regions(scene_names):
    for scene_name in scene_names:
        # load the room mesh.
        scene_name = scene_name.split('.')[0]
        room_mesh = trimesh.load(os.path.join(args.room_dir, scene_name, '{}_vh_clean_2.ply'.format(scene_name)))

        # align the room mesh
        lines = open(os.path.join(args.room_dir, scene_name, '{}.txt'.format(scene_name))).readlines()
        for line in lines:
            if 'axisAlignment' in line:
                axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
        axis_align_matrix = np.asarray(axis_align_matrix).reshape((4, 4))
        room_mesh = room_mesh.apply_transform(axis_align_matrix)

        # load the scene.
        scene = load_from_json(os.path.join(args.scene_dir, scene_name+'.json'))
        # for each accepted object in the room extract the surrounding region.
        for obj, obj_info in scene.items():
            visited = set([e.split('.')[0] for e in os.listdir(args.mesh_dir_output)])
            model_name = '{}-{}'.format(scene_name, obj)
            if model_name in visited:
                continue

            # copy the elements of the room mesh for the object.
            vertices = room_mesh.vertices.copy()
            faces = room_mesh.faces.copy()
            face_normals = room_mesh.face_normals.copy()

            # extract region for the object.
            aabb_scale = np.asarray(obj_info['aabb'][3:], dtype=np.float32)
            max_dist = aabb_scale / 2.0 * args.expansion_factor
            # aabb_transform = np.asarray(obj_info['transform'], dtype=np.float32).reshape(4, 4).transpose()
            # box_vis = trimesh.creation.box(aabb_scale, aabb_transform)
            # room_mesh.show()
            # trimesh.Scene([room_mesh, box_vis]).show()
            mesh_region = extract_mesh_region(obj_info, vertices, faces, face_normals, max_dist)

            # save the region.
            mesh_region.export(os.path.join(args.mesh_dir_output, '{}.ply'.format(model_name)))

            # add room to the list of visited rooms.
            visited.add(model_name)


def adjust_paths(exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def get_args():
    parser = argparse.ArgumentParser('Extract Regions', add_help=False)
    parser.add_argument('--dataset', default='scannet')
    parser.add_argument('--mode', default='val', help='train | val')
    parser.add_argument('--action', default='extract', help='extract | render')
    parser.add_argument('--scene_dir', default='../data/{}/scenes')
    parser.add_argument('--room_dir', default='/media/reza/Large/{}/scans/')
    parser.add_argument('--mesh_dir_output', default='../data/{}/mesh_regions')
    parser.add_argument('--seed', default=0, type=int, help='use different seed for parallel runs')
    parser.add_argument('--expansion_factor', default=1.5, type=int, help='used for finding numebr of points')
    parser.add_argument('--num_chunks', default=1, type=int, help='number of chunks for parallel run')
    parser.add_argument('--chunk_idx', default=0, type=int, help='chunk id for parallel run')

    return parser


def main():
    scene_names = os.listdir(args.scene_dir)
    np.random.seed(args.seed)
    np.random.shuffle(scene_names)
    chunk_size = int(np.ceil(len(scene_names) / args.num_chunks))
    derive_mesh_regions(scene_names=scene_names[args.chunk_idx * chunk_size: (args.chunk_idx + 1) * chunk_size])


if __name__ == '__main__':
    # read the args
    parser_ = argparse.ArgumentParser('Build Scenes', parents=[get_args()])
    args = parser_.parse_args()
    adjust_paths(exceptions=[])

    args.scene_dir = os.path.join(args.scene_dir, args.mode)
    args.mesh_dir_output = os.path.join(args.mesh_dir_output, args.mode)
    if not os.path.exists(args.mesh_dir_output):
        try:
            os.makedirs(args.mesh_dir_output)
        except FileExistsError:
            pass

    main()
    # To run in parallel you can use the command:
    # parallel -j5 "python3 -u extract_mesh_regions.py --mode {1} --seed {2} --num_chunks {3} --chunk_idx {4}" ::: val ::: 0 ::: 5 ::: 0 1 2 3 4
