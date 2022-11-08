import os
import argparse
import numpy as np
import trimesh

from scripts.helper import load_from_json
from scripts.box import Box


def extract_mesh_region(centroid, vertices, faces, face_normals, max_dist):
    # find the centroid of the object and the distance from each vertex to the centroid.
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


def derive_mesh_regions(target_subscenes_chunk):
    for target_subscene in target_subscenes_chunk:
        # load the room mesh.
        scene_name = target_subscene['scene_name']
        room_mesh = trimesh.load(os.path.join(args.room_dir, scene_name, '{}.annotated.ply'.format(scene_name)))

        # for each predicted box find the mesh region it corresponds to.
        visited = set(os.listdir(args.mesh_regions_dir))
        for i, box_vertices in enumerate(target_subscene['boxes']):
            key = '{}_{}_{}.ply'.format(query, scene_name, i)
            if key in visited:
                continue

            # copy the elements of the room mesh for the object.
            vertices = room_mesh.vertices.copy()
            faces = room_mesh.faces.copy()
            face_normals = room_mesh.face_normals.copy()

            # extract region for the object.
            centroid = box_vertices[0]
            predicted_box = Box(np.asarray(box_vertices, dtype=np.float32))
            max_dist = predicted_box.scale / 2.0
            # transformation = np.eye(4, dtype=np.float32)
            # transformation[:3, 3] = centroid
            # box_vis = trimesh.creation.box(predicted_box.scale, transformation)
            # room_mesh.show()
            # trimesh.Scene([room_mesh, box_vis]).show()
            mesh_region = extract_mesh_region(centroid, vertices, faces, face_normals, max_dist)
            # mesh_region.show()
            # tt
            # save the region.
            mesh_region.export(os.path.join(args.mesh_regions_dir, key))

            # add room to the list of visited rooms.
            visited.add(key)


def get_args():
    parser = argparse.ArgumentParser('Extract Regions', add_help=False)
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--mode', default='test', help='train | val')
    parser.add_argument('--room_dir', default='/media/reza/Large/{}/rooms/')
    parser.add_argument('--mesh_regions_dir', default='../data/{}/mesh_regions_predicted')
    parser.add_argument("--output_path", default='../results/{}/LayoutMatching/', type=str)
    parser.add_argument('--query_input_file_name', default='query_dict_top10.json')
    parser.add_argument('--results_folder_name', default='full_3dssr_real_query')
    parser.add_argument("--experiment_name", default='3detr_pre_rank', type=str)
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


if __name__ == '__main__':
    # read the args
    parser_ = argparse.ArgumentParser('Extract Regions', parents=[get_args()])
    args = parser_.parse_args()
    adjust_paths(args, exceptions=[])

    # load the query results.
    output_path = os.path.join(args.output_path, args.results_folder_name)
    query_input_file_name = args.query_input_file_name.split('.')[0] + '_{}_{}.json'.format(args.mode,
                                                                                            args.experiment_name)
    query_dict_input_path = os.path.join(output_path, query_input_file_name)
    query_results = load_from_json(query_dict_input_path)

    # use the right mode for the extracted pc's
    args.mesh_regions_dir = os.path.join(args.mesh_regions_dir, args.mode)

    # create the output pc dir if needed.
    if not os.path.exists(args.mesh_regions_dir):
        try:
            os.makedirs(args.mesh_regions_dir)
        except FileExistsError:
            pass

    # for each query extract mesh from the scene using the predicted target boxes.
    for idx, (query, results_info) in enumerate(query_results.items()):
        print('Processing query {}/{}'.format(idx+1, len(query_results)))
        target_subscenes = results_info['target_subscenes']
        np.random.seed(args.seed)
        np.random.shuffle(target_subscenes)
        chunk_size = int(np.ceil(len(target_subscenes) / args.num_chunks))
        derive_mesh_regions(target_subscenes[args.chunk_idx * chunk_size: (args.chunk_idx + 1) * chunk_size])

    # To run in parallel you can use the command:
    # parallel -j5 "python3 -u extract_mesh_from_predicted_boxes.py --mode {1} --seed {2} --num_chunks {3} --chunk_idx {4}" ::: test ::: 0 ::: 5 ::: 0 1 2 3 4
