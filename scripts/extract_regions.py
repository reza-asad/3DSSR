import os
import sys
import numpy as np
import trimesh

from scripts.helper import load_from_json, create_train_val_test


def extract_region(model_dir, obj_info, vertices, faces, face_normals, expansion_factor):
    # load the object mesh
    mesh = trimesh.load(os.path.join(model_dir, obj_info['file_name']))

    # find the centroid of the object and the dimensions of the region.
    centroid = np.asarray(obj_info['obbox'])[0, :]
    max_dist = mesh.bounding_box.extents / 2.0 * expansion_factor

    # compute the distance of each vertex to the centroid
    distance_x = np.abs(vertices[:, 0] - centroid[0])
    distance_y = np.abs(vertices[:, 1] - centroid[1])
    distance_z = np.abs(vertices[:, 2] - centroid[2])

    # find the vertices in the region.
    is_close = (distance_x < max_dist[0]) & (distance_y < max_dist[1]) & (distance_z < max_dist[2])
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


def derive_mesh_regions(room_dir, scene_dir, model_dir, scene_names, accepted_cats, results_dir, expansion_factor=2.0):
    for scene_name in scene_names:
        # skip if already extracted regions in the room.
        scene_name = scene_name.split('.')[0]
        if scene_name in visited:
            continue

        # load the room mesh.
        room_mesh = trimesh.load(os.path.join(room_dir, scene_name, '{}.annotated.ply'.format(scene_name)))

        # load the scene.
        scene = load_from_json(os.path.join(scene_dir, scene_name+'.json'))

        # for each accepted object in the room extract the region around it.
        for obj, obj_info in scene.items():
            if obj_info['category'][0] in accepted_cats:
                # copy the elements of the room mesh for the object.
                vertices = room_mesh.vertices.copy()
                faces = room_mesh.faces.copy()
                face_normals = room_mesh.face_normals.copy()

                # extract region for the object.
                mesh_region = extract_region(model_dir, obj_info, vertices, faces, face_normals, expansion_factor)

                # save the region.
                mesh_region.export(os.path.join(results_dir, obj_info['file_name']))

        # add room to the list of visited rooms.
        visited.add(scene_name)


def main(num_chunks, chunk_idx, action='extract_mesh'):
    # define some params
    data_dir = '../data/matterport3d'
    room_dir = os.path.join(data_dir, 'rooms')
    model_dir = os.path.join(data_dir, 'models')
    scenes_dir = os.path.join(data_dir, 'scenes', 'all')
    accepted_cats = load_from_json(os.path.join(data_dir, 'accepted_cats.json'))
    results_dir = os.path.join(data_dir, 'mesh_regions')
    results_dir_all = os.path.join(results_dir, 'all')
    if not os.path.exists(results_dir_all):
        try:
            os.makedirs(results_dir_all)
        except FileExistsError:
            pass

    if action == 'extract_mesh':
        scene_names = os.listdir(scenes_dir)
        chunk_size = int(np.ceil(len(scene_names) / num_chunks))
        derive_mesh_regions(room_dir=room_dir, scene_dir=scenes_dir, model_dir=model_dir,
                            scene_names=scene_names[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size],
                            accepted_cats=accepted_cats, results_dir=results_dir_all)

    if action == 'split_train_test_val':
        train_path = os.path.join(data_dir, 'scenes_train.txt')
        val_path = os.path.join(data_dir, 'scenes_val.txt')
        test_path = os.path.join(data_dir, 'scenes_test.txt')
        create_train_val_test(results_dir, train_path, val_path, test_path)


if __name__ == '__main__':
    visited = set()
    if len(sys.argv) == 1:
        main(1, 0, 'extract_mesh')
    elif len(sys.argv) == 2:
        main(1, 0, sys.argv[1])
    else:
        # To run in parallel you can use the command:
        # parallel -j5 "python3 -u extract_regions.py {1} {2}" ::: 5 ::: 0 1 2 3 4
        main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
