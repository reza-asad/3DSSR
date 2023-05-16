import os
import argparse
import numpy as np
import trimesh

from scripts.helper import load_from_json
from scripts.box import Box


def visualize_mesh_region(room_mesh, mesh_region, translation, obj_cat):
    # visualize the room
    room_mesh.show()

    # visualize the aabb in the room
    vertices = mesh_region.bounding_box.vertices
    centroid = np.mean(vertices, axis=0)
    centroid = np.expand_dims(centroid, axis=0)
    vertices = np.concatenate([centroid, vertices], axis=0)
    vertices = np.asarray(vertices, dtype=np.float64)
    aabb = Box(vertices)
    transformation = np.eye(4)
    transformation[:3, 3] = translation
    aabb = aabb.apply_transformation(transformation)
    aabb = trimesh.creation.box(aabb.scale, aabb.transformation)
    aabb.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("##FF0000")
    trimesh.Scene([room_mesh, aabb]).show()

    # visualize the mesh region
    mesh_region.show()
    print(obj_cat)


def find_median_box_scale(scene_names):
    # for each scene record the max scale along the aabb (half scale)
    max_half_scales = []
    for scene_name in scene_names:
        # load the scene
        scene = load_from_json(os.path.join(scenes_dir, scene_name))
        for obj, obj_info in scene.items():
            if obj_info['category'][0] in accepted_cats:
                vertices = np.asarray(obj_info['aabb'], dtype=np.float64)
                aabb = Box(vertices)
                max_half_scales.append(np.max(aabb.scale / 2.0))

    print('Median of the maximum half scale is: {}'.format(np.median(max_half_scales)))


def extract_region_mesh(obj_vertices, vertices, faces, face_normals, max_dist):
    # randomly select the centroid of the cube crop from the object vertices.
    centroid_idx = np.random.choice(range(len(obj_vertices)))
    centroid = obj_vertices[centroid_idx]

    # compute the distance of each vertex to the centroid
    distance_x = np.abs(vertices[:, 0] - centroid[0])
    distance_y = np.abs(vertices[:, 1] - centroid[1])
    distance_z = np.abs(vertices[:, 2] - centroid[2])

    # find the vertices in the region.
    is_close = (distance_x <= max_dist) & (distance_y <= max_dist) & (distance_z <= max_dist)
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

    return mesh, centroid


def derive_mesh_regions(args, scene_names):
    for scene_name in scene_names:
        # load the room mesh.
        scene_name = scene_name.split('.')[0]
        room_mesh = trimesh.load(os.path.join(args.room_dir, scene_name, '{}.annotated.ply'.format(scene_name)))

        # load the scene.
        scene = load_from_json(os.path.join(scenes_dir, scene_name+'.json'))

        # for each accepted object in the room extract the surrounding region.
        for obj, obj_info in scene.items():
            obj_cat = obj_info['category'][0]
            if obj_cat in accepted_cats:
                visited = set([e.split('.')[0] for e in os.listdir(results_dir)])
                model_name = obj_info['file_name'].split('.')[0]
                if model_name in visited:
                    continue

                # copy the elements of the room mesh for the object.
                vertices = room_mesh.vertices.copy()
                faces = room_mesh.faces.copy()
                face_normals = room_mesh.face_normals.copy()

                # load the model, bring it to the scene coordinate and take all its vertices
                obj_mesh = trimesh.load(os.path.join(args.models_dir, obj_info['file_name']))
                transformation = np.eye(4, dtype=np.float64)
                transformation[:3, 3] = np.asarray(obj_info['obbox'])[0, :]
                obj_mesh.apply_transform(transformation)
                obj_vertices = obj_mesh.vertices

                # extract the mesh region
                mesh_region, translation = extract_region_mesh(obj_vertices, vertices, faces, face_normals,
                                                               args.max_dist)

                # TODO: visualzie the mesh
                # visualize_mesh_region(room_mesh, mesh_region, translation, obj_cat)
                # t=y
                # save the region.
                mesh_region.export(os.path.join(results_dir, obj_info['file_name']))

                # add room to the list of visited rooms.
                visited.add(model_name)


def get_args():
    parser = argparse.ArgumentParser('Extract Regions', add_help=False)
    parser.add_argument('--action', default='extract', help='find_median_box_scale | extract')
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--accepted_cats_path', default='../data/{}/accepted_cats_top10.json')
    parser.add_argument('--mode', default='val', help='train | val | test')
    parser.add_argument('--room_dir', default='/media/reza/Large/Matterport3D_rooms/rooms/')
    parser.add_argument('--scene_dir', default='../results/{}/scenes_top10')
    parser.add_argument('--models_dir', default='../data/{}/models')
    parser.add_argument('--results_dir', default='../data/{}/mesh_regions_approximate')
    parser.add_argument('--num_points', default=4096, type=int, help='number of points randomly sampled form the pc.')
    parser.add_argument('--max_dist', default=0.65, type=int, help='the half scale of the cube enclosing each region')
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
    scene_names = os.listdir(scenes_dir)
    if args.action == 'extract':
        np.random.seed(args.seed)
        np.random.shuffle(scene_names)
        chunk_size = int(np.ceil(len(scene_names) / args.num_chunks))
        derive_mesh_regions(args, scene_names=scene_names[args.chunk_idx * chunk_size: (args.chunk_idx + 1) * chunk_size])
    elif args.action == 'find_median_box_scale':
        find_median_box_scale(scene_names)


if __name__ == '__main__':
    # read the args
    parser_ = argparse.ArgumentParser('Extract Regions', parents=[get_args()])
    args = parser_.parse_args()
    adjust_paths(args, exceptions=[])

    results_dir = os.path.join(args.results_dir, args.mode)
    scenes_dir = os.path.join(args.scene_dir, args.mode)
    accepted_cats = load_from_json(os.path.join(args.accepted_cats_path))
    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
        except FileExistsError:
            pass

    main()
    # To run in parallel you can use the command:
    # parallel -j5 "python3 -u extract_regions_approximate.py --mode {1} --seed {2} --num_chunks {3} --chunk_idx {4}" ::: test ::: 0 ::: 5 ::: 0 1 2 3 4
