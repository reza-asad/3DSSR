import os
import sys
import shutil
import numpy as np
import trimesh

from scripts.helper import load_from_json, create_img_table
from scripts.box import Box
from render_scene_functions import render_pc, render_mesh


def sample_files(files_dir, num_files, seed=12):
    file_names = sorted(os.listdir(files_dir))
    np.random.seed(seed)
    np.random.shuffle(file_names)

    return file_names[:num_files]


def extract_region_mesh(obj_info, vertices, faces, face_normals, max_dist):
    # find the centroid of the object.
    if 'obbox' in obj_info:
        centroid = np.asarray(obj_info['obbox'])[0, :]
    else:
        centroid = np.asarray(obj_info['aabb'])[0, :]

    # compute the distance of each vertex to the centroid
    distance_x = np.abs(vertices[:, 0] - centroid[0])
    distance_y = np.abs(vertices[:, 1] - centroid[1])
    distance_z = np.abs(vertices[:, 2] - centroid[2])

    # find the vertices in the region.
    is_close = (distance_x <= max_dist[0]) & (distance_y <= max_dist[1]) & (distance_z <= max_dist[2])
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


def extract_region_pc(obj_info, pc, num_points, max_dist):
    # find the centroid of the object.
    centroid = np.asarray(obj_info['obbox'])[0, :]

    # compute the distance of each point to the centroid
    distance_x = np.abs(pc[:, 0] - centroid[0])
    distance_y = np.abs(pc[:, 1] - centroid[1])

    # find the points in the region.
    is_close = (distance_x <= max_dist[0]) & (distance_y <= max_dist[1])
    pc_region = pc[is_close]
    if len(pc_region) < num_points:
        pc_region = pc.copy()

    # translate the vertices so that the center of the object is at the origin.
    pc_region = pc_region - centroid

    return pc_region


def derive_mesh_regions(scene_names):
    for scene_name in scene_names:
        # load the room mesh.
        scene_name = scene_name.split('.')[0]
        room_mesh = trimesh.load(os.path.join(room_dir, scene_name, '{}.annotated.ply'.format(scene_name)))
        # room_mesh.show()

        # load the scene.
        scene = load_from_json(os.path.join(scenes_dir, scene_name+'.json'))

        # for each accepted object in the room extract the surrounding region.
        for obj, obj_info in scene.items():
            # if obj != '5':
            #     continue
            if obj_info['category'][0] in accepted_cats:
                visited = set([e.split('.')[0] for e in os.listdir(results_dir)])
                model_name = obj_info['file_name'].split('.')[0]
                if model_name in visited:
                    continue

                # copy the elements of the room mesh for the object.
                vertices = room_mesh.vertices.copy()
                faces = room_mesh.faces.copy()
                face_normals = room_mesh.face_normals.copy()

                # extract region for the object.
                if region_size == 'fixed':
                    room_aabbox_extents = room_mesh.bounding_box.extents
                    max_dist = room_aabbox_extents / 2.0 * scene_coverage
                elif region_size == 'variable':
                    # load the object mesh
                    mesh = trimesh.load(os.path.join(model_dir, obj_info['file_name']))
                    max_dist = mesh.bounding_box.extents / 2.0 * expansion_factor
                elif region_size == 'given':
                    box_corners = np.asarray(obj_info['aabb'], dtype=np.float64)
                    aabb = Box(box_corners)
                    max_dist = aabb.scale / 2.0
                    # box_vis = trimesh.creation.box(aabb.scale, aabb.transformation)
                    # trimesh.Scene([room_mesh, box_vis]).show()
                else:
                    raise NotImplementedError('region_size {} not implremented'.format(region_size))
                mesh_region = extract_region_mesh(obj_info, vertices, faces, face_normals, max_dist)
                # vertices = np.asarray(scene[obj]['aabb'], dtype=np.float64)
                # print(scene[obj]['category'][0])
                # obbox = Box(vertices)
                # transformation = np.eye(4)
                # transformation[:3, 3] = -obbox.translation
                # obbox = obbox.apply_transformation(transformation)
                # obbox = trimesh.creation.box(obbox.scale, obbox.transformation)
                # obbox.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("##FF0000")
                # trimesh.Scene([mesh_region]).show()
                # obj = trimesh.load(os.path.join(model_dir, scene[obj]['file_name']))
                # obj.show()
                # t=y
                # save the region.
                mesh_region.export(os.path.join(results_dir, obj_info['file_name']))

                # add room to the list of visited rooms.
                visited.add(model_name)


def derive_pc_regions(scene_names, num_points=4096):
    for scene_name in scene_names:
        # load the scene.
        scene_name = scene_name.split('.')[0]
        scene = load_from_json(os.path.join(scenes_dir, scene_name+'.json'))

        # load the point cloud representation of the scene
        pc = np.load(os.path.join(room_dir, scene_name+'.npy'))
        if len(pc) == 0:
            print(scene_name)

        pc_scene = trimesh.points.PointCloud(pc)

        # for each accepted object in the room extract the region around it.
        for obj, obj_info in scene.items():
            if obj_info['category'][0] in accepted_cats:
                visited = set([e.split('.')[0] for e in os.listdir(results_dir)])
                model_name = obj_info['file_name'].split('.')[0]
                if model_name in visited:
                    continue

                if region_size == 'fixed':
                    room_aabbox_extents = pc_scene.bounding_box.extents
                    max_dist = room_aabbox_extents / 2.0 * scene_coverage
                else:
                    # load the object mesh
                    mesh = trimesh.load(os.path.join(model_dir, obj_info['file_name']))
                    max_dist = mesh.bounding_box.extents / 2.0 * expansion_factor

                # create a dense point cloud object representing the mesh region
                pc_region = extract_region_pc(obj_info, pc, num_points, max_dist)
                # TODO: visualize
                # radii = np.linalg.norm(pc_region, axis=1)
                # colors = trimesh.visual.interpolate(radii, color_map='viridis')
                # trimesh.points.PointCloud(pc_region, colors=colors).show()
                # t=y
                # save the region.
                np.save(os.path.join(results_dir, obj_info['file_name'].split('.')[0] + '.npy'), pc_region)

                # add room to the list of visited rooms.
                visited.add(model_name)


def find_processed_scenes(scene_names):
    # find all the objects for each processed scene.
    scene_obj = {}
    for file_name in os.listdir(results_dir):
        scene_name, obj = file_name.split('-')
        obj = obj.split('.')[0]
        if scene_name in scene_obj:
            scene_obj[scene_name].append(obj)
        else:
            scene_obj[scene_name] = [obj]

    # find the accepted objects in each processed scene
    filtered_scene_names = []
    for scene_name in scene_names:
        scene_name = scene_name.split('.')[0]
        if scene_name not in scene_obj:
            filtered_scene_names.append(scene_name+'.json')
        else:
            scene = load_from_json(os.path.join(scenes_dir, scene_name+'.json'))
            itr = 0
            for obj, obj_info in scene.items():
                if (obj_info['category'][0] in accepted_cats) and (obj in scene_obj[scene_name]):
                    itr += 1
            if itr != len(scene_obj[scene_name]):
                filtered_scene_names.append(scene_name+'.json')

    return filtered_scene_names


def main(num_chunks, chunk_idx, action='extract'):
    if action == 'extract':
        scene_names = os.listdir(scenes_dir)
        # scene_names = find_processed_scenes(scene_names)

        # processed_mesh = set([e.split('-')[0] for e in os.listdir(os.path.join(data_dir, 'mesh_regions', mode))])
        # scene_names = [e for e in scene_names if e.split('.')[0] in processed_mesh]
        np.random.seed(3)
        np.random.shuffle(scene_names)
        chunk_size = int(np.ceil(len(scene_names) / num_chunks))
        if region_type == 'mesh':
            derive_mesh_regions(scene_names=scene_names[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size])
        elif region_type == 'pc':
            derive_pc_regions(scene_names=scene_names[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size])
    elif action == 'render':
        if region_type == 'mesh':
            mesh_file_names = sample_files(results_dir, num_imgs)
            render_mesh(results_dir, mesh_file_names, results_dir_rendered, resolution, rendering_kwargs, region=True)
        elif region_type == 'pc':
            pc_file_names = sample_files(results_dir, num_imgs)
            render_pc(results_dir, pc_file_names, results_dir_rendered, resolution, rendering_kwargs, region=True)
    elif action == 'create_img_table':
        imgs = os.listdir(results_dir_rendered)
        create_img_table(results_dir_rendered, 'imgs', imgs, 'img_table.html', ncols=4, captions=imgs, topk=num_imgs)
    elif action == 'save_room_subset':
        pc_file_names = sample_files(results_dir, num_imgs)
        out_dir = 'pc_regions_subset'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        for pc_file_name in pc_file_names:
            shutil.copy(os.path.join(results_dir, pc_file_name), os.path.join(out_dir, pc_file_name))


if __name__ == '__main__':
    # set up rendering parameters
    resolution = (512, 512)
    rendering_kwargs = {'fov': np.pi / 4, 'light_directional_intensity': 0.01, 'light_point_intensity_center': 0.0,
                        'wall_thickness': 5}
    num_imgs = 20

    # define the type of extract and its size
    region_type = 'mesh'
    region_size = 'given'
    scene_coverage = 0.3
    expansion_factor = 1.5

    # define paths
    mode = 'test'
    dataset_name = 'matterport3d'
    split_char = '_' if dataset_name == 'matterport3d' else '-'
    data_dir = '../data/{}'.format(dataset_name)
    model_dir = os.path.join(data_dir, 'models')
    scenes_dir = os.path.join(data_dir, 'scenes', mode)
    accepted_cats = load_from_json(os.path.join(data_dir, 'accepted_cats.json'))
    if region_type == 'pc':
        room_dir = '/media/reza/Large/matterport3d/rooms_pc'
        results_dir = os.path.join(data_dir, 'pc_regions', mode)#'/media/reza/Large/pc_regions/{}'.format(mode)
        results_dir_rendered = os.path.join(data_dir, 'pc_regions_rendered', 'imgs')
    else:
        room_dir = '../data/matterport3d/rooms'
        results_dir = os.path.join(data_dir, 'mesh_regions_predicted_nms', mode)
        results_dir_rendered = os.path.join(data_dir, 'mesh_regions_rendered', 'imgs')
        scenes_dir = os.path.join('../results/{}/predicted_boxes_large/scenes_predicted_nms_raw'.format(dataset_name), mode)

    for folder in [results_dir]:
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
            except FileExistsError:
                pass

    if len(sys.argv) == 1:
        main(1, 0, 'extract')
    elif len(sys.argv) == 2:
        main(1, 0, sys.argv[1])
    else:
        # To run in parallel you can use the command:
        # parallel -j5 "python3 -u extract_regions.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: extract
        main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
