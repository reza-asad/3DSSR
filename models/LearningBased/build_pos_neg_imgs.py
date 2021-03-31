import os
import sys
import numpy as np
import trimesh
from PIL import Image
from time import time

from scripts.helper import load_from_json, prepare_mesh_for_scene, write_to_json
from scripts.renderer import Render


def sample_negative_objects(graph, pos_obj, iou_threshold=0.2):
    # sample negative candidates by comparing IoU
    neg_objects = []
    neg_candidates = []
    for metrics in graph[pos_obj]['ring_info'].values():
        for obj, iou in metrics['iou']:
            if iou < iou_threshold:
                neg_candidates.append((obj, iou))

    # sort the candidates from highest to lowest IoU, if there are any.
    if len(neg_candidates) > 0:
        sorted_neg_candidates = sorted(neg_candidates, reverse=True, key=lambda x: x[1])
        neg_objects = list(list(zip(*sorted_neg_candidates))[0])

    return neg_objects


def prepare_scene(graph, obj_to_mesh, center_object, ceiling_cats=['ceiling', 'void']):
    obj_scene = []
    scene = []
    for obj, mesh in obj_to_mesh.items():
        cat = graph[obj]['category'][0]

        # if the mesh is a ceiling mesh skip
        if cat in ceiling_cats:
            continue

        # add objects to the context and object scene.
        scene.append(mesh)
        if obj == center_object:
            obj_scene.append(mesh)

    # build the scene and extract camera pose and room dimensions
    obj_scene = trimesh.Scene(obj_scene)
    scene = trimesh.Scene(scene)
    room_dimension = scene.extents
    camera_pose, _ = scene.graph[scene.camera.name]

    return obj_scene, scene, camera_pose, room_dimension


def adjust_camera_view(camera_pose, obj_centroid, scene_centroid, min_angle=15, max_angle=25):
    # translate the camera pose to be above the object of interest.
    camera_pose[:2, 3] = obj_centroid[:2]

    # find the axis along which the rotation of the camera happens.
    obj_to_scene = scene_centroid - obj_centroid
    obj_to_scene_xy = obj_to_scene[:2]
    obj_to_scene_xy = obj_to_scene_xy / np.linalg.norm(obj_to_scene_xy)
    direction = [-obj_to_scene_xy[1], obj_to_scene_xy[0], 0]

    # the sign of the angle is positive if the room's centroid is on the right of the direction otherwise negative.
    mid_to_scene_centroid = scene_centroid - (obj_centroid + obj_to_scene / 2.0)
    mid_to_scene_centroid_xy = mid_to_scene_centroid[:2]
    mid_to_scene_centroid_xy = mid_to_scene_centroid_xy / np.linalg.norm(mid_to_scene_centroid_xy)

    perpendicular_direction = [mid_to_scene_centroid_xy[0], mid_to_scene_centroid_xy[1], 0]
    z_axis = np.cross(direction, perpendicular_direction)
    random_angle = np.random.uniform(min_angle, max_angle)
    angle = np.radians(random_angle)
    if z_axis[2] > 0:
        angle = -angle

    # rotate the camera pose to point at the object of interest
    rotation = trimesh.transformations.rotation_matrix(angle=angle, direction=direction, point=obj_centroid)
    camera_pose = np.dot(rotation, camera_pose)

    return camera_pose


def center_view_render(graph_raw, graph, resolution, obj_to_mesh, obj):
    # extract the scene, camera pose and room dimensions.
    object_scene, context_scene, camera_pose, room_dimension = prepare_scene(graph_raw, obj_to_mesh,
                                                                             center_object=obj)

    # adjust the camera to point at the center object with a random angle
    obj_centroid = np.asarray(graph[obj]['obbox'][0])
    camera_pose = adjust_camera_view(camera_pose, obj_centroid, context_scene.centroid)

    # render object and context scenes.
    r = Render(rendering_kwargs)
    obj_img = r.center_view_render(object_scene, resolution, camera_pose, room_dimension)
    context_img = r.center_view_render(context_scene, resolution, camera_pose, room_dimension)

    return obj_img, context_img, camera_pose


def process_scenes(scene_names, models_dir, scene_graph_dir, scene_graph_dir_raw, resolution=(224, 224)):
    t = time()
    idx = 0
    scene_to_pos_negatives = {}
    for scene_name in scene_names:
        # for each scene build scene graphs
        idx += 1
        print('Processing scene {} ... '.format(scene_name))
        print('Iteration {}/{}'.format(idx, len(scene_names)))
        t2 = time()
        if scene_name in visited:
            continue

        # load the graphs representing the scene
        graph = load_from_json(os.path.join(scene_graph_dir, scene_name + '.json'))
        graph_raw = load_from_json(os.path.join(scene_graph_dir_raw, scene_name + '.json'))
        objects = graph.keys()

        # for each object in the graph select a number of negative objects.
        pos_obj_to_negatives = {}
        for pos_obj in objects:
            # sample the negative objects based on their IoU with the center object
            pos_obj_to_negatives[pos_obj] = sample_negative_objects(graph, pos_obj)

        # load and collect the meshes for all objects in the scene once.
        obj_to_mesh = {}
        for obj in graph_raw.keys():
            obj_to_mesh[obj] = prepare_mesh_for_scene(models_dir, graph_raw, obj)

        # create a rendering view for each positive object and its corresponding negative objects.
        seen_negative = set()
        scene_to_pos_negatives[scene_name] = {'pos_to_negatives': {}, 'camera_pose': {}}
        for pos_obj, negatives in pos_obj_to_negatives.items():
            # render the positive object and save the object and context img for that.
            obj_img, context_img, camera_pose = center_view_render(graph_raw, graph, resolution, obj_to_mesh, pos_obj)

            # save the rendered images for the positive object
            pos_img_name = pos_obj + '.png'
            Image.fromarray(obj_img).save(os.path.join(data_output, scene_name, 'positives', 'object_images',
                                                       pos_img_name))
            Image.fromarray(context_img).save(os.path.join(data_output, scene_name, 'positives', 'context_images',
                                                           pos_img_name))

            # render the negatives that you haven't rendered before
            for negative in negatives:
                if negative not in seen_negative:
                    # render the positive object and save the object and context img for that.
                    obj_img, context_img, _ = center_view_render(graph_raw, graph, resolution, obj_to_mesh, negative)
                    seen_negative.add(negative)

                    # save the rendered images for the positive object
                    neg_img_name = negative + '.png'
                    Image.fromarray(obj_img).save(os.path.join(data_output, scene_name, 'negatives',
                                                               'object_images', neg_img_name))
                    Image.fromarray(context_img).save(os.path.join(data_output, scene_name, 'negatives',
                                                                   'context_images', neg_img_name))

            # record the negatives assigned for each positive in this scene and the camera pose.
            scene_to_pos_negatives[scene_name]['pos_to_negatives'][pos_obj] = negatives
            scene_to_pos_negatives[scene_name]['camera_pose'][pos_obj] = camera_pose.tolist()

        duration_house = (time() - t2) / 60
        print('Scene {} took {} minutes to process'.format(scene_name, round(duration_house, 2)))
        print('-' * 50)

    duration_all = (time() - t) / 60
    print('Processing {} scenes took {} minutes'.format(idx, round(duration_all, 2)))

    return scene_to_pos_negatives


def main(num_chunks, chunk_idx):
    make_scene_folders = False
    build_data = False
    combine_data = False

    # define the paths
    models_dir = '../../data/matterport3d/models'
    scene_graph_dir = '../../results/matterport3d/LearningBased/scene_graphs_cl/all'
    scene_graph_dir_raw = '../../results/matterport3d/LearningBased/scene_graphs/all'
    scene_to_pos_negatives_path = '../../results/matterport3d/LearningBased/scene_to_pos_negatives_{}.json'.format(chunk_idx)

    # make a folder for each scene with sub-folders containing the positives and negatives
    if make_scene_folders:
        graph_names = os.listdir(scene_graph_dir)
        for graph_name in graph_names:
            scene_name = graph_name.split('.')[0]
            # create folder for the scene
            os.mkdir(os.path.join(data_output, scene_name))
            # create sub-folders for the positive object and context images.
            os.makedirs(os.path.join(data_output, scene_name, 'positives', 'object_images'))
            os.makedirs(os.path.join(data_output, scene_name, 'positives', 'context_images'))
            # create sub-folders for the negative object and context images.
            os.makedirs(os.path.join(data_output, scene_name, 'negatives', 'object_images'))
            os.makedirs(os.path.join(data_output, scene_name, 'negatives', 'context_images'))

    if build_data:
        # process the scenes in batches
        scene_names = os.listdir(data_output)
        # scene_names = ['1pXnuDYAj8r_room18']
        chunk_size = int(np.ceil(len(scene_names) / num_chunks))
        scene_to_pos_negatives = process_scenes(scene_names[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size],
                                                models_dir, scene_graph_dir, scene_graph_dir_raw)
        # save the negatives assigned to each positive data
        write_to_json(scene_to_pos_negatives, scene_to_pos_negatives_path)

    if combine_data:
        data_dir = '../../results/matterport3d/LearningBased'
        common_name = 'scene_to_pos_negatives'
        combined_scene_to_pos_negatives = {}
        file_names = [e for e in os.listdir(data_dir) if common_name in e]
        for file_name in file_names:
            scene_to_pos_negatives_path = os.path.join(data_dir, file_name)
            scene_to_pos_negatives = load_from_json(scene_to_pos_negatives_path)
            for scene_name, info in scene_to_pos_negatives.items():
                combined_scene_to_pos_negatives[scene_name] = info

        # save the combined data
        write_to_json(combined_scene_to_pos_negatives, os.path.join(data_dir, common_name+'.json'))


if __name__ == '__main__':
    rendering_kwargs = {'fov': np.pi / 6, 'light_directional_intensity': 0.01, 'light_point_intensity_center': 0.0,
                        'wall_thickness': 5}
    data_output = '../../results/matterport3d/LearningBased/positive_negative_imgs'
    if not os.path.exists(data_output):
        os.makedirs(data_output)
    visited = set()
    if len(sys.argv) == 1:
        main(1, 0)
    else:
        # To run in parallel you can use the command:
        # export PYTHONPATH="${PYTHONPATH}:/home/reza/Documents/research/3DSSR"
        # parallel -j5 "python3 -u build_pos_neg_imgs.py main {1} {2}" ::: 5 ::: 0 1 2 3 4
        main(int(sys.argv[2]), int(sys.argv[3]))
