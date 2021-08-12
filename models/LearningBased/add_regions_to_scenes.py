import os
import sys
import numpy as np
from time import time

from scripts.mesh import Mesh
from scripts.helper import load_from_json, write_to_json, create_train_val_test


def filter_by_accepted_cats(scene, accepted_cats):
    filtered_scene = {}
    for obj, obj_info in scene.items():
        if obj_info['category'][0] in accepted_cats:
            filtered_scene[obj] = obj_info

    return filtered_scene


def add_regions(models_dir, scene):
    for obj, obj_info in scene.items():
        # load the mesh region
        file_name = obj_info['file_name']
        mesh = Mesh(os.path.join(models_dir, file_name), transform=obj_info['transform']).load(with_transform=True)

        # find the axis aligned bounding box around the region and record it.
        bbox = mesh.bounding_box
        obj_info['bbox_region'] = [bbox.centroid.tolist()] + bbox.vertices.tolist()

    return scene


def process_scenes(scene_names, models_dir, scene_dir_input, accepted_cats):
    t = time()
    idx = 0
    for scene_name in scene_names:
        # for each scene build scene graphs
        idx += 1
        print('Processing scene {} ... '.format(scene_name))
        print('Iteration {}/{}'.format(idx, len(scene_names)))
        t2 = time()
        if scene_name in visited:
            continue

        # laod the scene
        scene = load_from_json(os.path.join(scene_dir_input, scene_name))

        # filter the graph to only contain the objects with accepted category
        scene = filter_by_accepted_cats(scene, accepted_cats)

        # only proceed with non-empty scenes
        if len(scene) < 1:
            print('Skipped empty scene {} '.format(scene_name))
            continue

        # add the region to the scene
        scene = add_regions(models_dir, scene)

        # save the scene recipe
        scene_path = os.path.join(scene_dir_output_all, scene_name)
        write_to_json(scene, scene_path)
        visited.add(scene_name)

        duration_house = (time() - t2) / 60
        print('Scene {} took {} minutes to process'.format(scene_name, round(duration_house, 2)))
        print('-' * 50)

    duration_all = (time() - t) / 60
    print('Processing {} scenes took {} minutes'.format(idx, round(duration_all, 2)))


def main(num_chunks, chunk_idx, action='build_scenes'):
    # define the paths
    data_dir = '../../data/matterport3d'
    models_dir = os.path.join(data_dir, 'mesh_regions')
    scene_dir_input = os.path.join(data_dir, 'scenes', 'all')

    if action == 'add_regions':
        # load accepted categories
        accepted_cats = set(load_from_json(os.path.join(data_dir, 'accepted_cats.json')))
        # process the scenes in batches
        scene_names = os.listdir(scene_dir_input)
        chunk_size = int(np.ceil(len(scene_names) / num_chunks))
        process_scenes(scene_names[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size], models_dir,
                       scene_dir_input, accepted_cats)

    if action == 'split_train_test_val':
        train_path = os.path.join(data_dir, 'scenes_train.txt')
        val_path = os.path.join(data_dir, 'scenes_val.txt')
        test_path = os.path.join(data_dir, 'scenes_test.txt')
        create_train_val_test(scene_dir_output, train_path, val_path, test_path)


if __name__ == '__main__':
    scene_dir_output = '../../results/matterport3d/LearningBased/scenes_with_regions'
    scene_dir_output_all = os.path.join(scene_dir_output, 'all')
    if not os.path.exists(scene_dir_output_all):
        try:
            os.makedirs(scene_dir_output_all)
        except FileExistsError:
            pass
        visited = set()
    else:
        visited = set(os.listdir(scene_dir_output_all))
    if len(sys.argv) == 1:
        main(1, 0, 'add_regions')
    elif len(sys.argv) == 2:
        main(1, 0, sys.argv[1])
    else:
        # To run in parallel you can use the command:
        # parallel -j5 "python3 -u add_regions_to_scenes.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: add_regions
        main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
