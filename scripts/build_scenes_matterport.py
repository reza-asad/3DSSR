import os
import sys
import numpy as np
from time import time

from base_scene import BaseScene
from scripts.helper import load_from_json, write_to_json, create_train_val_test


def process_scenes(scene_files, metadata_path, models_dir, scene_dir, accepted_cats):
    t = time()
    idx = 0
    for scene_name in scene_files:
        # for each scene build scene graphs
        print('Processing scene {} ... '.format(scene_name))
        print('Iteration {}/{}'.format(idx, len(scene_files)))
        t2 = time()

        seen = os.listdir(scene_dir)
        seen = [e.replace('.json', '') for e in seen]
        seen = set(seen)
        if scene_name in seen:
            continue

        # first initialize the graph
        scene = BaseScene(models_dir, accepted_cats)
        scene.build_from_matterport(scene_name, metadata_path)

        # save the scene recipe
        scene_graph_path = os.path.join(scene_dir, scene_name+'.json')
        if len(scene.graph) > 0:
            write_to_json(scene.graph, scene_graph_path)

        duration_house = (time() - t2) / 60
        print('Scene {} took {} minutes to process'.format(scene_name, round(duration_house, 2)))
        print('-' * 50)

    duration_all = (time() - t) / 60
    print('Processing {} scenes took {} minutes'.format(idx, round(duration_all, 2)))


def main(num_chunks, chunk_idx, action='build_scenes'):
    # define the paths
    room_dir = '../data/matterport3d/rooms'
    scene_files = os.listdir(room_dir)
    metadata_path = '../data/matterport3d/metadata.csv'
    models_dir = '../data/matterport3d/models'
    scene_dir = '../data/matterport3d/scenes/all'
    accepted_cats = load_from_json('../data/matterport3d/accepted_cats.json')
    if not os.path.exists(scene_dir):
        try:
            os.makedirs(scene_dir)
        except FileExistsError:
            pass

    if action == 'build_scenes':
        # process the scenes in batches
        chunk_size = int(np.ceil(len(scene_files) / num_chunks))
        process_scenes(scene_files[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size],
                       metadata_path,
                       models_dir,
                       scene_dir,
                       accepted_cats)
    if action == 'split_train_test_val':
        data_dir = '../data/matterport3d'
        scene_graph_dir = '../data/matterport3d/scenes'
        train_path = os.path.join(data_dir, 'scenes_train.txt')
        val_path = os.path.join(data_dir, 'scenes_val.txt')
        test_path = os.path.join(data_dir, 'scenes_test.txt')
        create_train_val_test(scene_graph_dir, train_path, val_path, test_path)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main(1, 0, 'build_scenes')
    elif len(sys.argv) == 2:
        main(1, 0, sys.argv[1])
    else:
        # To run in parallel you can use the command:
        # parallel -j5 "python3 -u build_scenes_matterport.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: build_scenes
        main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
