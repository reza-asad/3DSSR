import os
import sys
import numpy as np
from time import time

from base_scene import BaseScene
from scripts.helper import write_to_json, create_train_val_test


def process_scenes(scene_files_chunk):
    t = time()
    idx = 0
    for scene_name in scene_files_chunk:
        # for each scene build scene graphs
        print('Processing scene {} ... '.format(scene_name))
        print('Iteration {}/{}'.format(idx, len(scene_files_chunk)))
        t2 = time()

        seen = os.listdir(scene_dir)
        seen = [e.replace('.json', '') for e in seen]
        seen = set(seen)
        if scene_name in seen:
            continue

        # first initialize the graph
        scene = BaseScene(models_dir)
        scene.build_from_matterport(scene_name, metadata_path)
        # scene.visualize()

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
    if action == 'build_scenes':
        # process the scenes in batches
        chunk_size = int(np.ceil(len(scene_files) / num_chunks))
        process_scenes(scene_files[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size])

    if action == 'split_train_test_val':
        scene_graph_dir = os.path.join(data_dir, 'scenes')
        train_path = os.path.join(data_dir, train_split_name)
        val_path = os.path.join(data_dir, val_split_name)
        test_path = os.path.join(data_dir, test_split_name)
        create_train_val_test(scene_graph_dir, train_path, val_path, test_path, split_char=None)


if __name__ == '__main__':
    # define the paths
    data_dir = '../data/scannet'
    room_dir = os.path.join(data_dir, 'rooms')
    scene_files = os.listdir(room_dir)
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    models_dir = os.path.join(data_dir, 'models')
    scene_dir = os.path.join(data_dir, 'scenes', 'all')
    if not os.path.exists(scene_dir):
        try:
            os.makedirs(scene_dir)
        except FileExistsError:
            pass

    train_split_name = 'scenes_train.txt'
    val_split_name = 'scenes_val.txt'
    test_split_name = 'scenes_test.txt'

    if len(sys.argv) == 1:
        main(1, 0, 'build_scenes')
    elif len(sys.argv) == 2:
        main(1, 0, sys.argv[1])
    else:
        # To run in parallel you can use the command:
        # parallel -j5 "python3 -u build_scenes.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: build_scenes
        main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
