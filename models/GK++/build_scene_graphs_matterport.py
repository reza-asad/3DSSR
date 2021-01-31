import os
import sys
import numpy as np
from time import time

from build_scene_graphs import SceneGraph
from helper import create_train_val_test


def process_scenes(scene_files, models_dir, metadata_path, scene_graph_dir, test_objects, num_samples, dist_eps,
                   angle_eps):
    t = time()
    idx = 0
    for scene_name in scene_files:
        # for each scene build scene graphs
        print('Processing scene {} ... '.format(scene_name))
        print('Iteration {}/{}'.format(idx, len(scene_files)))
        t2 = time()

        seen = os.listdir(scene_graph_dir)
        seen = [e.replace('.json', '') for e in seen]
        seen = set(seen)
        if scene_name in seen:
            continue

        # first initialize the graph
        scene_graph = SceneGraph(models_dir, scene_graph_dir, scene_name)
        scene_graph.build_from_matterport(scene_name, metadata_path)

        # visualize the scenes
        # scene_graph.visualize(graph=scene_graph.graph)

        # derive scene graphs and save them
        scene_graph.build_scene_graph(test_objects, num_samples=num_samples, chunk_size=500, dist_eps=dist_eps,
                                      angle_eps=angle_eps, room_cats=['wall', 'unknown', 'roof', 'ceiling'],
                                      bad_object_cats=['remove', 'delete', 'void'])
        if len(scene_graph.graph) > 0:
            scene_graph.to_json()

        duration_house = (time() - t2) / 60
        print('Scene {} took {} minutes to process'.format(scene_name, round(duration_house, 2)))
        print('-' * 50)

    duration_all = (time() - t) / 60
    print('Processing {} scenes took {} minutes'.format(idx, round(duration_all, 2)))


def main(num_chunks, chunk_idx):
    build_scene_graphs = False
    split_train_test_val = True

    # define the paths
    room_dir = '../../data/matterport3d/rooms'
    scene_files = os.listdir(room_dir)
    # scene_files = ['1pXnuDYAj8r_room17', '1pXnuDYAj8r_room16', '1pXnuDYAj8r_room15']
    test_objects = []
    models_dir = '../../data/matterport3d/models'
    metadata_path = '../../data/matterport3d/metadata.csv'
    scene_graph_dir = '../../results/matterport3d/GK++/scene_graphs/all'

    if build_scene_graphs:
        # process the houses in batches
        chunk_size = int(np.ceil(len(scene_files) / num_chunks))
        process_scenes(scene_files[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size],
                       models_dir,
                       metadata_path,
                       scene_graph_dir,
                       test_objects=test_objects,
                       num_samples=1000,
                       dist_eps=0.1,
                       angle_eps=1)

    if split_train_test_val:
        data_dir = '../../data/matterport3d'
        scene_graph_dir = '../../results/matterport3d/GK++/scene_graphs'
        train_path = os.path.join(data_dir, 'scenes_train.txt')
        val_path = os.path.join(data_dir, 'scenes_val.txt')
        test_path = os.path.join(data_dir, 'scenes_test.txt')
        create_train_val_test(scene_graph_dir, train_path, val_path, test_path)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main(1, 0)
    else:
        # To run in parallel you can use the command:
        # parallel -j5 "python3 -u build_scene_graphs_matterport.py main {1} {2}" > test_logs.txt ::: 5 ::: 0 1 2 3 4
        main(int(sys.argv[2]), int(sys.argv[3]))
