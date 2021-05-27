import os
import sys
import numpy as np
from time import time

from build_scene_graphs import SceneGraph
from scripts.helper import load_from_json, write_to_json, create_train_val_test


def process_scenes(scene_names, models_dir, scene_graph_dir_input, accepted_cats):
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

        # first initialize the graph
        scene_graph = SceneGraph(scene_graph_dir_input, scene_name, models_dir, accepted_cats, obbox_expansion=1.0)
        # filter the graph to only contain the objects with accepted category
        scene_graph.filter_by_accepted_cats()
        # only proceed with the graph if it has at least two elements (source and neighbour).
        if len(scene_graph.graph) < 2:
            print('Skipped scene {} as there is 1 or 0 object there ... '.format(scene_name))
            continue
        scene_graph.build_graph()

        # save the scene recipe
        scene_graph_path = os.path.join(scene_graph_dir_output, scene_name)
        write_to_json(scene_graph.graph, scene_graph_path)

        duration_house = (time() - t2) / 60
        print('Scene {} took {} minutes to process'.format(scene_name, round(duration_house, 2)))
        print('-' * 50)

    duration_all = (time() - t) / 60
    print('Processing {} scenes took {} minutes'.format(idx, round(duration_all, 2)))


def main(num_chunks, chunk_idx, action='build_scenes'):
    # define the paths
    models_dir = '../../data/matterport3d/models'
    scene_graph_dir_input = '../../data/matterport3d/scene_graphs/all'

    if action == 'build_scenes':
        # load accepted categories
        accepted_cats = set(load_from_json('../../data/matterport3d/accepted_cats.json'))
        # process the scenes in batches
        scene_names = os.listdir(scene_graph_dir_input)
        chunk_size = int(np.ceil(len(scene_names) / num_chunks))
        process_scenes(scene_names[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size], models_dir,
                       scene_graph_dir_input, accepted_cats)

    if action == 'split_train_test_val':
        data_dir = '../../data/matterport3d'
        scene_graph_dir = '../../results/matterport3d/LearningBased/scene_graphs'
        train_path = os.path.join(data_dir, 'scenes_train.txt')
        val_path = os.path.join(data_dir, 'scenes_val.txt')
        test_path = os.path.join(data_dir, 'scenes_test.txt')
        create_train_val_test(scene_graph_dir, train_path, val_path, test_path)


if __name__ == '__main__':
    scene_graph_dir_output = '../../results/matterport3d/LearningBased/scene_graphs/all'
    if not os.path.exists(scene_graph_dir_output):
        os.makedirs(scene_graph_dir_output)
        visited = set()
    else:
        visited = set(os.listdir(scene_graph_dir_output))
    if len(sys.argv) == 1:
        main(1, 0, 'build_scenes')
    elif len(sys.argv) == 2:
        main(1, 0, sys.argv[1])
    else:
        # To run in parallel you can use the command:
        # parallel -j5 "python3 -u build_scene_graphs_matterport.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: build_scenes
        main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
