import os
import sys
import numpy as np
from time import time

from build_scene_graphs import SceneGraph
from scripts.helper import load_from_json, write_to_json, visualize_scene, visualize_graph, create_train_val_test


def process_scenes(scene_names, models_dir, scene_graph_dir_input, graphviz_dir, accepted_cats, colormap):
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
        # objects = ['18', '31']
        scene_graph.build_graph()

        # visualize the scenes
        # graphviz_path = os.path.join(graphviz_dir, scene_name.split('.')[0])
        # visualize_graph(scene_graph.graph, graphviz_path, colormap=colormap, accepted_cats=accepted_cats,
        #                 add_label_id=True, with_fc=False)
        # visualize_scene(scene_graph_dir_input, models_dir, scene_name, accepted_cats=accepted_cats, objects=objects,
        #                 with_backbone=True, as_obbox=True)

        # save the scene recipe
        scene_graph_path = os.path.join(scene_graph_dir_output, scene_name)
        write_to_json(scene_graph.graph, scene_graph_path)

        duration_house = (time() - t2) / 60
        print('Scene {} took {} minutes to process'.format(scene_name, round(duration_house, 2)))
        print('-' * 50)

    duration_all = (time() - t) / 60
    print('Processing {} scenes took {} minutes'.format(idx, round(duration_all, 2)))


def main(num_chunks, chunk_idx):
    build_scenes = False
    split_train_test_val = True

    # define the paths
    models_dir = '../../data/matterport3d/models'
    scene_graph_dir_input = '../../data/matterport3d/scene_graphs/all'
    graphviz_dir = '../../results/matterport3d/GNN/graphviz_samples'
    if not os.path.exists(graphviz_dir):
        os.makedirs(graphviz_dir)

    if build_scenes:
        # load accepted categories
        accepted_cats = set(load_from_json('../../data/matterport3d/accepted_cats.json'))
        colormap = load_from_json('../../data/matterport3d/color_map.json')
        # process the scenes in batches
        scene_names = os.listdir(scene_graph_dir_input)
        # scene_names = ['1pXnuDYAj8r_room18.json']
        # scene_names = ['8194nk5LbLH_room3.json']
        # scene_names = ['ZMojNkEp431_room14.json']
        # scene_names = ['uNb9QFRL6hY_room11.json']
        # scene_names = ['yqstnuAEVhm_room24.json']
        chunk_size = int(np.ceil(len(scene_names) / num_chunks))
        process_scenes(scene_names[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size], models_dir,
                       scene_graph_dir_input, graphviz_dir, accepted_cats, colormap)

    if split_train_test_val:
        data_dir = '../../data/matterport3d'
        scene_graph_dir = '../../results/matterport3d/GNN/scene_graphs'
        train_path = os.path.join(data_dir, 'scenes_train.txt')
        val_path = os.path.join(data_dir, 'scenes_val.txt')
        test_path = os.path.join(data_dir, 'scenes_test.txt')
        create_train_val_test(scene_graph_dir, train_path, val_path, test_path)


if __name__ == '__main__':
    scene_graph_dir_output = '../../results/matterport3d/GNN/scene_graphs/all'
    if not os.path.exists(scene_graph_dir_output):
        os.makedirs(scene_graph_dir_output)
        visited = set()
    else:
        visited = set(os.listdir(scene_graph_dir_output))
    if len(sys.argv) == 1:
        main(1, 0)
    else:
        # To run in parallel you can use the command:
        # export PYTHONPATH="${PYTHONPATH}:/home/reza/Documents/research/3DSSR"
        # parallel -j5 "python3 -u build_scene_graphs_matterport.py main {1} {2}" ::: 5 ::: 0 1 2 3 4
        main(int(sys.argv[2]), int(sys.argv[3]))
