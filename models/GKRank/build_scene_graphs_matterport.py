import os
import sys
import pandas as pd
import numpy as np
from time import time

from scene_graphs import SceneGraph
from helper import create_train_val_test, load_from_json


def process_scenes(scene_files, models_dir, metadata_path, accepted_cats, scene_graph_dir, test_objects, num_samples,
                   dist_eps, angle_eps):
    t = time()
    idx = 0
    for scene_name in scene_files:
        # for each scene build scene graphs
        print('Processing scene {} ... '.format(scene_name))
        print('Iteration {}/{}'.format(idx, len(scene_files)))
        t2 = time()

        if scene_name in visited:
            continue

        # first initialize the graph
        scene_graph = SceneGraph(models_dir, scene_graph_dir, scene_name, accepted_cats)
        # add node attributes
        scene_graph.build_from_matterport(scene_name, metadata_path)
        # filter the graph to only contain the objects with accepted category
        scene_graph.filter_by_accepted_cats()

        if len(scene_graph.graph) == 0:
            print('Skipped scene {} as there are 0 object there ... '.format(scene_name))
            continue
        # derive scene graphs and save them
        scene_graph.build_scene_graph(test_objects, num_samples=num_samples, chunk_size=500, dist_eps=dist_eps,
                                      angle_eps=angle_eps, room_cats=['wall', 'unknown', 'roof', 'ceiling'],
                                      bad_object_cats=['remove', 'delete', 'void'])
        scene_graph.to_json()
        visited.add(scene_name)

        duration_house = (time() - t2) / 60
        print('Scene {} took {} minutes to process'.format(scene_name, round(duration_house, 2)))
        print('-' * 50)

    duration_all = (time() - t) / 60
    print('Processing {} scenes took {} minutes'.format(idx, round(duration_all, 2)))


def main(num_chunks, chunk_idx, action='build_scene_graphs'):
    # define the paths
    test_objects = []

    if action == 'build_scene_graphs':
        # process the houses in batches
        chunk_size = int(np.ceil(len(scene_files) / num_chunks))
        process_scenes(scene_files[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size],
                       models_dir,
                       metadata_path,
                       accepted_cats,
                       scene_graph_dir,
                       test_objects=test_objects,
                       num_samples=1000,
                       dist_eps=0.1,
                       angle_eps=1)


if __name__ == '__main__':
    mode = 'test'
    models_dir = '../../data/matterport3d/mesh_regions/{}'.format(mode)
    metadata_path = '../../data/matterport3d/metadata.csv'
    accepted_cats = set(load_from_json('../../data/matterport3d/accepted_cats.json'))
    data_dir = '../../data/matterport3d'

    scene_graph_dir = '../../results/matterport3d/GKRank/scene_graphs/{}'.format(mode)
    if not os.path.exists(scene_graph_dir):
        try:
            os.makedirs(scene_graph_dir)
        except FileExistsError:
            pass

    # find the scene names for the requested mode
    df_metadata = pd.read_csv(metadata_path)
    all_scene_names = df_metadata.loc[df_metadata['split'] == mode, 'room_name'].unique().tolist()

    # find the scenes that are already processed (if any)
    visited = os.listdir(scene_graph_dir)
    visited = set([e.replace('.json', '') for e in visited])
    scene_files = [e for e in all_scene_names if e not in visited]

    if len(sys.argv) == 1:
        main(1, 0, 'build_scene_graphs')
    elif len(sys.argv) == 2:
        main(1, 0, sys.argv[1])
    else:
        # To run in parallel you can use the command:
        # parallel -j5 "python3 -u build_scene_graphs_matterport.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: build_scene_graphs
        main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
