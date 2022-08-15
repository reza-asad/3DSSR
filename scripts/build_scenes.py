import os
import argparse
import numpy as np
import pandas as pd
from time import time

from base_scene import BaseScene
from scripts.helper import write_to_json


def process_scenes(scene_names_chunk):
    t = time()
    idx = 0
    for scene_name in scene_names_chunk:
        # for each scene build scene graphs
        idx += 1
        print('Processing scene {} ... '.format(scene_name))
        print('Iteration {}/{}'.format(idx, len(scene_names_chunk)))
        t2 = time()

        seen = os.listdir(args.scene_dir)
        seen = [e.replace('.json', '') for e in seen]
        seen = set(seen)
        if scene_name in seen:
            continue

        # first initialize the graph
        scene = BaseScene()
        scene.build_from_metadata(scene_name, df_metadata)

        # save the scene recipe
        scene_graph_path = os.path.join(args.scene_dir, scene_name+'.json')
        if len(scene.graph) > 0:
            write_to_json(scene.graph, scene_graph_path)

        duration_house = (time() - t2) / 60
        print('Scene {} took {} minutes to process'.format(scene_name, round(duration_house, 2)))
        print('-' * 50)

    duration_all = (time() - t) / 60
    print('Processing {} scenes took {} minutes'.format(idx, round(duration_all, 2)))


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def get_args():
    parser = argparse.ArgumentParser('Build Scenes', add_help=False)
    parser.add_argument('--dataset', default='scannet')
    parser.add_argument('--mode', default='train', help='train | val')
    parser.add_argument('--metadata_path', default='../data/{}/metadata.csv')
    parser.add_argument('--scene_dir', default='../data/{}/scenes')
    parser.add_argument('--seed', default=0, type=int, help='use different seed for parallel runs')
    parser.add_argument('--num_chunks', default=1, type=int, help='number of chunks for parallel run')
    parser.add_argument('--chunk_idx', default=0, type=int, help='chunk id for parallel run')

    return parser


def main():
    # process the scenes in batches
    chunk_size = int(np.ceil(len(scene_names) / args.num_chunks))
    process_scenes(scene_names[args.chunk_idx * chunk_size: (args.chunk_idx + 1) * chunk_size])


if __name__ == '__main__':
    # read the args
    parser_ = argparse.ArgumentParser('Build Scenes', parents=[get_args()])
    args = parser_.parse_args()
    adjust_paths(args, exceptions=[])

    # load the metadata.
    df_metadata = pd.read_csv(args.metadata_path)
    df_metadata['key'] = df_metadata.apply(lambda x: '-'.join([str(x['room_name']), str(x['objectId'])]), axis=1)
    scene_names = df_metadata.loc[df_metadata['split'] == args.mode, 'room_name'].unique()
    args.scene_dir = os.path.join(args.scene_dir, args.mode)
    if not os.path.exists(args.scene_dir):
        try:
            os.makedirs(args.scene_dir)
        except FileExistsError:
            pass

    main()
    # parallel -j5 "python3 -u build_scenes.py --mode {1} --seed {2} --num_chunks {3} --chunk_idx {4}" ::: val ::: 0 ::: 5 ::: 0 1 2 3 4