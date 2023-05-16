import os
import shutil
import numpy as np
import argparse
from subprocess import Popen
from time import time, sleep


def get_args():
    parser = argparse.ArgumentParser('Rendering 3D Subscenes', add_help=False)

    parser.add_argument('--mode', dest='mode', default='val', help='test|val')
    parser.add_argument('--topk', dest='topk', default=10, type=int, help='Number of images rendered for each query.')
    # parser.add_argument('--query_list', dest='query_list', type=str, nargs='+', default=["chair-16"])
    parser.add_argument('--query_list', dest='query_list', type=str, nargs='+',
                        default=["lighting-31", "plant-38", "chair-23", "chair-16", "chair-9"])
    parser.add_argument('--action', default='all', help='make_folders | render | create_img_table | all')

    return parser


def timeit(process, process_name, sleep_time=5):
    t0 = time()
    while process.poll() is None:
        print('{} ...'.format(process_name))
        sleep(sleep_time)
    duration = (time() - t0) / 60
    print('{} Took {} minutes'.format(process_name, np.round(duration, 2)))


def main():
    # get the arguments
    parser = argparse.ArgumentParser('Rendering 3D Subscenes', parents=[get_args()])
    args = parser.parse_args()

    # List of models rendered in the paper
    configs = [
        ('BruteForce', 'BruteForce', None, '../results/matterport3d'),
        # ('RandomRank', 'RandomRank', '', '../results/matterport3d'),
        # ('CatRank', 'CatRank', '', '../results/matterport3d'),
        # ('OracleRankV2', 'OracleRankV2', '', '../results/{}'),
        # ('GKRank', 'GKRank', '', '../results/matterport3d'),
        # ('LearningBased', 'supervised_point_transformer',
        #  'region_classification_transformer_2_32_4096_non_equal_full_region_top10', '../results/matterport3d'),
        # ('LearningBased', '3D_DINO_point_transformer', '3D_DINO_regions_non_equal_full_top10_seg',
        #  '../results/matterport3d'),
        # ('PointTransformerSeg10', 'CSC_point_transformer_10', '',
        #  '/home/reza/Documents/research/ContrastiveSceneContexts/data_processed_subset/matterport3d/embeddings'),
        # ('PointTransformerSeg', 'CSC_point_transformer', '',
        #  '/home/reza/Documents/research/ContrastiveSceneContexts/data_processed_subset/matterport3d/embeddings')
        # ('LearningBased', '3D_DINO_point_transformer_full', '3D_DINO_full', '../results/matterport3d'),
        # ('LearningBased', 'supervised_point_transformer_full', 'region_classification_transformer_full',
        #  '../results/matterport3d')
        # ('embeddings', 'CSC_point_transformer_full', 'PointTransformerSeg_full',
        #  '/home/reza/Documents/research/ContrastiveSceneContexts/data_processed_subset/matterport3d/')
    ]

    for model_name, experiment_name, results_folder_name, cp_dir in configs:
        # delete the existing rendered results
        rendering_path = '../results/matterport3d/rendered_results/{}/{}'.format(args.mode, experiment_name)
        if os.path.exists(rendering_path):
            shutil.rmtree(rendering_path)

        # make folders for rendering each query at scene and cropped scales.
        query_list = ""
        for q in args.query_list:
            query_list = query_list + q + " "
        command_template = 'python3 render_results.py --action {action} --mode {mode} --model_name {model_name} ' \
                           '--experiment_name {experiment_name} --cp_dir {cp_dir} --topk {topk} ' \
                           '--query_list {query_list} --results_folder_name {results_folder_name}'

        if args.action in ['make_folders', 'all']:
            command_make_folders = command_template.format(action='make_folders',
                                                           mode=args.mode,
                                                           model_name=model_name,
                                                           experiment_name=experiment_name,
                                                           cp_dir=cp_dir,
                                                           topk=args.topk,
                                                           query_list=query_list,
                                                           results_folder_name=results_folder_name)

            process_making_folders = Popen(command_make_folders, shell=True)
            timeit(process_making_folders, 'Making Folders')
        if args.action in ['render', 'all']:
            # render the results in parallel
            command_render1 = 'parallel -j5 python3 -u render_results.py --num_chunks {1} --chunk_idx {2} --action {3} ' \
                              '--mode {4} --model_name {5} --experiment_name {6} --cp_dir {7} --topk {8} ' \
                              '--query_list {9} --results_folder_name {10} ::: 5 ::: 0 1 2 3 4 ::: '
            command_render2 = '{action} ::: {mode} ::: {model_name} ::: {experiment_name} ::: {cp_dir} ::: {topk} ' \
                              '::: {query_list} ::: {results_folder_name}'.format(action='render',
                                                                                  mode=args.mode,
                                                                                  model_name=model_name,
                                                                                  experiment_name=experiment_name,
                                                                                  cp_dir=cp_dir,
                                                                                  topk=args.topk,
                                                                                  query_list=query_list,
                                                                                  results_folder_name=results_folder_name)

            process_rendering = Popen(command_render1 + command_render2, shell=True)
            timeit(process_rendering, 'Rendering', sleep_time=20)
        if args.action in ['create_img_table', 'all']:
            # create image tables for the cropped images.
            command_create_img_table = command_template.format(action='create_img_table',
                                                               mode=args.mode,
                                                               model_name=model_name,
                                                               experiment_name=experiment_name,
                                                               cp_dir=cp_dir,
                                                               topk=args.topk,
                                                               query_list=query_list,
                                                               results_folder_name=results_folder_name)
            process_creating_img_table = Popen(command_create_img_table, shell=True)
            timeit(process_creating_img_table, 'Create Img Table')


if __name__ == '__main__':
    main()



