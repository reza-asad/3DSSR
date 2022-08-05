import numpy as np
import argparse
from subprocess import Popen
from time import time, sleep


def get_args():
    parser = argparse.ArgumentParser('Extract or Sample Point Clouds', add_help=False)
    parser.add_argument('--action', default='sample_crop', help='extract | sample_crop')
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--mode', default='train', help='train | val | test')
    parser.add_argument('--mesh_regions_dir', default='../data/{}/mesh_regions')
    parser.add_argument('--pc_dir', default='../data/{}/pc_regions')
    parser.add_argument('--results_dir', default='../data/{}/pc_region_crops')
    parser.add_argument('--results_folder_name', default='crops')
    parser.add_argument('--num_points', default=4096, type=int, help='number of points randomly sampled form the pc.')
    parser.add_argument('--num_runs', default=20, type=int, help='number of times running the script')

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
    parser = argparse.ArgumentParser('Extract or Sample Point Clouds', parents=[get_args()])
    args = parser.parse_args()

    if args.action == 'sample_crop':
        for i in range(args.num_runs):
            # render the results in parallel
            command_1 = 'parallel -j5 python3 -u extract_point_clouds.py --num_chunks {1} --chunk_idx {2} ' \
                        '--action {3} --dataset {4} --mode {5} --pc_dir {6} --results_dir {7} ' \
                        '--results_folder_name {8} --num_points {9} --seed {10} ::: 5 ::: ' \
                        '0 1 2 3 4 ::: '
            command_2 = '{action} ::: {dataset} ::: {mode} ::: {pc_dir} ::: {results_dir} ::: {results_folder_name} ' \
                        '::: {num_points} ::: {seed}'.format(action=args.action,
                                                             dataset=args.dataset,
                                                             mode=args.mode,
                                                             pc_dir=args.pc_dir,
                                                             results_dir=args.results_dir,
                                                             results_folder_name=args.results_folder_name + '_{}'.format(i+1),
                                                             num_points=args.num_points,
                                                             seed=i)
            process_sampling = Popen(command_1 + command_2, shell=True)
            timeit(process_sampling, 'Sampling and cropping', sleep_time=60)


if __name__ == '__main__':
    main()



