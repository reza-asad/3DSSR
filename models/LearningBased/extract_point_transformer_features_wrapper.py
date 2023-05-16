import os
import shutil
import numpy as np
import argparse
from subprocess import Popen
from time import time, sleep


def get_args():
    parser = argparse.ArgumentParser('Extract or Sample Point Clouds', add_help=False)
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--mode', default='train', help='train | val | test')
    parser.add_argument('--accepted_cats_path', default='../../data/{}/accepted_cats.json')
    parser.add_argument('--metadata_path', default='../../data/{}/metadata.csv')
    parser.add_argument('--pc_dir', default='../../data/{}/pc_region_crops')
    parser.add_argument('--cp_dir', default='../../results/{}/LearningBased/')
    parser.add_argument('--results_folder_name', default='3D_DINO_full')
    parser.add_argument('--features_dir_name', default='features', type=str)
    parser.add_argument('--classifier_type', default='DINO', help='supervised | DINO')
    parser.add_argument('--pretrained_weights_name', default='checkpoint0200.pth', type=str,
                        help="Name of the pretrained model.")
    parser.add_argument('--max_coord', default=3.65, type=float, help='3.65 for MP3D')
    parser.add_argument('--theta', default=0, type=int)

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

    crop_folders = [e for e in os.listdir(args.pc_dir.format(args.dataset)) if 'crop' in e]
    crop_folders = sorted(crop_folders, key=lambda x: int(x.split('_')[-1]))
    for i, crop_folder in enumerate(crop_folders):
        pc_dir = os.path.join(args.pc_dir, crop_folder)
        # render the results in parallel
        command = 'python -m torch.distributed.launch --nproc_per_node=1 extract_point_transformer_features_v2.py ' \
                  '--dataset {dataset} --mode {mode} --accepted_cats_path {accepted_cats_path} ' \
                  '--metadata_path {metadata_path} --pc_dir {pc_dir} --features_dir_name {features_dir_name} ' \
                  '--classifier_type {classifier_type} --results_folder_name {results_folder_name} ' \
                  '--pretrained_weights_name {pretrained_weights_name} --max_coord {max_coord} --theta {theta}'
        command = command.format(dataset=args.dataset,
                                 mode=args.mode,
                                 accepted_cats_path=args.accepted_cats_path,
                                 metadata_path=args.metadata_path,
                                 pc_dir=pc_dir,
                                 features_dir_name=args.features_dir_name + '_{}'.format(i+1),
                                 classifier_type=args.classifier_type,
                                 results_folder_name=args.results_folder_name,
                                 pretrained_weights_name=args.pretrained_weights_name,
                                 max_coord=args.max_coord,
                                 theta=args.theta)

        process_sampling = Popen(command, shell=True)
        timeit(process_sampling, 'Encoding crops', sleep_time=60)

    # move the features to the crop region directory.
    features_dirs = [args.features_dir_name + '_{}'.format(i+1) for i in range(len(crop_folders))]
    for features_dir in features_dirs:
        src = os.path.join(args.cp_dir.format(args.dataset), args.results_folder_name, features_dir)
        dest = os.path.join(args.pc_dir.format(args.dataset), features_dir)
        shutil.move(src, dest)


if __name__ == '__main__':
    main()



