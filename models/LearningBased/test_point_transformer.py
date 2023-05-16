import os
import argparse
from time import time
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from region_dataset import Region
from models.LearningBased.region_dataset_normalized_crop import Region as RegionNorm
from transformer_models import PointTransformerSeg
from projection_models import DINOHead
from scripts.helper import load_from_json, write_to_json
from train_point_transformer import evaluate_net
import utils

alpha = 1
gamma = 2.


def run_classifier(args):
    # create the training dataset
    if args.crop_normalized:
        dataset = RegionNorm(args.pc_dir, args.scene_dir, args.metadata_path, max_coord=args.max_coord,
                             num_local_crops=0, num_global_crops=0, mode=args.mode, cat_to_idx=args.cat_to_idx,
                             num_points=args.num_point, file_name_to_idx=args.file_name_to_idx)
    else:
        dataset = Region(args.pc_dir, args.scene_dir, args.metadata_path, num_local_crops=0, num_global_crops=0,
                         mode=args.mode, cat_to_idx=args.cat_to_idx, num_points=args.num_point,
                         file_name_to_idx=args.file_name_to_idx)

    # create the dataloaders
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)

    # load the model
    backbone = PointTransformerSeg(args)
    head = DINOHead(in_dim=args.transformer_dim, out_dim=args.num_class, use_bn=args.use_bn_in_head,
                    norm_last_layer=args.norm_last_layer)
    classifier = utils.DINO(backbone, head, num_local_crops=0, num_global_crops=1, network_type='teacher')
    classifier = torch.nn.DataParallel(classifier).cuda()
    checkpoint = torch.load(os.path.join(args.cp_dir, args.best_model_name))
    classifier.load_state_dict(checkpoint['model_state_dict'])

    classifier.eval()

    # evaluate the model
    validation_loss, per_class_accuracy, predicted_labels = evaluate_net(classifier, loader, args.cat_to_idx)

    # save the predicted label for each file name
    file_name_to_label = {}
    for idx, label in predicted_labels.items():
        file_name = args.idx_to_file_name[idx]
        file_name_to_label[file_name] = label
    write_to_json(file_name_to_label, os.path.join(args.cp_dir, 'predicted_labels_{}.json'.format(args.mode)))

    # save the per class accuracies.
    per_class_accuracy_final = {}
    num_correct_total, num_total = 0, 0
    for c, (num_correct_c, num_total_c) in per_class_accuracy.items():
        if num_total_c != 0:
            accuracy = float(num_correct_c) / num_total_c
            per_class_accuracy_final[c] = accuracy
            print('Per Class Accuracy for {}: {}'.format(c, per_class_accuracy_final[c]))
            num_total += num_total_c
            num_correct_total += num_correct_c

    print('Micro AVG: {}'.format(float(num_correct_total / num_total) * 100))
    macro_avg = np.mean(list(per_class_accuracy_final.values()))
    print('Macro AVG: {}'.format(macro_avg * 100))
    write_to_json(per_class_accuracy_final, os.path.join(args.cp_dir, 'per_class_accuracy_{}.json'.format(args.mode)))


def get_args():
    parser = argparse.ArgumentParser('Point Transformer Classification', add_help=False)

    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--mode', dest='mode', default='val', help='val|test')
    parser.add_argument('--accepted_cats_path', default='../../data/{}/accepted_cats.json')
    parser.add_argument('--pc_dir', dest='pc_dir', default='../../data/{}/pc_regions')
    parser.add_argument('--scene_dir', dest='scene_dir', default='../../results/{}/scenes')
    parser.add_argument('--metadata_path', default='../../data/{}/metadata.csv')
    parser.add_argument('--cp_dir', default='../../results/{}/LearningBased/')
    parser.add_argument('--results_folder_name', default='region_classification_transformer_full')
    parser.add_argument('--best_model_name', dest='best_model_name', default='CP_best.pth')

    parser.add_argument('--num_point', dest='num_point', default=4096, type=int)
    parser.add_argument('--nblocks', dest='nblocks', default=2, type=int)
    parser.add_argument('--nneighbor', dest='nneighbor', default=16, type=int)
    parser.add_argument('--input_dim', dest='input_dim', default=3, type=int)
    parser.add_argument('--transformer_dim', dest='transformer_dim', default=32, type=int)
    parser.add_argument('--batch_size', dest='batch_size', default=8, type=int)
    parser.add_argument('--crop_normalized', action='store_true', default=True)
    parser.add_argument('--max_coord', default=3.65, type=float, help='14.30 MP3D| 5.37 Scannet| 5.02 ShapeNetSem')
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag)
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag)

    return parser


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def main():
    # get the arguments
    parser = argparse.ArgumentParser('Point Transformer Classification', parents=[get_args()])
    args = parser.parse_args()
    adjust_paths(args, exceptions=[])
    args.cp_dir = os.path.join(args.cp_dir, args.results_folder_name)

    # prepare the accepted categories for training.
    accepted_cats = load_from_json(args.accepted_cats_path)
    accepted_cats = sorted(accepted_cats)
    args.cat_to_idx = {cat: i for i, cat in enumerate(accepted_cats)}
    args.num_class = len(args.cat_to_idx)

    # find a mapping from the region files to their indices.
    df = pd.read_csv(args.metadata_path)
    is_accepted = df['mpcat40'].apply(lambda x: x in args.cat_to_idx)
    df = df.loc[is_accepted]
    df = df.loc[(df['split'] == 'train') | (df['split'] == args.mode)]
    file_names = df[['room_name', 'objectId']].apply(lambda x: '-'.join([str(x[0]), str(x[1]) + '.npy']), axis=1).tolist()
    file_names = sorted(file_names)
    file_name_to_idx = {file_name: i for i, file_name in enumerate(file_names)}
    args.file_name_to_idx = file_name_to_idx
    args.idx_to_file_name = {i: file_name for i, file_name in enumerate(file_names)}

    # time the training
    t = time()
    run_classifier(args)
    t2 = time()
    print("Testing took %.3f minutes" % ((t2 - t) / 60))


if __name__ == '__main__':
    main()



