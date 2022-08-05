import os
import argparse
from time import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from subscene_dataset import Scene
from models.LearningBased.transformer_models import PointTransformerSeg
from models.LearningBased.projection_models import DINOHead
from scripts.helper import load_from_json, write_to_json
import models.LearningBased.utils as utils
from classifier_subscene import Classifier, Backbone, compute_accuracy


def load_classifier(args):
    # The classifier base model.
    shape_backbone = PointTransformerSeg(args)
    head = DINOHead(in_dim=args.transformer_dim, out_dim=args.num_class, use_bn=args.use_bn_in_head,
                    norm_last_layer=args.norm_last_layer)
    classifier = utils.DINO(shape_backbone, head, num_local_crops=0, num_global_crops=1, network_type='teacher')
    classifier = torch.nn.DataParallel(classifier).cuda()

    # supervised model with no scene context.
    if args.model_type == 'supervised':
        checkpoint_path = os.path.join(args.cp_dir_pret, args.results_folder_name_pret, args.pre_training_checkpoint)
        checkpoint = torch.load(checkpoint_path)
        classifier.load_state_dict(checkpoint['model_state_dict'])
    elif args.model_type == 'context_supervised':
        # arguments for the layout matching model.
        layout_args = argparse.Namespace()
        exceptions = {'input_dim': 35, 'nblocks': 0}
        for k, v in vars(args).items():
            if k in exceptions:
                vars(layout_args)[k] = exceptions[k]
            else:
                vars(layout_args)[k] = v

        # combine the backbones.
        layout_backbone = PointTransformerSeg(layout_args)
        backbone = Backbone(shape_backbone=shape_backbone, layout_backbone=layout_backbone)
        head = DINOHead(in_dim=args.transformer_dim, out_dim=args.num_class, use_bn=args.use_bn_in_head,
                        norm_last_layer=args.norm_last_layer)
        classifier = Classifier(backbone=backbone, head=head)
        classifier = torch.nn.DataParallel(classifier).cuda()

        # load pretrained weights.
        checkpoint_path = os.path.join(args.cp_dir, args.checkpoint)
        checkpoint = torch.load(checkpoint_path)
        classifier.load_state_dict(checkpoint['model_state_dict'])

    else:
        raise NotImplementedError('{} not implemented'.format(args.model_type))

    return classifier


def eval_net(args):
    # create the validation dataset
    dataset = Scene(scene_dir=args.scene_dir, pc_dir=args.pc_dir, metadata_path=args.metadata_path,
                    accepted_cats_path=args.accepted_cats_path, max_coord_box=args.max_coord_box,
                    max_coord_scene=args.max_coord_scene, num_points=args.num_point, num_objects=args.num_objects,
                    mode=args.mode)
    print('{} {} scenes'.format(len(dataset), args.mode))

    # create the dataloader
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

    # load the model.
    classifier = load_classifier(args)
    classifier.eval()

    print('Start evaluating...')
    per_class_accuracy = {cat: (0, 0) for cat in args.cat_to_idx.keys()}
    file_name_to_label = {}
    with torch.no_grad():
        for i, data in enumerate(loader):
            # load data
            pc = data['pc'].squeeze(dim=0)
            centroid = data['centroid'].squeeze(dim=0)
            label = data['label'].squeeze(dim=0)

            # move data to the right device
            pc = pc.to(dtype=torch.float32).cuda()
            centroid = centroid.to(dtype=torch.float32).cuda()
            label = label.to(dtype=torch.long).cuda()

            # apply the model
            if args.model_type == 'supervised':
                output = classifier(pc)
            elif args.model_type == 'context_supervised':
                output = classifier(pc, centroid)
            else:
                raise NotImplementedError('{} not implemented'.format(args.model_type))
            _, predictions = torch.max(torch.softmax(output, dim=1), dim=1)
            compute_accuracy(per_class_accuracy, predictions, label, args.cat_to_idx)

            # record the predictions for each val/test data.
            for j, file_name in enumerate(data['file_name']):
                file_name_to_label[file_name[0]] = predictions[j].item()

    # save the predictions for each val/test data.
    if args.model_type == 'context_supervised':
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
    if args.model_type == 'context_supervised':
        write_to_json(per_class_accuracy_final, os.path.join(args.cp_dir, 'per_class_accuracy_{}.json'.format(args.mode)))


def get_args():
    parser = argparse.ArgumentParser('Point Transformer Classification', add_help=False)
    # Data
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--mode', default='val', help='val | test')
    parser.add_argument('--model_type', default='supervised', help='supervised | context_supervised')
    parser.add_argument('--accepted_cats_path', default='../../data/{}/accepted_cats.json')
    parser.add_argument('--metadata_path',  default='../../data/{}/metadata.csv')
    parser.add_argument('--pc_dir', default='../../data/{}/pc_regions')
    parser.add_argument('--scene_dir', default='../../results/{}/scenes')
    parser.add_argument('--cp_dir', default='../../results/{}/LayoutMatching/')
    parser.add_argument('--cp_dir_pret', default='../../results/{}/LearningBased/')
    parser.add_argument('--results_folder_name', default='exp_supervise_pret')
    parser.add_argument('--results_folder_name_pret', default='region_classification_transformer_full')
    parser.add_argument('--checkpoint', default='CP_best.pth')

    # Model
    parser.add_argument('--num_point', default=4096, type=int)
    parser.add_argument('--nblocks', default=2, type=int)
    parser.add_argument('--nneighbor', default=16, type=int)
    parser.add_argument('--input_dim', default=3, type=int)
    parser.add_argument('--transformer_dim', default=32, type=int)
    parser.add_argument('--max_coord_box', default=3.65, type=float, help='3.65 for MP3D')
    parser.add_argument('--max_coord_scene', default=13.07, type=float, help='13.07 for MP3D')
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag)
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag)

    # Optim
    parser.add_argument('--num_objects', default=10, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pre_training_checkpoint', default='CP_best.pth')

    return parser


def adjust_paths(args):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v):
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def main():
    # get the arguments
    parser = argparse.ArgumentParser('Point Transformer Classification', parents=[get_args()])
    args = parser.parse_args()
    adjust_paths(args)

    # create a directory for checkpoints
    args.cp_dir = os.path.join(args.cp_dir, args.results_folder_name)
    if not os.path.exists(args.cp_dir):
        os.makedirs(args.cp_dir)

    # prepare the accepted categories for training.
    accepted_cats = load_from_json(args.accepted_cats_path)
    accepted_cats = sorted(accepted_cats)
    args.cat_to_idx = {cat: i for i, cat in enumerate(accepted_cats)}
    args.num_class = len(args.cat_to_idx)

    # time the training
    t = time()
    eval_net(args)
    t2 = time()
    print("Training took %.3f minutes" % ((t2 - t) / 60))


if __name__ == '__main__':
    main()



