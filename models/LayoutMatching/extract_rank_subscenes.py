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


def apply_3dssr(args):
    args.scene_dir = os.path.join(args.scene_dir, args.mode)

    # set the input and output paths for the query dict.
    query_dict_input_path = os.path.join(args.query_dir, args.query_input_file_name)
    query_output_file_name = args.query_input_file_name.split('.')[0] + '_{}_{}.json'.format(args.mode,
                                                                                             args.experiment_name)
    query_dict_output_path = os.path.join(args.cp_dir, args.results_folder_name, query_output_file_name)

    # read the query dict.
    query_dict = load_from_json(query_dict_input_path)



