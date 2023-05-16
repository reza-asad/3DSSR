import os
import copy
import argparse
from time import time
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from models.LearningBased.region_dataset import Region
from models.LearningBased.region_dataset_normalized_crop import Region as RegionNorm
from models.LearningBased.transformations import PointcloudScale, PointcloudJitter, PointcloudTranslate, PointcloudRotatePerturbation
from models.LearningBased.transformer_models import PointTransformerSeg
from models.LearningBased.projection_models import DINOHead
from scripts.helper import load_from_json
import models.LearningBased.utils as utils

alpha = 1
gamma = 2.


def compute_accuracy(per_class_accuracy, predicted_labels, labels, cat_to_idx):
    for c, (num_correct, num_total) in per_class_accuracy.items():
        c_idx = cat_to_idx[c]
        labels_c = labels[labels == c_idx]
        predicted_labels_c = predicted_labels[labels == c_idx]

        num_total += len(labels_c)
        num_correct += sum(predicted_labels_c == labels_c)
        per_class_accuracy[c] = (num_correct, num_total)


def evaluate_net(classifier, valid_loader, cat_to_idx):
    # set models to evaluation mode
    classifier.eval()

    total_validation_loss = 0
    per_class_accuracy = {cat: (0, 0) for cat in cat_to_idx.keys()}
    predicted_labels = {}
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            # load data
            file_names = data['file_name']
            pc = data['crops'][0]
            labels = data['labels'].squeeze(dim=1)

            # move data to the right device
            pc = pc.to(dtype=torch.float32).cuda()
            labels = labels.to(dtype=torch.long).cuda()

            # apply the model
            output = classifier(pc)

            # compute loss
            ce_loss = torch.nn.functional.cross_entropy(output, labels, reduction='none')
            pt = torch.exp(-ce_loss)
            loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()
            total_validation_loss += loss.item()

            _, predictions = torch.max(torch.softmax(output, dim=1), dim=1)
            compute_accuracy(per_class_accuracy, predictions, labels, cat_to_idx)

            # map file_names to predicted labels.
            for j, file_name in enumerate(file_names):
                file_name = int(file_name.item())
                predicted_labels[file_name] = predictions[j].item()

    return total_validation_loss/(i + 1), per_class_accuracy, predicted_labels


def train_net(args):
    # create a list of transformations to be applied to the point cloud.
    transformations = []
    if args.aug_rotation:
        transformations.append(PointcloudRotatePerturbation(angle_sigma=0.06, angle_clip=0.18))
    if args.aug_scale:
        transformations.append(PointcloudScale())
    if args.aug_translation:
        transformations.append(PointcloudTranslate())
    if len(transformations) > 0:
        transformations = transforms.Compose(transformations)
    else:
        transformations = None

    # create the training dataset
    if args.crop_normalized:
        train_dataset = RegionNorm(args.pc_dir, args.scene_dir, args.metadata_path, max_coord=args.max_coord,
                                   num_local_crops=0, num_global_crops=0, mode='train', cat_to_idx=args.cat_to_idx,
                                   num_points=args.num_point, transforms=transformations,
                                   file_name_to_idx=args.file_name_to_idx)
        val_dataset = RegionNorm(args.pc_dir, args.scene_dir, args.metadata_path, max_coord=args.max_coord,
                                 num_local_crops=0, num_global_crops=0, mode='val', cat_to_idx=args.cat_to_idx,
                                 num_points=args.num_point, file_name_to_idx=args.file_name_to_idx)
    else:
        train_dataset = Region(args.pc_dir, args.scene_dir, args.metadata_path, num_local_crops=0, num_global_crops=0,
                               mode='train', cat_to_idx=args.cat_to_idx, num_points=args.num_point,
                               transforms=transformations, file_name_to_idx=args.file_name_to_idx)
        val_dataset = Region(args.pc_dir, args.scene_dir, args.metadata_path, num_local_crops=0, num_global_crops=0,
                             mode='val', cat_to_idx=args.cat_to_idx, num_points=args.num_point,
                             file_name_to_idx=args.file_name_to_idx)

    # create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    # load the model
    backbone = PointTransformerSeg(args)
    head = DINOHead(in_dim=args.transformer_dim, out_dim=args.num_class, use_bn=args.use_bn_in_head,
                    norm_last_layer=args.norm_last_layer)
    classifier = utils.DINO(backbone, head, num_local_crops=0, num_global_crops=1, network_type='teacher')

    classifier = torch.nn.DataParallel(classifier).cuda()
    classifier.train()

    # # check if training is form scratch or a best model
    # try:
    #     checkpoint = torch.load(os.path.join(args.cp_dir, args.best_model_name))
    #     start_epoch = checkpoint['epoch']
    #     classifier.load_state_dict(checkpoint['model_state_dict'])
    #     print('Use pretrain model')
    # except FileNotFoundError:
    #     print('No existing model, starting training from scratch...')
    #     start_epoch = 0

    # check if you should use pre-training.
    if args.pre_training:
        checkpoint_path = os.path.join(args.cp_dir_pret, args.results_folder_name_pret, args.pre_training_checkpoint)
        utils.load_pretrained_weights(classifier.module.backbone, checkpoint_path, 'teacher')

    # define the optimizer and learning rate scheduler.
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)

    # track losses and use that along with patience to stop early.
    global_epoch = 0
    global_step = 0
    training_losses = []
    validation_losses = []
    best_validation_loss = float('inf')
    best_epoch = 0
    best_accuracy = 0

    print('Start training...')
    for epoch in range(0, args.epochs):
        t = time()
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epochs))
        epoch_loss = 0

        for i, data in enumerate(train_loader):
            # load data
            pc = data['crops'][0]
            labels = data['labels'].squeeze(dim=1)

            # move data to the right device
            pc = pc.to(dtype=torch.float32).cuda()
            labels = labels.to(dtype=torch.long).cuda()

            # apply the model
            output = classifier(pc)

            # compute loss
            ce_loss = torch.nn.functional.cross_entropy(output, labels, reduction='none')
            pt = torch.exp(-ce_loss)
            loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            # keep track of the losses
            epoch_loss += loss.item()

        scheduler.step()
        print('Epoch %d finished! - Loss: %.10f' % (epoch + 1, epoch_loss / (i + 1)))
        t2 = time()
        print("Training epoch took %.3f minutes" % ((t2 - t) / 60))

        # evaluate the model on the validation dataset.
        validation_loss, accuracy, _ = evaluate_net(classifier, val_loader, args.cat_to_idx)
        validation_losses.append(validation_loss)
        print('Evaluated after epoch {} ...'.format(epoch + 1))
        print('***Total validation loss: {}'.format(validation_loss))
        for c, (num_correct, num_total) in accuracy.items():
            if num_total > 0:
                print('class {}: {}'.format(c, float(num_correct) / num_total))

        if validation_loss < best_validation_loss:
            state = {
                'epoch': epoch + 1,
                'accuracy': accuracy,
                'validation_loss': validation_loss,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, os.path.join(args.cp_dir, 'CP_best.pth'))
            best_validation_loss = validation_loss
            best_accuracy = accuracy
            best_epoch = epoch

        # set the models back to training mode
        classifier.train()

        # save model from every nth epoch
        if args.save_cp and ((epoch + 1) % 20 == 0):
            state = {
                'epoch': epoch + 1,
                'accuracy': accuracy,
                'validation_loss': validation_loss,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, os.path.join(args.cp_dir, 'CP_{}.pth'.format(epoch + 1)))
            print('Checkpoint %d saved !' % (epoch + 1))
        training_losses.append(epoch_loss / (i + 1))
        global_epoch += 1
        print('-' * 50)

    # save train/valid losses
    if args.save_cp:
        np.save(os.path.join(args.cp_dir, 'training_loss'), training_losses)
        np.save(os.path.join(args.cp_dir, 'valid_loss'), validation_losses)

    # report the attributes for the best model
    print('Best Validation loss is {}'.format(best_validation_loss))
    print('Best model comes from epoch: {}'.format(best_epoch + 1))
    print('Best accuracy on validation: ')
    for c, (num_correct, num_total) in best_accuracy.items():
        if num_total > 0:
            print('class {}: {}'.format(c, float(num_correct)/num_total))


def get_args():
    parser = argparse.ArgumentParser('Point Transformer Classification', add_help=False)

    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--accepted_cats_path', default='../../data/{}/accepted_cats_top10.json')
    parser.add_argument('--metadata_path',  default='../../data/{}/metadata_non_equal_full_top10.csv')
    parser.add_argument('--pc_dir', default='../../data/{}/pc_regions')
    parser.add_argument('--scene_dir', default='../../data/{}/scenes')
    parser.add_argument('--cp_dir', default='../../results/{}/LearningBased/')
    parser.add_argument('--cp_dir_pret', default='../../results/{}/LearningBased/')
    parser.add_argument('--results_folder_name', default='region_classification_transformer_test')
    parser.add_argument('--results_folder_name_pret', default='3D_DINO_regions_non_equal_full_top10')
    parser.add_argument('--best_model_name', default='CP_best.pth')

    parser.add_argument('--num_point', default=4096, type=int)
    parser.add_argument('--nblocks', default=2, type=int)
    parser.add_argument('--nneighbor', default=16, type=int)
    parser.add_argument('--input_dim', default=3, type=int)
    parser.add_argument('--transformer_dim', default=32, type=int)
    parser.add_argument('--crop_normalized', default=False, type=utils.bool_flag)
    parser.add_argument('--max_coord', default=14.30, type=float, help='14.30 for MP3D| 5.37 for scannet| '
                                                                     '5.02 for shapenetsem')
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag)
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag)

    parser.add_argument('--aug_rotation', action='store_true', default=False)
    parser.add_argument('--aug_translation', action='store_true', default=False)
    parser.add_argument('--aug_scale', action='store_true', default=False)

    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pre_training', default=False, type=utils.bool_flag)
    parser.add_argument('--pre_training_checkpoint', default='checkpoint0010.pth', help='pre-training checkpoint file name')
    parser.add_argument('--save_cp', action='store_true', default=True, help='save the trained models')

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

    # find a mapping from the region files to their indices.
    df = pd.read_csv(args.metadata_path)
    is_accepted = df['mpcat40'].apply(lambda x: x in args.cat_to_idx)
    df = df.loc[is_accepted]
    df = df.loc[(df['split'] == 'train') | (df['split'] == 'val')]
    file_names = df[['room_name', 'objectId']].apply(lambda x: '-'.join([str(x[0]), str(x[1]) + '.npy']), axis=1).tolist()
    file_names = sorted(file_names)
    file_name_to_idx = {file_name: i for i, file_name in enumerate(file_names)}
    args.file_name_to_idx = file_name_to_idx

    # time the training
    t = time()
    train_net(args)
    t2 = time()
    print("Training took %.3f minutes" % ((t2 - t) / 60))


if __name__ == '__main__':
    main()



