import os
import copy
from optparse import OptionParser
from time import time
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
import importlib

from region_dataset import Region
from transformations import PointcloudToTensor, PointcloudScale, PointcloudJitter, PointcloudTranslate, \
    PointcloudRotatePerturbation
from transformer_models import PointTransformerCls
from scripts.helper import load_from_json

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
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            # load data
            pc = data['global_crops'].squeeze(dim=1)
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

    return total_validation_loss/(i + 1), per_class_accuracy


def train_net(cat_to_idx, args):
    # create a list of transformations to be applied to the point cloud.
    transform = transforms.Compose([
        PointcloudToTensor(),
        PointcloudRotatePerturbation(angle_sigma=0.06, angle_clip=0.18),
        PointcloudScale(),
        PointcloudTranslate(),
        PointcloudJitter(std=0.01, clip=0.05)
    ])

    # create the training dataset
    train_dataset = Region(args.mesh_dir, args.pc_dir, args.scene_dir, args.metadata_path, args.accepted_cats_path,
                           num_local_crops=0, num_global_crops=0, mode='train', cat_to_idx=cat_to_idx, num_files=800,
                           transforms=transform)
    val_dataset = Region(args.mesh_dir, args.pc_dir, args.scene_dir, args.metadata_path, args.accepted_cats_path,
                         num_local_crops=0, num_global_crops=0, mode='val', cat_to_idx=cat_to_idx, num_files=200,
                         transforms=transform)

    # create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)

    # load the model
    classifier = PointTransformerCls(args).cuda()

    # check if training is form scratch or a best model
    try:
        checkpoint = torch.load(os.path.join(args.cp_dir, args.best_model_name))
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print('Use pretrain model')
    except FileNotFoundError:
        print('No existing model, starting training from scratch...')
        start_epoch = 0

    # define the optimizer and learning rate scheduler.
    classifier.train()
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
    best_model = classifier
    best_accuracy = 0

    print('Start training...')
    for epoch in range(start_epoch, args.epochs):
        t = time()
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epochs))
        epoch_loss = 0

        for i, data in enumerate(train_loader):
            # load data
            pc = data['global_crops'].squeeze(dim=1)
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
        validation_loss, accuracy = evaluate_net(classifier, val_loader, cat_to_idx)
        validation_losses.append(validation_loss)
        print('Evaluating after epoch {} ...'.format(epoch + 1))
        print('Total validation loss: {}'.format(validation_loss))
        for c, (num_correct, num_total) in accuracy.items():
            print('class {}: {}'.format(c, float(num_correct) / num_total))

        if validation_loss < best_validation_loss:
            best_model = copy.deepcopy(classifier)
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

    # save train/valid losses and the best model
    if args.save_cp:
        np.save(os.path.join(args.cp_dir, 'training_loss'), training_losses)
        np.save(os.path.join(args.cp_dir, 'valid_loss'), validation_losses)
        state = {
            'epoch': best_epoch + 1,
            'accuracy': best_accuracy,
            'best_validation_loss': best_validation_loss,
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, os.path.join(args.cp_dir, 'CP_best.pth'))

    # report the attributes for the best model
    print('Best Validation loss is {}'.format(best_validation_loss))
    print('Best model comes from epoch: {}'.format(best_epoch + 1))
    print('Best accuracy on validation: ')
    for c, (num_correct, num_total) in best_accuracy.items():
        print('class {}: {}'.format(c, float(num_correct)/num_total))


def get_args():
    parser = OptionParser()
    parser.add_option('--accepted_cats_path', dest='accepted_cats_path',
                      default='../../data/matterport3d/accepted_cats.json')
    parser.add_option('--mesh_dir', dest='mesh_dir',
                      default='../../data/matterport3d/mesh_regions')
    parser.add_option('--pc_dir', dest='pc_dir',
                      default='../../data/matterport3d/point_cloud_regions')
    parser.add_option('--scene_dir', dest='scene_dir', default='../../data/matterport3d/scenes')
    parser.add_option('--metadata_path', dest='metadata_path', default='../../data/matterport3d/metadata.csv')
    parser.add_option('--cp_dir', dest='cp_dir',
                      default='../../results/matterport3d/LearningBased/region_classification_transformer')
    parser.add_option('--best_model_name', dest='best_model_name', default='CP_best.pth')

    parser.add_option('--num_point', dest='num_point', default=4096, type='int')
    parser.add_option('--nblocks', dest='nblocks', default=4, type='int')
    parser.add_option('--nneighbor', dest='nneighbor', default=16, type='int')
    parser.add_option('--input_dim', dest='input_dim', default=3, type='int')
    # TODO: use 512
    parser.add_option('--transformer_dim', dest='transformer_dim', default=256, type='int')

    parser.add_option('--epochs', dest='epochs', default=100, type='int', help='number of epochs')
    parser.add_option('--lr', dest='lr', default=1e-4, type='float')
    parser.add_option('--weight_decay', dest='weight_decay', default=1e-4, type='float')
    parser.add_option('--batch_size', dest='batch_size', default=8, type='int')
    parser.add_option('--save_cp', action='store_true', dest='save_cp', default=True, help='save the trained models')

    (options, args) = parser.parse_args()
    return options


def main():
    # get the arguments
    args = get_args()

    # create a directory for checkpoints
    if not os.path.exists(args.cp_dir):
        os.makedirs(args.cp_dir)

    # prepare the accepted categories for training.
    accepted_cats = load_from_json(args.accepted_cats_path)
    accepted_cats = sorted(accepted_cats)
    cat_to_idx = {cat: i for i, cat in enumerate(accepted_cats)}
    args.num_class = len(cat_to_idx)

    # time the training
    t = time()
    train_net(cat_to_idx, args)
    t2 = time()
    print("Training took %.3f minutes" % ((t2 - t) / 60))


if __name__ == '__main__':
    main()



