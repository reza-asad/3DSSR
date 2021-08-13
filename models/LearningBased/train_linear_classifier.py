import os
import copy
from optparse import OptionParser
from time import time
import torch
from torchvision import transforms
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

from region_dataset import Region
from transformations import PointcloudScale, PointcloudJitter, PointcloudTranslate, PointcloudRotatePerturbation
from projection_models import MLP
from capsnet_models import PointCapsNet
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


def evaluate_net(model_dic, capsule_net, valid_loader, cat_to_idx):
    # set models to evaluation mode
    for model_name, model in model_dic.items():
        model.eval()

    total_validation_loss = 0
    per_class_accuracy = {cat: (0, 0) for cat in cat_to_idx.keys()}
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            # load data
            global_crops = data['global_crops']
            labels = data['labels'].squeeze(dim=1)

            # set datatype
            global_crops = global_crops.to(dtype=torch.float32).cuda()
            labels = labels.to(dtype=torch.long).cuda()

            # apply the pre-trained capsulenet model first
            global_crops = global_crops.transpose(2, 1)
            global_caps = capsule_net(global_crops).reshape(-1, 64 * 64)

            # apply the model
            output = model_dic['lin_layer'](global_caps)

            # compute loss
            ce_loss = torch.nn.functional.cross_entropy(output, labels, reduction='none')
            pt = torch.exp(-ce_loss)
            loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()
            total_validation_loss += loss.item()

            _, predictions = torch.max(torch.softmax(output, dim=1), dim=1)
            compute_accuracy(per_class_accuracy, predictions, labels, cat_to_idx)

    # display per class accuracy
    for c, (num_correct, num_total) in per_class_accuracy.items():
        if num_total != 0:
            print(c, float(num_correct) / num_total)
    print('Total validation loss: {}'.format(total_validation_loss/(i + 1)))

    return total_validation_loss/(i + 1), per_class_accuracy


def train_net(cat_to_idx, args):
    # create a list of transformations to be applied to the point cloud.
    transform = transforms.Compose([
        PointcloudRotatePerturbation(angle_sigma=0.06, angle_clip=0.18),
        PointcloudScale(),
        PointcloudTranslate(),
        PointcloudJitter(std=0.01, clip=0.05)
    ])

    # create the training dataset
    train_dataset = Region(args.data_dir, args.pc_dir, args.scene_dir, args.metadata_path, args.accepted_cats_path,
                           num_local_crops=0, num_global_crops=0, mode='train', cat_to_idx=cat_to_idx, num_files=None,
                           transforms=transform)
    val_dataset = Region(args.data_dir, args.pc_dir, args.scene_dir, args.metadata_path, args.accepted_cats_path,
                         num_local_crops=0, num_global_crops=0, mode='val', cat_to_idx=cat_to_idx, num_files=None)

    # create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)

    # load the pre-trained capsulenet.
    capsule_net = PointCapsNet(1024, 16, 64, 64, args.num_points)
    capsule_net.load_state_dict(torch.load(args.best_capsule_net))
    capsule_net = torch.nn.DataParallel(capsule_net).cuda()
    capsule_net.eval()

    # load the model.
    lin_layer = MLP(64*64, args.hidden_dim, len(cat_to_idx)).cuda()

    # prepare for training.
    model_dic = {'lin_layer': lin_layer}
    for model_name, model in model_dic.items():
        model.train()

    # define the optimizer and loss criteria
    params = []
    for model_name, model in model_dic.items():
        params += list(model.parameters())
    optimizer = optim.Adam(params, lr=args.lr)

    # track losses.
    training_losses = []
    validation_losses = []
    best_validation_loss = float('inf')
    best_epoch = 0
    best_model_dic = model_dic
    best_accuracy = 0

    print('Training...')
    for epoch in range(args.epochs):
        t = time()
        print('Epoch %d/%d' % (epoch + 1, args.epochs))
        epoch_loss = 0

        # if validation loss is not improving after x many iterations, stop early.
        for i, data in enumerate(train_loader):
            # load data
            global_crops = data['global_crops']
            labels = data['labels'].squeeze(dim=1)

            # set datatype
            global_crops = global_crops.to(dtype=torch.float32).cuda()
            labels = labels.to(dtype=torch.long).cuda()

            # apply the pre-trained capsulenet model first
            global_crops = global_crops.transpose(2, 1)
            global_caps = capsule_net(global_crops).reshape(-1, 64 * 64)
            output = lin_layer(global_caps)

            # compute loss
            ce_loss = torch.nn.functional.cross_entropy(output, labels, reduction='none')
            pt = torch.exp(-ce_loss)
            loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # keep track of the losses
            epoch_loss += loss.item()

        print('Epoch %d finished! - Loss: %.10f' % (epoch + 1, epoch_loss / (i + 1)))
        t2 = time()
        print("Training epoch took %.3f minutes" % ((t2 - t) / 60))

        # compute validation loss.
        validation_loss, accuracy = evaluate_net(model_dic, capsule_net, val_loader, cat_to_idx)
        validation_losses.append(validation_loss)
        if validation_loss < best_validation_loss:
            best_model_dic = copy.deepcopy(model_dic)
            best_validation_loss = validation_loss
            best_accuracy = accuracy
            best_epoch = epoch

        # set the models back to training mode
        for model_name, model in model_dic.items():
            model.train()

        # save model from every nth epoch
        if args.save_cp and ((epoch + 1) % 20 == 0):
            for model_name, model in model_dic.items():
                torch.save(model.state_dict(), os.path.join(args.cp_dir, 'CP_{}_{}.pth'.format(model_name, epoch + 1)))
            print('Checkpoint %d saved !' % (epoch + 1))
        training_losses.append(epoch_loss / (i + 1))
        print('-' * 50)

    # save train/valid losses and the best model
    if args.save_cp:
        np.save(os.path.join(args.cp_dir, 'training_loss'), training_losses)
        np.save(os.path.join(args.cp_dir, 'valid_loss'), validation_losses)
        for model_name, model in best_model_dic.items():
            torch.save(model.state_dict(), os.path.join(args.cp_dir, 'CP_{}_best.pth'.format(model_name)))

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
    parser.add_option('--data_dir', dest='data_dir',
                      default='../../data/matterport3d/mesh_regions')
    parser.add_option('--pc_dir', dest='pc_dir',
                      default='../../data/matterport3d/point_cloud_regions')
    parser.add_option('--scene_dir', dest='scene_dir', default='../../data/matterport3d/scenes')
    parser.add_option('--metadata_path', dest='metadata_path', default='../../data/matterport3d/metadata.csv')
    parser.add_option('--cp_dir', dest='cp_dir',
                      default='../../results/matterport3d/LearningBased/region_classification_capsnet_linear')
    parser.add_option('--best_capsule_net', dest='best_capsule_net',
                      default='../../results/matterport3d/LearningBased/region_classification_capsnet_linear/'
                              'best_capsule_net.pth')

    parser.add_option('--epochs', dest='epochs', default=100, type='int', help='number of epochs')
    parser.add_option('--num_points', dest='num_points', default=4096, type='int')
    parser.add_option('--lr', dest='lr', default=1e-3, type='float')
    parser.add_option('--batch_size', dest='batch_size', default=7, type='int')
    parser.add_option('--hidden_dim', dest='hidden_dim', default=1024, type='int')
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

    # time the training
    t = time()
    train_net(cat_to_idx, args)
    t2 = time()
    print("Training took %.3f minutes" % ((t2 - t) / 60))


if __name__ == '__main__':
    main()



