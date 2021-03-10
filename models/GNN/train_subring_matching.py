import os
from optparse import OptionParser
from time import time
import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

from scene_dataset import Scene


def rotate_objects(features, features_thetas, rot):
    features_rot = features.clone()
    features_thetas_rot = features_thetas.clone()

    # rotate the thetas
    features_thetas_rot += rot
    features_thetas_rot = features_thetas_rot % (2 * np.pi)

    # compute cos and sin of the rotated features
    cos_theta = torch.cos(features_thetas_rot)
    sin_theta = torch.sin(features_thetas_rot)

    # change the old cos, sin features to the rotated ones
    rotated_cos_sin = torch.cat([cos_theta, sin_theta], dim=2)
    features_rot[:, :, -2:] = rotated_cos_sin

    return features_rot, features_thetas_rot


def train_net(data_dir, num_epochs, lr, device, hidden_dim, num_layers, save_cp=True, cp_folder='subring_matching_cat',
              eval_itr=1000, patience=5):
    # create a directory for checkpoints
    check_point_dir = '/'.join(data_dir.split('/')[:-1])
    model_dir = os.path.join(check_point_dir, cp_folder)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # add transformations and create the training dataset
    train_dataset = Scene(data_dir, mode='train')

    # create dataloaders
    #TODO: change num workesrs and shuffle
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=False)

    print('Training...')
    for epoch in range(num_epochs):
        print('Epoch %d/%d' % (epoch + 1, num_epochs))
        for i, data in enumerate(train_loader):
            bp_graph = data['bp_graphs'][0]
            # check at training every node is matched with itself
            for q_context, q_context_info in bp_graph.items():
                if q_context != q_context_info['match'][0][0]:
                    print(q_context, q_context_info['match'][0][0].item())
                    raise Exception('node did not match to iteself')
            continue


def get_args():
    parser = OptionParser()
    parser.add_option('--epochs', dest='epochs', default=200, type='int', help='number of epochs')
    parser.add_option('--data-dir', dest='data_dir', default='../../results/matterport3d/GNN/scene_graphs_cl',
                      help='data directory')
    parser.add_option('--hidden_dim', dest='hidden_dim', default=1024, type='int')
    parser.add_option('--num_layers', dest='num_layers', default=2, type='int')
    parser.add_option('--patience', dest='patience', default=20, type='int')
    parser.add_option('--eval_iter', dest='eval_iter', default=1000, type='int')
    parser.add_option('--cp_folder', dest='cp_folder', default='subring_matching_dev')
    parser.add_option('--gpu', action='store_true', dest='gpu', default=True, help='use cuda')

    (options, args) = parser.parse_args()
    return options


def main():
    # get the arguments
    args = get_args()

    # Set the right device for all the models
    device = torch.device('cpu')
    if args.gpu:
        device = torch.device('cuda')

    # time the training
    t = time()
    train_net(data_dir=args.data_dir, num_epochs=args.epochs, lr=1e-4, device=device, hidden_dim=args.hidden_dim,
              num_layers=args.num_layers, patience=args.patience, eval_itr=args.eval_iter, cp_folder=args.cp_folder)
    t2 = time()
    print("Training took %.3f minutes" % ((t2 - t) / 60))


if __name__ == '__main__':
    main()




