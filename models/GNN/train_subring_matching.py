import os
from optparse import OptionParser
from time import time
import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

from ring_dataset import RingDataset
from gnn_models import Discriminator, LinearLayer


def train_net(data_dir, num_epochs, lr, device, hidden_dim, save_cp=True, model_name='subring_matching', patience=5):
    # create a directory for checkpoints
    check_point_dir = '/'.join(data_dir.split('/')[:-1])
    model_dir = os.path.join(check_point_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # add transformations and create the training dataset
    train_dataset = RingDataset(data_dir, mode='train')

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)

    # load the models and prepare for training
    input_dim = list(enumerate(train_loader))[1][1]['positive_negatives'].shape[-1]
    lin_layer = LinearLayer(input_dim=input_dim, output_dim=hidden_dim)
    disc = Discriminator(hidden_dim=hidden_dim)

    lin_layer = lin_layer.to(device=device)
    disc = disc.to(device=device)
    disc.train()
    lin_layer.train()
    models = {'disc': disc, 'lin_layer': lin_layer}

    # define the optimizer and loss criteria
    optimizer = optim.Adam(disc.parameters(), lr=lr)
    criteria = torch.nn.BCEWithLogitsLoss()

    training_losses = []
    print('Training...')
    best_epoch_loss = 10**5
    count = 0
    for epoch in range(num_epochs):
        print('Epoch %d/%d' % (epoch + 1, num_epochs))
        epoch_loss = 0
        for i, data in enumerate(train_loader):
            # separate positives and negatives.
            pos_neg = data['positive_negatives'].to(device=device, dtype=torch.float32)
            num_pos = data['num_positives']
            num_neg = data['positive_negatives'].shape[1] - num_pos

            # apply linear layer and find the mean summary of the positive features
            hidden = lin_layer(pos_neg)
            gated_mean = torch.sigmoid(torch.mean(hidden[:, :num_pos, :], dim=1))

            # apply the discriminator to positive and negative examples.
            logits = disc(gated_mean, hidden)

            # compute the loss
            pos_label = torch.ones(1, num_pos)
            neg_label = torch.ones(1, num_neg)
            pos_label = pos_label.to(device=device)
            neg_label = neg_label.to(device=device)
            label = torch.cat([pos_label, neg_label], dim=1)
            loss = criteria(logits, label)

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            training_losses.append(loss.item())

        # if epoch loss is not improving, stop early.
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            count = 0
        else:
            count += 1
        if count == patience:
            print('Stopped Early')
            break

        # save model from every nth epoch
        if save_cp and ((epoch + 1) % 10 == 0):
            for model_name, model in models.items():
                torch.save(model.state_dict(), os.path.join(model_dir, 'CP_{}_{}.pth'.format(model_name, epoch + 1)))
                print('Checkpoint %d saved !' % (epoch + 1))
        print('Epoch %d finished! - Loss: %.6f' % (epoch + 1, epoch_loss / (i + 1)))

    if save_cp:
        np.save(os.path.join(model_dir, 'training_loss'), training_losses)
    #     for model_name, model in models.items():
    #         torch.save(model.state_dict(), os.path.join(model_dir, 'CP_{}_best.pth'.format(model_name)))


def get_args():
    parser = OptionParser()
    parser.add_option('--epochs', dest='epochs', default=50, type='int', help='number of epochs')
    parser.add_option('--data-dir', dest='data_dir', default='../../results/matterport3d/GNN/scene_graphs_cl',
                      help='data directory')
    parser.add_option('--hidden_dim', dest='hidden_dim', default=256, type='int')
    parser.add_option('--patience', dest='patience', default=5, type='int')
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
    train_net(data_dir=args.data_dir, num_epochs=args.epochs, lr=1e-6, device=device, hidden_dim=args.hidden_dim,
              patience=args.patience)
    t2 = time()
    print("Training took %.3f minutes" % ((t2 - t) / 60))


if __name__ == '__main__':
    main()




