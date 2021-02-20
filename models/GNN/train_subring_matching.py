import os
import copy
from optparse import OptionParser
from time import time
import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

from ring_dataset import RingDataset
from gnn_models import Discriminator, LinearLayer, GCN_RES


def normalize_adj_mp(adj, nb_nodes, device):
    self_loop = torch.eye(nb_nodes, nb_nodes)
    self_loop = self_loop.to(device=device, dtype=torch.float32)

    degree = torch.zeros_like(adj)
    for j in range(adj.shape[1]):
        # check the adj has no self loops to begin with
        assert adj[0, j, :, :].diagonal().sum() == 0

        adj[0, j, :, :] += self_loop
        row_sum = torch.sum(adj[0, j, :, :], dim=1)
        d_inv = torch.pow(row_sum, -1)
        d_inv[torch.isinf(d_inv)] = 0.
        ind = np.diag_indices(nb_nodes)
        degree[0, j, ind[0], ind[1]] = d_inv
        adj[0, j, :, :] = torch.mm(degree[0, j, :, :], adj[0, j, :, :])
    return adj


def compute_accuracy(per_class_accuracy, predicted_labels, labels):
    for c, (num_correct, num_total) in per_class_accuracy.items():
        labels_c = labels[labels == c]
        predicted_labels_c = predicted_labels[labels == c]

        num_total += len(labels_c)
        num_correct += sum(predicted_labels_c == labels_c).item()
        per_class_accuracy[c] = (num_correct, num_total)


def evaluate_net(models_dic, valid_loader, criterion, device):
    # set models to evaluation mode
    for model_name, model in models_dic.items():
        model.eval()

    num_samples = 0
    total_validation_loss = 0
    num_correct = 0
    per_class_accuracy = {0: (0, 0), 1: (0, 0)}
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            # read the features, label and adj
            features = data['features'].squeeze(dim=0)
            adj = data['adj'].squeeze(dim=0)
            label = data['label'].squeeze(dim=0)

            features = features.to(device=device, dtype=torch.float32)
            adj = adj.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            # apply linear layer and find the mean summary of the positive features
            pos_idx = label == 1
            hidden_pos = models_dic['lin_layer'](features[:, pos_idx[0, :], :])
            summary = torch.sigmoid(torch.mean(hidden_pos, dim=1))

            # add self loops and normalize the adj
            nb_nodes = adj.shape[-1]
            adj = normalize_adj_mp(adj, nb_nodes, device)

            # apply gnn on the full ring
            hidden = models_dic['gcn_res'](features, adj)

            # apply the discriminator to separate the positive and negative node embeddings.
            logits = models_dic['disc'](summary, hidden)
            loss = criterion(logits, label)

            # predict if an edge connection is positive or negative
            prediction = torch.sigmoid(logits) > 0.5
            num_correct += torch.sum(prediction == label).item()
            num_samples += len(label)
            total_validation_loss += loss.item()

            # compute per class accuracy
            compute_accuracy(per_class_accuracy, prediction, label)

    # display per class accuracy and validation loss
    for c, (num_correct, num_total) in per_class_accuracy.items():
        if num_total != 0:
            print('class {}: '.format(c), float(num_correct) / num_total)
    print('Total validation loss {}: '.format(total_validation_loss/num_samples))
    print('-'*50)

    return total_validation_loss/num_samples, per_class_accuracy


def train_net(data_dir, num_epochs, lr, device, hidden_dim, num_layers, save_cp=True,
              model_name='subring_matching_base', eval_itr=1000, patience=5):
    # create a directory for checkpoints
    check_point_dir = '/'.join(data_dir.split('/')[:-1])
    model_dir = os.path.join(check_point_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # add transformations and create the training dataset
    train_dataset = RingDataset(data_dir, mode='train')
    valid_dataset = RingDataset(data_dir, mode='val')

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=4, shuffle=True)

    # load the models and prepare for training
    sample_data = list(enumerate(train_loader))[1][1]
    input_dim = sample_data['features'].shape[-1]
    num_edge_type = sample_data['adj'].shape[2]
    lin_layer = LinearLayer(input_dim=input_dim, output_dim=hidden_dim)
    gcn_res = GCN_RES(input_dim, hidden_dim, 'prelu', num_edge_type, num_layers)
    disc = Discriminator(hidden_dim=hidden_dim)

    lin_layer = lin_layer.to(device=device)
    gcn_res = gcn_res.to(device=device)
    disc = disc.to(device=device)
    lin_layer.train()
    gcn_res.train()
    disc.train()
    models_dic = {'lin_layer': lin_layer, 'gcn_res': gcn_res, 'disc': disc}

    # define the optimizer and loss criteria
    optimizer = optim.Adam(disc.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    training_losses = []
    validation_losses = []
    best_validation_loss = float('inf')
    best_accuracy_dic = {}
    best_epoch = 0
    best_model_dic = models_dic
    bad_epoch = 0

    print('Training...')
    for epoch in range(num_epochs):
        print('Epoch %d/%d' % (epoch + 1, num_epochs))
        epoch_loss = 0
        num_samples = 0

        # if validation loss is not improving after x many iterations, stop early.
        if bad_epoch == patience:
            print('Stopped Early')
            break

        for i, data in enumerate(train_loader):
            # read the features, label and adj
            features = data['features'].squeeze(dim=0)
            adj = data['adj'].squeeze(dim=0)
            label = data['label'].squeeze(dim=0)

            features = features.to(device=device, dtype=torch.float32)
            adj = adj.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            # apply linear layer and find the mean summary of the positive features
            pos_idx = label == 1
            hidden_pos = models_dic['lin_layer'](features[:, pos_idx[0, :], :])
            pos_summary = torch.sigmoid(torch.mean(hidden_pos, dim=1))

            # add self loops and normalize the adj
            nb_nodes = adj.shape[-1]
            adj = normalize_adj_mp(adj, nb_nodes, device)

            # apply gnn on the full ring
            hidden = gcn_res(features, adj)

            # apply the discriminator to separate the positive and negative node embeddings.
            logits = disc(pos_summary, hidden)

            # compute the loss
            loss = criterion(logits, label)

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # keep track of the losses
            epoch_loss += loss.item()
            num_samples += len(features)

            # compute accuracy and loss on validation data to pick the best model
            if (i+1) % eval_itr == 0:
                validation_loss, accuracy_dic = evaluate_net(models_dic, valid_loader, criterion, device=device)
                validation_losses.append(validation_loss)
                if validation_loss < best_validation_loss:
                    best_model_dic = copy.deepcopy(models_dic)
                    best_validation_loss = validation_loss
                    best_accuracy_dic = accuracy_dic
                    best_epoch = epoch
                    bad_epoch = 0
                else:
                    bad_epoch += 1

        # save model from every nth epoch
        if save_cp and ((epoch + 1) % 10 == 0):
            for model_name, model in models_dic.items():
                torch.save(model.state_dict(), os.path.join(model_dir, 'CP_{}_{}.pth'.format(model_name, epoch + 1)))
                print('Checkpoint %d saved !' % (epoch + 1))
        training_losses.append(epoch_loss / num_samples)
        print('Epoch %d finished! - Loss: %.6f' % (epoch + 1, epoch_loss / (i + 1)))

    # save train/valid losses and the best model
    if save_cp:
        np.save(os.path.join(model_dir, 'training_loss'), training_losses)
        np.save(os.path.join(model_dir, 'valid_loss'), validation_losses)
        for model_name, model in best_model_dic.items():
            torch.save(model.state_dict(), os.path.join(model_dir, 'CP_{}_best.pth'.format(model_name)))

    # report the attributes for the best model
    print('Best Validation loss is {}'.format(best_validation_loss))
    print('Best model comes from epoch: {}'.format(best_epoch))
    print('Best accuracy on validation: ')
    for c, (num_correct, num_total) in best_accuracy_dic.items():
        print('class {}: {}'.format(c, float(num_correct)/num_total))


def get_args():
    parser = OptionParser()
    parser.add_option('--epochs', dest='epochs', default=50, type='int', help='number of epochs')
    parser.add_option('--data-dir', dest='data_dir', default='../../results/matterport3d/GNN/scene_graphs_cl',
                      help='data directory')
    parser.add_option('--hidden_dim', dest='hidden_dim', default=128, type='int')
    parser.add_option('--num_layers', dest='num_layers', default=2, type='int')
    parser.add_option('--patience', dest='patience', default=20, type='int')
    parser.add_option('--eval_iter', dest='eval_iter', default=1000, type='int')
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
    train_net(data_dir=args.data_dir, num_epochs=args.epochs, lr=1e-5, device=device, hidden_dim=args.hidden_dim,
              num_layers=args.num_layers, patience=args.patience, eval_itr=args.eval_iter)
    t2 = time()
    print("Training took %.3f minutes" % ((t2 - t) / 60))


if __name__ == '__main__':
    main()




