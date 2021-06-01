import os
import copy
from optparse import OptionParser
from time import time
import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

from graph_dataset import Scene
from models import GCN_RES, Classifier
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


def evaluate_net(model_dic, valid_loader, cat_to_idx, device):
    # set models to evaluation mode
    for model_name, model in model_dic.items():
        model.eval()

    total_validation_loss = 0
    per_class_accuracy = {cat: (0, 0) for cat in cat_to_idx.keys()}
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            # load data
            features = data['features'].squeeze(dim=0)
            labels = data['labels'].squeeze(dim=0)
            adj = data['adj'].squeeze(dim=0)

            # move data to the right device
            features = features.to(device=device, dtype=torch.float32)
            adj = adj.to(device=device, dtype=torch.float)
            labels = labels.to(device=device, dtype=torch.long)

            # apply the model
            H = model_dic['gnn'](features, adj)
            output = model_dic['lin_layer'](H)

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


def train_net(scene_graph_dir, latent_caps_dir, cat_to_idx, num_epochs, lr, device, hidden_dim, num_layers,
              save_cp=True, eval_itr=1000, patience=5, cp_dir=None):
    # create the training dataset
    train_dataset = Scene(scene_graph_dir, latent_caps_dir, cat_to_idx, mode='train')
    val_dataset = Scene(scene_graph_dir, latent_caps_dir, cat_to_idx, mode='val')

    # create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4, shuffle=True)

    # find the required dimensions for building the GNN.
    sample_data = list(enumerate(train_loader))[1][1]
    input_dim = sample_data['features'].shape[-1]
    num_edge_types = sample_data['adj'].shape[1]

    # load the model and set it to the right device
    gnn = GCN_RES(input_dim, hidden_dim, 'prelu', num_edge_types, num_layers=num_layers)
    lin_layer = Classifier(hidden_dim, len(cat_to_idx))

    gnn = gnn.to(device=device)
    lin_layer = lin_layer.to(device=device)

    # prepare for training.
    model_dic = {'gnn': gnn, 'lin_layer': lin_layer}
    for model_name, model in model_dic.items():
        model.train()

    # define the optimizer and loss criteria
    params = []
    for model_name, model in model_dic.items():
        params += list(model.parameters())
    optimizer = optim.Adam(params, lr=lr)

    # track losses and use that along with patience to stop early.
    training_losses = []
    validation_losses = []
    best_validation_loss = float('inf')
    best_epoch = 0
    best_model_dic = model_dic
    best_accuracy = 0
    bad_epoch = 0

    print('Training...')
    for epoch in range(num_epochs):
        print('Epoch %d/%d' % (epoch + 1, num_epochs))
        epoch_loss = 0

        # if validation loss is not improving after x many iterations, stop early.
        if bad_epoch == patience:
            print('Stopped Early')
            break

        for i, data in enumerate(train_loader):
            # load data
            features = data['features'].squeeze(dim=0)
            labels = data['labels'].squeeze(dim=0)
            adj = data['adj'].squeeze(dim=0)

            # move data to the right device
            features = features.to(device=device, dtype=torch.float32)
            adj = adj.to(device=device, dtype=torch.float)
            labels = labels.to(device=device, dtype=torch.long)

            # apply the model
            H = gnn(features, adj)
            output = lin_layer(H)

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

            # compute validation loss to pick the best model
            if (i+1) % eval_itr == 0:
                validation_loss, accuracy = evaluate_net(model_dic, val_loader, cat_to_idx, device=device)
                validation_losses.append(validation_loss)
                if validation_loss < best_validation_loss:
                    best_model_dic = copy.deepcopy(model_dic)
                    best_validation_loss = validation_loss
                    best_accuracy = accuracy
                    best_epoch = epoch
                    bad_epoch = 0
                else:
                    bad_epoch += 1
                # set the models back to training mode
                for model_name, model in model_dic.items():
                    model.train()

        # save model from every nth epoch
        print('Epoch %d finished! - Loss: %.10f' % (epoch + 1, epoch_loss / (i + 1)))
        print('-' * 50)
        if save_cp and ((epoch + 1) % 20 == 0):
            for model_name, model in model_dic.items():
                torch.save(model.state_dict(), os.path.join(cp_dir, 'CP_{}_{}.pth'.format(model_name, epoch + 1)))
            print('Checkpoint %d saved !' % (epoch + 1))
        training_losses.append(epoch_loss / (i + 1))

    # save train/valid losses and the best model
    if save_cp:
        np.save(os.path.join(cp_dir, 'training_loss'), training_losses)
        np.save(os.path.join(cp_dir, 'valid_loss'), validation_losses)
        for model_name, model in best_model_dic.items():
            torch.save(model.state_dict(), os.path.join(cp_dir, 'CP_{}_best.pth'.format(model_name)))

    # report the attributes for the best model
    print('Best Validation loss is {}'.format(best_validation_loss))
    print('Best model comes from epoch: {}'.format(best_epoch + 1))
    print('Best accuracy on validation: ')
    for c, (num_correct, num_total) in best_accuracy.items():
        print('class {}: {}'.format(c, float(num_correct)/num_total))


def get_args():
    parser = OptionParser()
    parser.add_option('--epochs', dest='epochs', default=200, type='int', help='number of epochs')
    parser.add_option('--accepted_cats_path', dest='accepted_cats_path',
                      default='../../data/matterport3d/accepted_cats.json')
    parser.add_option('--scene_graph_dir', dest='scene_graph_dir',
                      default='../../results/matterport3d/LearningBased/scene_graphs')
    parser.add_option('--latent_caps_dir', dest='latent_caps_dir',
                      default='../../data/matterport3d/latent_caps')
    parser.add_option('--hidden_dim', dest='hidden_dim', default=1024, type='int')
    parser.add_option('--num_layers', dest='num_layers', default=1, type='int')
    parser.add_option('--lr', dest='lr', default=1e-5, type='float')
    parser.add_option('--patience', dest='patience', default=20, type='int')
    parser.add_option('--eval_iter', dest='eval_iter', default=1000, type='int')
    parser.add_option('--cp_dir', dest='cp_dir',
                      default='../../results/matterport3d/LearningBased/gnn_cat_prediction')
    parser.add_option('--gpu', action='store_true', dest='gpu', default=True, help='use cuda')

    (options, args) = parser.parse_args()
    return options


def main():
    # get the arguments
    args = get_args()

    # create a directory for checkpoints
    if not os.path.exists(args.cp_dir):
        os.makedirs(args.cp_dir)

    # Set the right device for all the models
    device = torch.device('cpu')
    if args.gpu:
        device = torch.device('cuda')

    # prepare the accepted categories for training.
    accepted_cats = load_from_json(args.accepted_cats_path)
    accepted_cats = sorted(accepted_cats)
    cat_to_idx = {cat: i for i, cat in enumerate(accepted_cats)}

    # time the training
    t = time()
    train_net(scene_graph_dir=args.scene_graph_dir, latent_caps_dir=args.latent_caps_dir, cat_to_idx=cat_to_idx,
              num_epochs=args.epochs, lr=args.lr,  device=device,  hidden_dim=args.hidden_dim,
              num_layers=args.num_layers, patience=args.patience, eval_itr=args.eval_iter, cp_dir=args.cp_dir)
    t2 = time()
    print("Training took %.3f minutes" % ((t2 - t) / 60))


if __name__ == '__main__':
    main()




