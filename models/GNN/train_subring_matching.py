import os
import copy
from optparse import OptionParser
from time import time
import torch
from torch import optim
from torch.utils.data import DataLoader

from scene_dataset import Scene
from models import DeepSetAlign, LstmAlign, LinLayer


def evaluate_net(model, valid_loader, loss_criterion, lambda_reg, device):
    model.eval()
    num_samples = 0
    total_validation_loss = 0
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            # load data
            features = data['features'].squeeze(dim=0)
            theta = data['theta'].squeeze(dim=0)
            cos_sin = torch.zeros(1, 2)
            cos_sin[0, 0] = torch.cos(theta)
            cos_sin[0, 1] = torch.sin(theta)
            cos_sin_sum_squared = torch.ones(1)

            features = features.to(device=device, dtype=torch.float32)
            cos_sin = cos_sin.to(device=device, dtype=torch.float32)
            cos_sin_sum_squared = cos_sin_sum_squared.to(device=device, dtype=torch.float32)

            # apply the model and predict the best alignment
            cos_sin_hat = model(features)

            # compute loss
            loss = loss_criterion(cos_sin_hat[:, 0], cos_sin[:, 0]) + \
                   loss_criterion(cos_sin_hat[:, 1], cos_sin[:, 1]) + \
                   lambda_reg * loss_criterion(torch.sum(cos_sin_hat**2).unsqueeze_(dim=0), cos_sin_sum_squared)

            total_validation_loss += loss.item()
            num_samples += len(features)
    print('Total validation loss: {}'.format(total_validation_loss/num_samples))

    return total_validation_loss/num_samples


def train_net(data_dir, num_epochs, lr, device, hidden_dim, save_cp=True, cp_folder='scene_alignment_lstm', input_dim=5,
              eval_itr=1000, patience=5, lambda_reg=1):
    # create a directory for checkpoints
    check_point_dir = '/'.join(data_dir.split('/')[:-1])
    model_dir = os.path.join(check_point_dir, cp_folder)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # create the training dataset
    # TODO: turn this back to train
    train_dataset = Scene(data_dir, mode='val')
    valid_dataset = Scene(data_dir, mode='val')

    # create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=0, shuffle=True)

    # load the model
    # model = DeepSetAlign(input_dim, hidden_dim)
    lstm = LstmAlign(input_dim, hidden_dim, device)
    lin_layer = LinLayer(hidden_dim)
    lstm = lstm.to(device=device)
    lin_layer = lin_layer.to(device=device)
    lin_layer.train()

    # define the optimizer and loss criteria
    optimizer = optim.Adam(list(lstm.parameters()) + list(lin_layer.parameters()), lr=lr)
    smooth_l1_criterion = torch.nn.SmoothL1Loss()

    # track losses and use that along with patience to stop early.
    training_losses = []
    validation_losses = []
    best_validation_loss = float('inf')
    best_epoch = 0
    best_model = lstm
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
            # load data
            features = data['features'].squeeze(dim=0)
            theta = data['theta'].squeeze(dim=0)
            cos_sin = torch.zeros(1, 2)
            cos_sin[0, 0] = torch.cos(theta)
            cos_sin[0, 1] = torch.sin(theta)
            cos_sin_sum_squared = torch.ones(1)

            # np.save(os.path.join(model_dir, data['file_name'][0]), np.asarray(data['features']))
            # if i == 49:
            #     return
            # TODO: remove this after you remove the match column
            features = features[:, :, :-1]

            features = features.to(device=device, dtype=torch.float32)
            cos_sin = cos_sin.to(device=device, dtype=torch.float32)
            cos_sin_sum_squared = cos_sin_sum_squared.to(device=device, dtype=torch.float32)

            # sort the features by their IoU
            # indices = torch.sort(features[:, :, -1], descending=True)[1]
            sorted_features = features #features[:, indices[0], :]

            # apply the model and predict the best alignment
            # cos_sin_hat = model(features)

            # apply the lstm model
            h = lstm.init_hidden(batch_size=1)
            output, h = lstm(sorted_features, h)

            # num_candidates = sorted_features.shape[1]
            # for j in range(num_candidates):
            #     output, h = lstm(sorted_features[:, j:j+1, :], h)

            # apply a fc layer to regress the output of the lstm
            mean_output = torch.mean(output, dim=1).unsqueeze_(dim=1)
            cos_sin_hat = lin_layer(mean_output)

            # compute loss
            loss = smooth_l1_criterion(cos_sin_hat[0, :, 0], cos_sin[:, 0]) + \
                   smooth_l1_criterion(cos_sin_hat[0, :, 1], cos_sin[:, 1]) + \
                   lambda_reg * smooth_l1_criterion(torch.sum(cos_sin_hat**2).unsqueeze_(dim=0), cos_sin_sum_squared)

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # keep track of the losses
            epoch_loss += loss.item()
            num_samples += len(features)

            # # compute validation loss to pick the best model
            # if (i + 1) % eval_itr == 0:
            #     validation_loss = evaluate_net(model, valid_loader, smooth_l1_criterion, lambda_reg, device=device)
            #     validation_losses.append(validation_loss)
            #     if validation_loss < best_validation_loss:
            #         best_model = copy.deepcopy(model)
            #         best_validation_loss = validation_loss
            #         best_epoch = epoch
            #         bad_epoch = 0
            #     else:
            #         bad_epoch += 1

        # save model from every nth epoch
        print('Epoch %d finished! - Loss: %.6f' % (epoch + 1, epoch_loss / (i + 1)))
        print(cos_sin_hat[0, 0, 0].item(), cos_sin[0, 0].item(), cos_sin_hat[0, 0, 1].item(), cos_sin[0, 1].item())
        print('-' * 50)
    #     if save_cp and ((epoch + 1) % 20 == 0):
    #         torch.save(model.state_dict(), os.path.join(model_dir, 'CP_{}.pth'.format(epoch + 1)))
    #         print('Checkpoint %d saved !' % (epoch + 1))
    #     training_losses.append(epoch_loss / num_samples)
    #
    # # save train/valid losses and the best model
    # if save_cp:
    #     np.save(os.path.join(model_dir, 'training_loss'), training_losses)
    #     np.save(os.path.join(model_dir, 'valid_loss'), validation_losses)
    #     torch.save(best_model.state_dict(), os.path.join(model_dir, 'CP_best.pth'))
    #
    # # report the attributes for the best model
    # print('Best Validation loss is {}'.format(best_validation_loss))
    # print('Best model comes from epoch: {}'.format(best_epoch))


def get_args():
    parser = OptionParser()
    parser.add_option('--epochs', dest='epochs', default=400, type='int', help='number of epochs')
    parser.add_option('--data-dir', dest='data_dir', default='../../results/matterport3d/GNN/scene_graphs_cl',
                      help='data directory')
    parser.add_option('--hidden_dim', dest='hidden_dim', default=512, type='int')
    parser.add_option('--patience', dest='patience', default=50, type='int')
    parser.add_option('--eval_iter', dest='eval_iter', default=1000, type='int')
    parser.add_option('--cp_folder', dest='cp_folder', default='scene_alignment_lstm')
    parser.add_option('--input_dim', dest='input_dim', default=5)
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
    train_net(data_dir=args.data_dir, num_epochs=args.epochs, lr=1e-3, device=device, hidden_dim=args.hidden_dim,
              patience=args.patience, eval_itr=args.eval_iter, cp_folder=args.cp_folder, input_dim=args.input_dim)
    t2 = time()
    print("Training took %.3f minutes" % ((t2 - t) / 60))


if __name__ == '__main__':
    main()




