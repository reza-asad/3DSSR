import os
import copy
from optparse import OptionParser
from time import time
import torch
from torch import optim
from torch.utils.data import DataLoader
import torchvision.models as models
import numpy as np

from object_dataset import Object
from models import Lstm, CorrClassifier


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def evaluate_net(model_dic, wide_resnet50_2, valid_loader, criterion, device):
    # set models to evaluation mode
    for model_name, model in model_dic.items():
        model.eval()

    num_samples = 0
    total_validation_loss = 0
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            # skip if there are no negatives
            if data['skip']:
                continue

            # load data
            latent_caps = data['latent_caps']
            obj_imgs = data['obj_imgs'][0, ...]
            context_imgs = data['context_imgs'][0, ...]
            labels = data['labels']

            latent_caps = latent_caps.to(device=device, dtype=torch.float32)
            obj_imgs = obj_imgs.to(device=device, dtype=torch.float32)
            context_imgs = context_imgs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)

            # apply the lstm model on the latent caps
            h = model_dic['lstm_3d'].init_hidden(batch_size=1)
            output, h = model_dic['lstm_3d'](latent_caps, h)

            # package the features for the object and context 3D shapes. first make sure there is context at all.
            if len(output[0, :-1, :]) > 0:
                context_3d = torch.mean(output[0, :-1, :], dim=0).unsqueeze_(dim=0)
            else:
                context_3d = output[0, -1:, :]
            shape_features = torch.cat([output[0, -1:, :], context_3d], dim=1)

            # apply the pretrained resent model to the object and context images
            obj_img_features = wide_resnet50_2(obj_imgs)
            context_img_features = wide_resnet50_2(context_imgs)

            # concat the img features for object and context images.
            img_features = torch.cat([obj_img_features, context_img_features], dim=1)

            # prepend the 3d shape features to the img features
            all_features = torch.cat([shape_features, img_features], dim=0)
            all_features.unsqueeze_(dim=0)

            # apply the lstm model on the full features.
            output, h = model_dic['lstm_joint'](all_features, h)

            # apply a fc layer to predict the corresponding img to the 3d object. note to exclude the shape feature
            correspondence = model_dic['lin_layer'](output[0, 1:, :])
            correspondence.unsqueeze_(dim=0)

            # compute loss
            loss = criterion(correspondence, labels)

            total_validation_loss += loss.item()
            num_samples += 1
    print('Total validation loss: {}'.format(total_validation_loss/num_samples))

    return total_validation_loss/num_samples


def train_net(scene_graph_dir, imgs_dir, scene_to_pos_negatives, models_dir, latent_caps_dir, num_epochs, lr, device,
              hidden_dim, save_cp=True, cp_folder='obj_correspondence_lstm', input_dim_3d=4096, input_dim_joint=2048,
              eval_itr=1000, patience=5):
    # create a directory for checkpoints
    check_point_dir = '/'.join(scene_graph_dir.split('/')[:-1])
    model_dir = os.path.join(check_point_dir, cp_folder)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # create the training dataset
    train_dataset = Object(scene_graph_dir, imgs_dir, scene_to_pos_negatives, models_dir, latent_caps_dir, mode='train')
    valid_dataset = Object(scene_graph_dir, imgs_dir, scene_to_pos_negatives, models_dir, latent_caps_dir, mode='val')

    # create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=0, shuffle=True)

    # load the models
    lstm_3d = Lstm(input_dim_3d, hidden_dim, device)
    lstm_joint = Lstm(input_dim_joint, hidden_dim, device)
    lin_layer = CorrClassifier(hidden_dim)
    wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)

    # adjust the last layer of the resnet and freeze other layers.
    # set_parameter_requires_grad(wide_resnet50_2, feature_extracting=True)
    wide_resnet50_2.fc = torch.nn.Linear(2048, input_dim_joint//2, bias=False)

    # set models to the right device
    lstm_3d = lstm_3d.to(device=device)
    lstm_joint = lstm_joint.to(device=device)
    lin_layer = lin_layer.to(device=device)
    wide_resnet50_2 = wide_resnet50_2.to(device=device)

    model_dic = {'resnet': wide_resnet50_2, 'lstm_3d': lstm_3d, 'lstm_joint': lstm_joint, 'lin_layer': lin_layer}
    for model_name, model in model_dic.items():
        model.train()

    # define the optimizer and loss criteria
    params = []
    for model_name, model in model_dic.items():
        params += list(model.parameters())
    optimizer = optim.Adam(params, lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    # track losses and use that along with patience to stop early.
    training_losses = []
    validation_losses = []
    best_validation_loss = float('inf')
    best_epoch = 0
    best_model_dic = model_dic
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

        for data in train_loader:
            # first check if this scene is useful for training
            if data['skip']:
                continue

            # load data
            latent_caps = data['latent_caps']
            obj_imgs = data['obj_imgs'][0, ...]
            context_imgs = data['context_imgs'][0, ...]
            labels = data['labels']

            # print(latent_caps.shape)
            # print(obj_imgs.shape)
            # print(context_imgs.shape)
            # print(labels.shape)
            # print(labels)
            # # visualize the last object and context images
            # obj_img = obj_imgs[-1, ...].cpu().detach().numpy()
            # obj_img = obj_img.transpose(1, 2, 0)
            # obj_img = obj_img.astype(np.uint8)
            # context_img = context_imgs[-1, ...].cpu().detach().numpy()
            # context_img = context_img.transpose(1, 2, 0)
            # context_img = context_img.astype(np.uint8)
            # from PIL import Image
            # Image.fromarray(obj_img).show()
            # Image.fromarray(context_img).show()
            # t=y

            latent_caps = latent_caps.to(device=device, dtype=torch.float32)
            obj_imgs = obj_imgs.to(device=device, dtype=torch.float32)
            context_imgs = context_imgs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)

            # apply the lstm model on the latent caps
            h = lstm_3d.init_hidden(batch_size=1)
            output, h = lstm_3d(latent_caps, h)

            # package the features for the object and context 3D shapes. first make sure there is context at all.
            if len(output[0, :-1, :]) > 0:
                context_3d = torch.mean(output[0, :-1, :], dim=0).unsqueeze_(dim=0)
            else:
                context_3d = output[0, -1:, :]
            shape_features = torch.cat([output[0, -1:, :], context_3d], dim=1)

            # apply the pretrained resent model to the object and context images
            obj_img_features = wide_resnet50_2(obj_imgs)
            context_img_features = wide_resnet50_2(context_imgs)

            # concat the img features for object and context images.
            img_features = torch.cat([obj_img_features, context_img_features], dim=1)

            # prepend the 3d shape features to the img features
            all_features = torch.cat([shape_features, img_features], dim=0)
            all_features.unsqueeze_(dim=0)

            # apply the lstm model on the full features.
            output, h = lstm_joint(all_features, h)

            # apply a fc layer to predict the corresponding img to the 3d object. note to exclude the shape feature
            correspondence = lin_layer(output[0, 1:, :])
            correspondence.unsqueeze_(dim=0)

            # compute loss
            loss = criterion(correspondence, labels)

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # keep track of the losses
            epoch_loss += loss.item()
            num_samples += 1

            # compute validation loss to pick the best model
            if num_samples % eval_itr == 0:
                validation_loss = evaluate_net(model_dic, wide_resnet50_2, valid_loader, criterion, device=device)
                validation_losses.append(validation_loss)
                if validation_loss < best_validation_loss:
                    best_model_dic = copy.deepcopy(model_dic)
                    best_validation_loss = validation_loss
                    best_epoch = epoch
                    bad_epoch = 0
                else:
                    bad_epoch += 1
                # set the models back to training mode
                for model_name, model in model_dic.items():
                    model.train()

        # save model from every nth epoch
        print('Epoch %d finished! - Loss: %.10f' % (epoch + 1, epoch_loss / num_samples))
        print('-' * 50)
        if save_cp and ((epoch + 1) % 20 == 0):
            for model_name, model in model_dic.items():
                torch.save(model.state_dict(), os.path.join(model_dir, 'CP_{}_{}.pth'.format(model_name, epoch + 1)))
            print('Checkpoint %d saved !' % (epoch + 1))
        training_losses.append(epoch_loss / num_samples)

    # save train/valid losses and the best model
    if save_cp:
        np.save(os.path.join(model_dir, 'training_loss'), training_losses)
        np.save(os.path.join(model_dir, 'valid_loss'), validation_losses)
        for model_name, model in best_model_dic.items():
            torch.save(model.state_dict(), os.path.join(model_dir, 'CP_{}_best.pth'.format(model_name)))

    # report the attributes for the best model
    print('Best Validation loss is {}'.format(best_validation_loss))
    print('Best model comes from epoch: {}'.format(best_epoch))


def get_args():
    parser = OptionParser()
    parser.add_option('--epochs', dest='epochs', default=200, type='int', help='number of epochs')
    parser.add_option('--scene_graph_dir', dest='scene_graph_dir',
                      default='../../results/matterport3d/LearningBased/scene_graphs_cl')
    parser.add_option('--scene_to_pos_negatives', dest='scene_to_pos_negatives',
                      default='../../results/matterport3d/LearningBased/scene_to_pos_negatives.json')
    parser.add_option('--imgs_dir', dest='imgs_dir',
                      default='../../results/matterport3d/LearningBased/positive_negative_imgs')
    parser.add_option('--models_dir', dest='models_dir', default='../../data/matterport3d/models')
    parser.add_option('--latent_caps_dir', dest='latent_caps_dir',
                      default='../../../3D-point-capsule-networks/dataset/matterport3d/latent_caps')
    parser.add_option('--hidden_dim', dest='hidden_dim', default=1024, type='int')
    parser.add_option('--patience', dest='patience', default=20, type='int')
    parser.add_option('--eval_iter', dest='eval_iter', default=1000, type='int')
    parser.add_option('--cp_folder', dest='cp_folder', default='lstm_obj_correspondence')
    parser.add_option('--input_dim_3d', dest='input_dim_3d', default=4096)
    parser.add_option('--input_dim_joint', dest='input_dim_joint', default=4096)
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
    train_net(scene_graph_dir=args.scene_graph_dir, imgs_dir=args.imgs_dir, scene_to_pos_negatives=args.scene_to_pos_negatives,
              models_dir=args.models_dir, latent_caps_dir=args.latent_caps_dir, num_epochs=args.epochs, lr=1e-3,
              device=device, hidden_dim=args.hidden_dim, patience=args.patience, eval_itr=args.eval_iter,
              cp_folder=args.cp_folder, input_dim_3d=args.input_dim_3d, input_dim_joint=args.input_dim_joint)
    t2 = time()
    print("Training took %.3f minutes" % ((t2 - t) / 60))


if __name__ == '__main__':
    main()




