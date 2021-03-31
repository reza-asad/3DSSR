import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from time import time
from optparse import OptionParser
import torchvision.models as models

from scripts.helper import load_from_json, write_to_json
from object_dataset import Object
from models import Lstm, CorrClassifier
from scripts.box import Box
from scripts.iou import IoU


def translate_obbox(obbox, translation):
    # build the transformation matrix
    transformation = np.eye(4)
    transformation[:3, 3] = translation

    # apply tranlsation to the obbox
    obbox = obbox.apply_transformation(transformation)

    return obbox


def find_correspondence(query_info, scene_graph_dir, imgs_dir, scene_to_pos_negatives, models_dir, latent_caps_dir,
                        mode, model_dic, wide_resnet50_2, device):
    # load the query info
    query_scene_name = query_info['example']['scene_name']
    query_graph = load_from_json(os.path.join(scene_graph_dir, mode, query_scene_name))
    query_node = query_info['example']['query']
    context_objects = query_info['example']['context_objects']
    query_and_context = [query_node] + context_objects

    # create a data loader
    valid_dataset = Object(scene_graph_dir, imgs_dir, scene_to_pos_negatives, models_dir, latent_caps_dir, mode=mode,
                           query_scene_name=query_scene_name, query_nodes=query_and_context)
    loader = DataLoader(valid_dataset, batch_size=1, num_workers=0, shuffle=False)

    # apply the models to valid/test data
    for i, data in enumerate(loader):
        # load data
        candidates = [e[0] for e in data['candidates']]
        latent_caps = data['latent_caps']
        obj_imgs = data['obj_imgs'][0, ...]
        context_imgs = data['context_imgs'][0, ...]
        target_file_name = data['file_name'][0]

        # print(candidates)
        # print(len(latent_caps), latent_caps[0].shape)
        # print(obj_imgs.shape)
        # print(context_imgs.shape)
        # print(target_file_name)
        target_graph = load_from_json(os.path.join(scene_graph_dir, mode, target_file_name))
        # t=y

        # set the images on the right device
        obj_imgs = obj_imgs.to(device=device, dtype=torch.float32)
        context_imgs = context_imgs.to(device=device, dtype=torch.float32)

        # apply the pretrained resent model to the object and context images
        obj_img_features = wide_resnet50_2(obj_imgs)
        context_img_features = wide_resnet50_2(context_imgs)

        # concat the img features for object and context images.
        img_features = torch.cat([obj_img_features, context_img_features], dim=1)

        # feed each latent cap to the lstm models and predict if they correspond to the query objects.
        correspondence = {n: [] for n in query_and_context}
        for j, candidate in enumerate(candidates):
            # set the latent caps on the right device for this candidate
            latent_caps_c = latent_caps[j].to(device=device, dtype=torch.float32)

            # apply the lstm model on the latent caps
            h = model_dic['lstm_3d'].init_hidden(batch_size=1)
            output, h = model_dic['lstm_3d'](latent_caps_c, h)

            # package the features for the object and context 3D shapes. first make sure there is context at all.
            if len(output[0, :-1, :]) > 0:
                context_3d = torch.mean(output[0, :-1, :], dim=0).unsqueeze_(dim=0)
            else:
                context_3d = output[0, -1:, :]
            shape_features = torch.cat([output[0, -1:, :], context_3d], dim=1)

            # prepend the 3d shape features to the img features
            all_features = torch.cat([shape_features, img_features], dim=0)
            all_features.unsqueeze_(dim=0)

            # apply the lstm model on the full features.
            output, h = model_dic['lstm_joint'](all_features, h)

            # apply a fc layer to predict the corresponding img to the 3d object. note to exclude the shape feature
            match_prob = model_dic['lin_layer'](output[0, 1:, :])
            match_prob.unsqueeze_(dim=0)
            idx = torch.argmax(torch.sigmoid(match_prob[0, :, 0])).item()
            print(query_graph[str(idx)]['category'][0], target_graph[candidate]['category'][0])
        t=y

        query_info['category_correspondence'] = correspondence
        t=y


def get_args():
    parser = OptionParser()
    parser.add_option('--mode', dest='mode', default='val', help='val or test')
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
    parser.add_option('--experiment_name', dest='experiment_name', default='lstm_obj_correspondence')
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

    # initialize the models and set them on the right device
    lstm_3d = Lstm(args.input_dim_3d, args.hidden_dim, device)
    lstm_joint = Lstm(args.input_dim_joint, args.hidden_dim, device)
    lin_layer = CorrClassifier(args.hidden_dim)
    wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
    wide_resnet50_2.fc = torch.nn.Linear(2048, args.input_dim_joint//2, bias=False)

    # set models to the right device
    lstm_3d = lstm_3d.to(device=device)
    lstm_joint = lstm_joint.to(device=device)
    lin_layer = lin_layer.to(device=device)
    wide_resnet50_2 = wide_resnet50_2.to(device=device)

    # load the models and set the models to evaluation mode
    checkpoint_dir = '../../results/matterport3d/LearningBased/{}'.format(args.experiment_name)
    model_dic = {'resnet_fc': wide_resnet50_2.fc, 'lstm_3d': lstm_3d, 'lstm_joint': lstm_joint, 'lin_layer': lin_layer}
    for model_name, model in model_dic.items():
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'CP_{}_best.pth'.format(model_name))))
        model.eval()
    wide_resnet50_2.eval()

    # set the input and output paths for the query dict.
    query_dict_input_path = '../../queries/matterport3d/query_dict_{}.json'.format(args.mode)
    query_dict_output_path = '../../results/matterport3d/LearningBased/query_dict_{}_{}.json'.format(args.mode,
                                                                                                     args.experiment_name)

    # read the query dict
    query_dict = load_from_json(query_dict_input_path)

    # apply scene alignment for each query
    t0 = time()
    for i, (query, query_info) in enumerate(query_dict.items()):
        t = time()
        print('Iteration {}/{}'.format(i+1, len(query_dict)))
        print('Processing query: {}'.format(query))
        find_correspondence(query_info, args.scene_graph_dir, args.imgs_dir, args.scene_to_pos_negatives,
                            args.models_dir, args.latent_caps_dir, args.mode, model_dic, wide_resnet50_2, device)
        duration = (time() - t) / 60
        print('Processing the query took {} minutes'.format(round(duration, 2)))
    duration_all = (time() - t0) / 60
    print('Processing all queries too {} minutes'.format(round(duration_all, 2)))

    # save the changes to query dict
    write_to_json(query_dict, query_dict_output_path)


if __name__ == '__main__':
    main()

