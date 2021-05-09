import os
from optparse import OptionParser
from time import time
import torch
from torch.utils.data import DataLoader

from graph_dataset import Scene
from models import MLP
from scripts.helper import load_from_json, write_to_json
from train_classifier import evaluate_net

alpha = 1
gamma = 2.


def run_classifier(scene_graph_dir, latent_caps_dir, cat_to_idx, device, hidden_dim, cp_dir, mode):
    # create dataset and loader
    dataset = Scene(scene_graph_dir, latent_caps_dir, cat_to_idx, mode=mode)
    loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

    # find the required dimensions for building the classifier.
    sample_data = list(enumerate(loader))[1][1]
    input_dim = sample_data['features'].shape[-1]

    # initialize the models and set them on the right device
    lin_layer = MLP(input_dim, hidden_dim, len(cat_to_idx))
    lin_layer = lin_layer.to(device=device)

    # load the best models from training
    model_dic = {'lin_layer': lin_layer}
    for model_name, model in model_dic.items():
        model.load_state_dict(torch.load(os.path.join(cp_dir, 'CP_{}_best.pth'.format(model_name))))

    # set models to evaluation mode
    for model_name, model in model_dic.items():
        model.eval()

    # evaluate the model
    _, per_class_accuracy = evaluate_net(model_dic, loader, cat_to_idx, device=device)

    # save the per class accuracies.
    per_class_accuracy_final = {}
    for c, (num_correct, num_total) in per_class_accuracy.items():
        if num_total != 0:
            accuracy = float(num_correct) / num_total
            per_class_accuracy_final[c] = accuracy
    write_to_json(per_class_accuracy_final, os.path.join(cp_dir, 'per_class_accuracy.json'))


def get_args():
    parser = OptionParser()
    parser.add_option('--mode', dest='mode', default='test', help='val or test')
    parser.add_option('--accepted_cats_path', dest='accepted_cats_path',
                      default='../../data/matterport3d/accepted_cats.json')
    parser.add_option('--scene_graph_dir', dest='scene_graph_dir',
                      default='../../results/matterport3d/LearningBased/scene_graphs_cl')
    parser.add_option('--latent_caps_dir', dest='latent_caps_dir',
                      default='../../../3D-point-capsule-networks/dataset/matterport3d/latent_caps')
    parser.add_option('--hidden_dim', dest='hidden_dim', default=1024, type='int')
    parser.add_option('--cp_dir', dest='cp_dir',
                      default='../../results/matterport3d/LearningBased/linear_cat_prediction')
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
    run_classifier(scene_graph_dir=args.scene_graph_dir, latent_caps_dir=args.latent_caps_dir, cat_to_idx=cat_to_idx,
                   device=device,  hidden_dim=args.hidden_dim, cp_dir=args.cp_dir, mode=args.mode)
    t2 = time()
    print("Training took %.3f minutes" % ((t2 - t) / 60))


if __name__ == '__main__':
    main()




