import os
from optparse import OptionParser
from time import time
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from region_dataset import Region
from transformations import PointcloudToTensor
from projection_models import MLP
from capsnet_models import PointCapsNet
from scripts.helper import load_from_json, write_to_json
from train_linear_classifier import evaluate_net

alpha = 1
gamma = 2.


def run_classifier(device, cat_to_idx, args):
    # create a list of transformations to be applied to the point cloud.
    transform = transforms.Compose([
        PointcloudToTensor(),
    ])

    # create the training dataset
    dataset = Region(args.data_dir, args.scene_dir, num_local_crops=args.num_local_crops,
                     num_global_crops=args.num_global_crops, mode=args.mode, cat_to_idx=cat_to_idx,
                     transforms=transform)

    # create the dataloaders
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)

    # load the pre-trained capsulenet and set it to the right device
    capsule_net = PointCapsNet(1024, 16, 64, 64, args.num_points)
    capsule_net.load_state_dict(torch.load(args.best_capsule_net))
    capsule_net = capsule_net.to(device=device)
    capsule_net.eval()

    # load the model and set it to the right device
    lin_layer = MLP(64*64, args.hidden_dim, len(cat_to_idx))
    lin_layer = lin_layer.to(device=device)

    # load the best models from training
    model_dic = {'lin_layer': lin_layer}
    for model_name, model in model_dic.items():
        model.load_state_dict(torch.load(os.path.join(args.cp_dir, 'CP_{}_best.pth'.format(model_name))))

    # set models to evaluation mode
    for model_name, model in model_dic.items():
        model.eval()

    # evaluate the model
    _, per_class_accuracy = evaluate_net(model_dic, capsule_net, loader, cat_to_idx, device)

    # save the per class accuracies.
    per_class_accuracy_final = {}
    for c, (num_correct, num_total) in per_class_accuracy.items():
        if num_total != 0:
            accuracy = float(num_correct) / num_total
            per_class_accuracy_final[c] = accuracy
    write_to_json(per_class_accuracy_final, os.path.join(args.cp_dir, 'per_class_accuracy.json'))


def get_args():
    parser = OptionParser()
    parser.add_option('--mode', dest='mode', default='val')
    parser.add_option('--accepted_cats_path', dest='accepted_cats_path',
                      default='../../data/matterport3d/accepted_cats.json')
    parser.add_option('--data_dir', dest='data_dir',
                      default='../../data/matterport3d/mesh_regions')
    parser.add_option('--scene_dir', dest='scene_dir', default='../../data/matterport3d/scenes')
    parser.add_option('--cp_dir', dest='cp_dir',
                      default='../../results/matterport3d/LearningBased/region_classification_capsnet_linear')
    parser.add_option('--best_capsule_net', dest='best_capsule_net',
                      default='../../results/matterport3d/LearningBased/3D_DINO_exact_regions/best_capsule_net.pth')

    parser.add_option('--num_points', dest='num_points', default=4096, type='int')
    parser.add_option('--num_local_crops', dest='num_local_crops', default=0, type='int')
    parser.add_option('--num_global_crops', dest='num_global_crops', default=0, type='int')

    parser.add_option('--batch_size', dest='batch_size', default=7, type='int')
    parser.add_option('--hidden_dim', dest='hidden_dim', default=1024, type='int')

    parser.add_option('--gpu', action='store_true', dest='gpu', default=True, help='use cuda')
    parser.add_option('--save_cp', action='store_true', dest='save_cp', default=True, help='save the trained models')

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
    run_classifier(device, cat_to_idx, args)
    t2 = time()
    print("Training took %.3f minutes" % ((t2 - t) / 60))


if __name__ == '__main__':
    main()



