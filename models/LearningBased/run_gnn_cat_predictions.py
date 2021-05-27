import os
import torch
from torch.utils.data import DataLoader
from time import time
from optparse import OptionParser

from scripts.helper import load_from_json, write_to_json
from graph_dataset import Scene
from models import GCN_RES, Classifier
from train_gnn import compute_accuracy

alpha = 1
gamma = 2.


def run_gnn(scene_graph_dir, latent_caps_dir, output_dir, cat_to_idx, hidden_dim, num_layers, cp_dir, mode, device,
            topk=5):
    # create the output directory if it doesn't already exist.
    scene_graph_dir_out = os.path.join(output_dir, mode)
    if not os.path.exists(scene_graph_dir_out):
        os.makedirs(scene_graph_dir_out)

    # create dataset and loader
    dataset = Scene(scene_graph_dir, latent_caps_dir, cat_to_idx, mode=mode)
    loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)

    # find the required dimensions for building the GNN.
    sample_data = list(enumerate(loader))[1][1]
    input_dim = sample_data['features'].shape[-1]
    num_edge_types = sample_data['adj'].shape[1]

    # initialize the models and set them on the right device
    gnn = GCN_RES(input_dim, hidden_dim, 'prelu', num_edge_types, num_layers=num_layers)
    lin_layer = Classifier(hidden_dim, len(cat_to_idx))
    gnn = gnn.to(device=device)
    lin_layer = lin_layer.to(device=device)

    # load the best models from training
    model_dic = {'gnn': gnn, 'lin_layer': lin_layer}
    for model_name, model in model_dic.items():
        model.load_state_dict(torch.load(os.path.join(cp_dir, 'CP_{}_best.pth'.format(model_name))))

    # set models to evaluation mode
    for model_name, model in model_dic.items():
        model.eval()

    # find a mapping from the index categories to the categories. use this to save the actual predicted categories.
    idx_to_cats = {idx: cat for cat, idx in cat_to_idx.items()}

    per_class_accuracy = {cat: (0, 0) for cat in cat_to_idx.keys()}
    with torch.no_grad():
        for i, data in enumerate(loader):
            # load data
            features = data['features'].squeeze(dim=0)
            labels = data['labels'].squeeze(dim=0)
            adj = data['adj'].squeeze(dim=0)
            file_name = data['file_name'][0]

            # move data to the right device
            features = features.to(device=device, dtype=torch.float32)
            adj = adj.to(device=device, dtype=torch.float)
            labels = labels.to(device=device, dtype=torch.long)

            # apply the model
            H = model_dic['gnn'](features, adj)
            output = model_dic['lin_layer'](H)
            softmax_output = torch.softmax(output, dim=1)

            # compute predictions
            _, predictions = torch.max(softmax_output, dim=1)
            compute_accuracy(per_class_accuracy, predictions, labels, cat_to_idx)

            # find the topk predictions.
            _, topk_predictions = torch.topk(softmax_output, topk, dim=1)

            # load the graph and sort its objects to match the order of the predictions
            graph = load_from_json(os.path.join(scene_graph_dir, mode, file_name))
            objects = list(graph.keys())
            objects = sorted(objects, key=int)

            # map the tok predictions to the actual category and save it in the graph.
            for j, obj in enumerate(objects):
                graph[obj]['category_predicted'] = [idx_to_cats[idx.item()] for idx in topk_predictions[j, :]]

            # save the graph with predicted categories.
            # write_to_json(graph, os.path.join(scene_graph_dir_out, file_name))

    # display per class accuracy
    per_class_accuracy_final = {}
    for c, (num_correct, num_total) in per_class_accuracy.items():
        if num_total != 0:
            accuracy = float(num_correct) / num_total
            print(c, accuracy)
            per_class_accuracy_final[c] = accuracy

    # save the per class accuracies.
    # write_to_json(per_class_accuracy_final, os.path.join(cp_dir, 'per_class_accuracy.json'))


def get_args():
    parser = OptionParser()
    parser.add_option('--mode', dest='mode', default='test', help='val or test')
    parser.add_option('--accepted_cats_path', dest='accepted_cats_path',
                      default='../../data/matterport3d/accepted_cats.json')
    parser.add_option('--scene_graph_dir', dest='scene_graph_dir',
                      default='../../results/matterport3d/LearningBased/scene_graphs')
    parser.add_option('--latent_caps_dir', dest='latent_caps_dir',
                      default='../../../3D-point-capsule-networks/dataset/matterport3d/latent_caps')
    parser.add_option('--output_dir', dest='output_dir',
                      default='../../results/matterport3d/LearningBased/scene_graphs_with_predictions_gnn')
    parser.add_option('--hidden_dim', dest='hidden_dim', default=1024, type='int')
    parser.add_option('--num_layers', dest='num_layers', default=1, type='int')
    parser.add_option('--cp_dir', dest='cp_dir',
                      default='../../results/matterport3d/LearningBased/gnn_cat_prediction')
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

    # prepare the accepted categories for training.
    accepted_cats = load_from_json(args.accepted_cats_path)
    accepted_cats = sorted(accepted_cats)
    cat_to_idx = {cat: i for i, cat in enumerate(accepted_cats)}

    # apply scene alignment for each query
    t0 = time()
    run_gnn(scene_graph_dir=args.scene_graph_dir, latent_caps_dir=args.latent_caps_dir, output_dir=args.output_dir,
            cat_to_idx=cat_to_idx, hidden_dim=args.hidden_dim, num_layers=args.num_layers, cp_dir=args.cp_dir,
            mode=args.mode, device=device)
    duration_all = (time() - t0) / 60
    print('Processing all scenes took {} minutes'.format(round(duration_all, 2)))


if __name__ == '__main__':
    main()

