import os
import torch
from torch.utils.data import DataLoader
import numpy as np

from scripts.helper import load_from_json, write_to_json
from ring_dataset import RingDataset
from gnn_models import LinearLayer, Discriminator


def extract_features(graph, source_node, nb, edge_type):
    # extract cat features
    source_cat_feature = graph[source_node]['cat_vec']
    nb_cat_feature = graph[nb]['cat_vec']

    # extract distance and iou for the neighbour node
    edge_type_distances = graph[source_node]['ring_info'][edge_type]['distance']
    distance_feature = [dist for n, dist in edge_type_distances if n == nb]
    edge_type_ious = graph[source_node]['ring_info'][edge_type]['iou']
    iou_feature = [iou for n, iou in edge_type_ious if n == nb]

    # concat features and create the edge feature
    nb_feature = nb_cat_feature + distance_feature + iou_feature
    feature = source_cat_feature + nb_feature

    # convert the features to torch
    feature = torch.from_numpy(np.asarray(feature, dtype=np.float))

    return feature


def apply_lin_layer(query_info, model_names, data_dir, checkpoint_dir, hidden_dim, device, mode):
    # create data loader
    dataset = RingDataset(data_dir, mode=mode)
    loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)

    # load the saved models
    input_dim = list(enumerate(loader))[1][1]['features'].shape[-1]
    lin_layer = LinearLayer(input_dim=input_dim, output_dim=hidden_dim)
    disc = Discriminator(hidden_dim=hidden_dim)
    lin_layer = lin_layer.to(device=device)
    disc = disc.to(device=device)

    lin_layer.load_state_dict(torch.load(os.path.join(checkpoint_dir, model_names['lin_layer'])))
    disc.load_state_dict(torch.load(os.path.join(checkpoint_dir, model_names['disc'])))

    disc.eval()
    lin_layer.eval()

    # extract the positive using the query subgraph
    query_scene_name = query_info['example']['scene_name']
    query_node = query_info['example']['query']
    context_objects = query_info['example']['context_objects']
    query_scene = load_from_json(os.path.join(data_dir, mode, query_scene_name))
    positives = torch.zeros((1, len(context_objects) + 1, input_dim))
    for i, nb in enumerate(context_objects):
        edge_type = query_scene[query_node]['neighbours'][nb][0]
        feature = extract_features(query_scene, query_node, nb, edge_type)
        positives[:, i, :] = feature
    positives = positives.to(device=device)

    # apply linear layer and find the mean summary of the positive features
    hidden_pos = lin_layer(positives)
    gated_mean = torch.sigmoid(torch.mean(hidden_pos[:, :, :], dim=1))

    # apply the models to valid/test data
    for i, data in enumerate(loader):
        features = data['features']
        features = features.to(device=device)

        num_nodes = features.shape[1]
        for j in range(num_nodes):
            # apply the discriminator to positive and negative examples.
            print(features[:, j, :].shape)
            logits = disc(gated_mean, features[:, j, :])
            print(logits)
            t=y

    return 1


def main():
    mode = 'val'
    experiment_name = 'LinearLayer'
    query_dict_input_path = '../../queries/matterport3d/query_dict_{}.json'.format(mode)
    query_dict_output_path = '../../results/matterport3d/GNN/query_dict_{}_{}.json'.format(mode, experiment_name)
    model_names = {'lin_layer': 'CP_lin_layer_50.pth',
                   'disc': 'CP_disc_50.pth'}
    ring_data_dir = '../../results/matterport3d/GNN/scene_graphs_cl'
    checkpoints_dir = '../../results/matterport3d/GNN/subring_matching'
    hidden_dim = 256
    device = torch.device('cuda')

    # read the query dict
    query_dict = load_from_json(query_dict_input_path)

    # apply subgraph matching for each query
    for query, query_info in query_dict.items():
        target_subscenes = apply_lin_layer(query_info, model_names, ring_data_dir, checkpoints_dir, hidden_dim,
                                           device, mode)
        query_info['target_subscenes'] = target_subscenes


if __name__ == '__main__':
    main()

