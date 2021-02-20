import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter
from time import time

from scripts.helper import load_from_json, write_to_json
from ring_dataset import RingDataset
from gnn_models import LinearLayer, GCN_RES, Discriminator
from train_subring_matching import normalize_adj_mp


def extract_edge_feature(graph, source_node, nb, edge_type):
    # extract cat features
    nb_cat_feature = graph[nb]['cat_vec']

    # extract distance and iou for the neighbour node
    edge_type_distances = graph[source_node]['ring_info'][edge_type]['distance']
    distance_feature = [dist for n, dist in edge_type_distances if n == nb]
    edge_type_ious = graph[source_node]['ring_info'][edge_type]['iou']
    iou_feature = [iou for n, iou in edge_type_ious if n == nb]

    # extract direction
    direction = np.asarray(graph[source_node]['obbox'][0]) - np.asarray(graph[nb]['obbox'][0])

    # concat features and create the edge feature
    nb_feature = nb_cat_feature + distance_feature + iou_feature + direction.tolist()

    # convert the features to torch
    nb_feature = torch.from_numpy(np.asarray(nb_feature, dtype=np.float))

    return nb_feature


def find_best_context_objects(data_dir, mode, file_name, source_node_idx, gates, context_objects_cats, query_cat):
    # load the target scene and sort the object ids
    target_scene = load_from_json(os.path.join(data_dir, mode, file_name))
    objects = target_scene.keys()
    objects = sorted(objects, key=int)
    objects_cats = [target_scene[obj]['category'][0] for obj in objects]

    # if target and query node have different categories skip
    t_node = objects[source_node_idx]
    if target_scene[t_node]['category'][0] != query_cat:
        return None, [], []

    # map the category of context objects in the query scene to their frequency
    query_cat_to_freq = Counter(context_objects_cats)

    # map the category of the nodes in the target scene to their gate and node id
    target_cat_to_gates = {}
    for i, cat in enumerate(objects_cats):
        if cat not in target_cat_to_gates:
            target_cat_to_gates[cat] = [(gates[0, i].item(), objects[i], i)]
        else:
            target_cat_to_gates[cat].append((gates[0, i].item(), objects[i], i))
    # for each cat sort the candidates by the gate value
    for cat, val in target_cat_to_gates.items():
        target_cat_to_gates[cat] = sorted(val, key=lambda x: x[0], reverse=True)

    # select the best context objects
    gates_nodes_indices = []
    for cat, val in target_cat_to_gates.items():
        if cat in query_cat_to_freq:
            # case where there are more of the object category in the query. take all the target scene offers.
            if len(val) < query_cat_to_freq[cat]:
                gates_nodes_indices += val
            else:
                # case where there are more of the object category in the target scene. choose based on highest gate
                # value.
                num_available = query_cat_to_freq[cat]
                gates_nodes_indices += val[:num_available]

    # extract the object ids and indices for the best context objects
    context_objects = []
    context_objects_indices = set()
    for _, context_object, context_object_idx in gates_nodes_indices:
        context_objects.append(context_object)
        context_objects_indices.add(context_object_idx)

    # find the indices of the non-context objects
    non_context_objects_indices = [i for i in range(len(objects)) if i not in context_objects_indices]

    return t_node, context_objects, non_context_objects_indices


def apply_ring_gnn(query_info, model_names, data_dir, checkpoint_dir, hidden_dim, num_layers, device, mode):
    # load the query info
    query_scene_name = query_info['example']['scene_name']
    query_node = query_info['example']['query']
    context_objects = query_info['example']['context_objects']
    query_scene = load_from_json(os.path.join(data_dir, mode, query_scene_name))

    # find the category of the query node and the context objects
    query_cat = query_scene[query_node]['category'][0]
    context_objects_cats = [query_scene[context_object]['category'][0] for context_object in context_objects]

    # create data loader
    dataset = RingDataset(data_dir, mode=mode, with_query_ring=True, query_cat=query_cat)
    loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

    # load the saved models
    sample_data = list(enumerate(loader))[1][1]

    input_dim = sample_data['features'].shape[-1]
    num_edge_type = sample_data['adj'].shape[2]
    lin_layer = LinearLayer(input_dim=input_dim, output_dim=hidden_dim)
    gcn_res = GCN_RES(input_dim, hidden_dim, 'prelu', num_edge_type, num_layers)
    disc = Discriminator(hidden_dim=hidden_dim)

    lin_layer = lin_layer.to(device=device)
    gcn_res = gcn_res.to(device=device)
    disc = disc.to(device=device)

    lin_layer.load_state_dict(torch.load(os.path.join(checkpoint_dir, model_names['lin_layer'])))
    gcn_res.load_state_dict(torch.load(os.path.join(checkpoint_dir, model_names['gcn_res'])))
    disc.load_state_dict(torch.load(os.path.join(checkpoint_dir, model_names['disc'])))

    lin_layer.eval()
    gcn_res.eval()
    disc.eval()

    # extract the features for the query node
    positives = torch.zeros((1, len(context_objects), input_dim))
    for i, nb in enumerate(context_objects):
        edge_type = query_scene[query_node]['neighbours'][nb][0]
        feature = extract_edge_feature(query_scene, query_node, nb, edge_type)
        positives[:, i, :] = feature
    positives = positives.to(device=device)

    # apply linear layer and find the mean summary of the positive features
    hidden_pos = lin_layer(positives)
    pos_summary = torch.sigmoid(torch.mean(hidden_pos[:, :, :], dim=1))

    # apply the models to valid/test data
    target_subscenes = []
    for i, data in enumerate(loader):
        # read the features, label and adj
        features = data['features']
        adj = data['adj']
        file_name = data['file_name'][0]
        source_node_indices = data['source_node_idx']

        # skip if the file name is the same as the query scene name
        if file_name == query_scene_name:
            continue

        features = features.to(device=device, dtype=torch.float32)
        adj = adj.to(device=device, dtype=torch.float32)

        num_nodes = features.shape[1]
        for j in range(num_nodes):
            # add self loops and normalize the adj
            nb_nodes = adj.shape[-1]
            adj_j = normalize_adj_mp(adj[:, j, :, :, :], nb_nodes, device)

            # apply gnn on the full ring
            hidden = gcn_res(features[:, j, :, :], adj_j)

            # apply the discriminator to separate the positive and negative node embeddings.
            logits_j = disc(pos_summary, hidden)
            gates = torch.sigmoid(logits_j)

            # pick the context objects based on the gates
            source_node_idx = int(source_node_indices[0, j])
            t_node, t_context_objects, non_context_objects_indices = find_best_context_objects(data_dir,
                                                                                               mode,
                                                                                               file_name,
                                                                                               source_node_idx,
                                                                                               gates,
                                                                                               context_objects_cats,
                                                                                               query_cat)

            # Add the target subscene only if target node was found
            if t_node is not None:
                # zero out the source node and non-context objects
                gates[0, source_node_idx] = 0
                gates[0, non_context_objects_indices] = 0

                # compute the gated mean of the features for the given ring
                hidden_target = lin_layer(features[:, j, :, :])
                hidden_target = hidden_target[0, :, :] * gates[:, :].t()
                target_summary = torch.sigmoid(torch.mean(hidden_target, dim=0))

                # record the target summary vector
                sim = torch.dot(target_summary, pos_summary[0]) / (torch.norm(target_summary) * torch.norm(pos_summary[0]))
                target_subscene = {'scene_name': file_name, 'target': t_node, 'context_objects': t_context_objects,
                                   'sim': sim.item()}
                target_subscenes.append(target_subscene)

    # sort the target subscenes based on how close each target embedding is to the positive summary
    target_subscenes = sorted(target_subscenes, reverse=True, key=lambda x: x['sim'])

    return target_subscenes


def main():
    mode = 'val'
    experiment_name = 'cat_dir'
    query_dict_input_path = '../../queries/matterport3d/query_dict_{}.json'.format(mode)
    query_dict_output_path = '../../results/matterport3d/GNN/query_dict_{}_{}.json'.format(mode, experiment_name)
    model_names = {'lin_layer': 'CP_lin_layer_best.pth',
                   'gcn_res': 'CP_gcn_res_best.pth',
                   'disc': 'CP_disc_best.pth'}
    ring_data_dir = '../../results/matterport3d/GNN/scene_graphs_cl'
    checkpoints_dir = '../../results/matterport3d/GNN/subring_matching'
    hidden_dim = 256
    num_layers = 2
    device = torch.device('cuda')

    # read the query dict
    query_dict = load_from_json(query_dict_input_path)

    # apply subgraph matching for each query
    t0 = time()
    for i, (query, query_info) in enumerate(query_dict.items()):
        t = time()
        print('Iteration {}/{}'.format(i+1, len(query_dict)))
        print('Processing query: {}'.format(query))
        target_subscenes = apply_ring_gnn(query_info, model_names, ring_data_dir, checkpoints_dir, hidden_dim,
                                          num_layers, device, mode)
        query_info['target_subscenes'] = target_subscenes
        duration = (time() - t) / 60
        print('Processing the query took {} minutes'.format(round(duration, 2)))
    duration_all = (time() - t0) / 60
    print('Processing all queries too {} minutes'.format(round(duration_all, 2)))

    # save the changes to query dict
    write_to_json(query_dict, query_dict_output_path)


if __name__ == '__main__':
    main()

