import os
import torch
from torch.utils.data import Dataset
import numpy as np

from scripts.helper import load_from_json


class RingDataset(Dataset):
    def __init__(self, data_dir, mode, with_query_ring=False, query_cat=None, low_prob=0.3, high_prob=0.8,
                 negative_ratio=0.5):
        self.data_dir = data_dir
        self.mode = mode
        self.with_query_ring = with_query_ring
        self.query_cat = query_cat
        if with_query_ring and query_cat is not None:
            self.file_names = self.filter_by_query_cat(os.listdir(os.path.join(self.data_dir, self.mode)))
        else:
            self.file_names = os.listdir(os.path.join(self.data_dir, self.mode))

        self.low_prob = low_prob
        self.high_prob = high_prob
        self.negative_ratio = negative_ratio

    def __len__(self):
        return len(self.file_names)

    def filter_by_query_cat(self, file_names):
        filtered_file_names = []
        for file_name in file_names:
            graph = load_from_json(os.path.join(self.data_dir, self.mode, file_name))
            for _, node_info in graph.items():
                if node_info['category'][0] == self.query_cat:
                    filtered_file_names.append(file_name)
                    break
        return filtered_file_names

    @staticmethod
    def compute_feature_size(graph):
        # feature size is distance + iou
        feature_size = 1 + 1

        return feature_size

    @staticmethod
    def extract_edge_feature(graph, source_node, nb, edge_type):
        # extract distance and iou for the neighbour node
        edge_type_distances = graph[source_node]['ring_info'][edge_type]['distance']
        distance_feature = [dist for n, dist in edge_type_distances if n == nb]
        edge_type_ious = graph[source_node]['ring_info'][edge_type]['iou']
        iou_feature = [iou for n, iou in edge_type_ious if n == nb]

        # concat features and create the edge feature
        nb_feature = distance_feature + iou_feature

        # convert the features to torch
        nb_feature = torch.from_numpy(np.asarray(nb_feature, dtype=np.float))

        return nb_feature

    @staticmethod
    def build_adj(graph, source_node):
        # initialize the edge types
        N = len(graph)
        relation_dic = {'supports': 0, 'supported': 1, 'encloses': 2, 'enclosed': 3, 'contact': 4, 'fc': 5}

        # for consistency sort the graph nodes by their id
        objects = graph.keys()
        objects = sorted(objects, key=int)
        obj_to_index = {obj: idx for idx, obj in enumerate(objects)}

        # initialize the adj matrix
        adj = np.zeros((len(relation_dic), N, N), dtype=int)
        for nb, relations in graph[source_node]['neighbours'].items():
            # populate source-to-nb
            for relation in relations:
                idx = relation_dic[relation]
                adj[idx, obj_to_index[source_node], obj_to_index[nb]] = 1
            # populate nb-to-source
            reverse_relations = graph[nb]['neighbours'][source_node]
            for relation in reverse_relations:
                idx = relation_dic[relation]
                adj[idx, obj_to_index[nb], obj_to_index[source_node]] = 1

            # test the adj is symmetric for contact relation
            assert np.all(adj[4, :, :] == adj[4, :, :].transpose())

        return torch.from_numpy(adj)

    def draw_positive(self, graph, source_node, num_neighbours):
        # populate the ring with edge types and node ids
        ring = []
        ring_info = graph[source_node]['ring_info']
        for edge_type in ring_info.keys():
            node_cats = ring_info[edge_type]['cat']
            for node, _ in node_cats:
                ring.append((edge_type, node))

        # randomly choose a subring. the minimum number of neighbours that we sample is 1.
        subring_size = int(np.floor(np.random.uniform(self.low_prob, self.high_prob) * num_neighbours))
        subring_size = np.maximum(subring_size, 1)
        rand_idx = np.random.choice(len(ring), subring_size, replace=False)
        subring = np.asarray(ring)[rand_idx]

        # print('Ring size: ', len(ring))
        # print('Subring size: ', len(subring))
        # print('*' * 50)
        return subring

    def extract_all_edge_features(self, graph, source_node, objects):
        features = torch.zeros(len(objects), self.compute_feature_size(graph))
        for i, obj in enumerate(objects):
            # case where we add features for the source node
            if obj == source_node:
                # take the cat vec for source node and replicate it for nb. distance feature is 0 and iou is 1.
                feature = [0] + [1]
                feature = torch.from_numpy(np.asarray(feature))
            else:
                relations = graph[source_node]['neighbours'][obj]
                feature = self.extract_edge_feature(graph, source_node, obj, relations[0])
            features[i, :] = feature

        return features

    def __getitem__(self, idx):
        # load the graph
        graph_path = os.path.join(self.data_dir, self.mode, self.file_names[idx])
        graph = load_from_json(graph_path)
        num_edge_types = len(graph[list(graph.keys())[0]]['ring_info'])

        # for consistency sort the graph nodes by their id
        objects = graph.keys()
        objects = sorted(objects, key=int)
        obj_to_index = {obj: idx for idx, obj in enumerate(objects)}

        data = {'file_name': self.file_names[idx]}
        # if a query ring is not given draw random positives by randomly selecting a subring.
        if not self.with_query_ring:
            # choose a random source node and its associate ring
            source_node = np.random.choice(objects, 1)[0]

            # select a subring (positive neighbours)
            num_neighbours = len(objects) - 1
            subring = self.draw_positive(graph, source_node, num_neighbours)

            # extract all features for the given source node
            features = torch.zeros(1, len(objects), self.compute_feature_size(graph))
            features[0, :, :] = self.extract_all_edge_features(graph, source_node, objects)

            # extract adj matrix
            adj = self.build_adj(graph, source_node)
            adj.unsqueeze_(0)

            # add label to the data given the selected subring. source node is excluded.
            data['label'] = torch.zeros(1, len(objects))
            for _, node in subring:
                idx = obj_to_index[node]
                data['label'][:, idx] = 1

            # add the index of the source node
            data['source_node_idx'] = obj_to_index[source_node]
        else:
            # determine the potential source nodes; i.e nodes with the same category as query node
            source_nodes = [node for node in graph.keys() if graph[node]['category'][0] == self.query_cat]

            # extract features for all potential source nodes and the corresponding adj matrices
            features = torch.zeros(len(source_nodes), len(objects), self.compute_feature_size(graph))
            adj = torch.zeros(len(source_nodes), num_edge_types, len(objects), len(objects))
            data['source_node_idx'] = torch.zeros(len(source_nodes))
            for i, source_node in enumerate(source_nodes):
                features[i, :, :] = self.extract_all_edge_features(graph, source_node, objects)
                adj[i, :, :, :] = self.build_adj(graph, source_node)
                # record the index for the source node
                data['source_node_idx'][i] = obj_to_index[source_node]

        data['features'] = features
        data['adj'] = adj

        return data

