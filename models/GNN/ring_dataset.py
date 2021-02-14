import os
import torch
from torch.utils.data import Dataset
import numpy as np

from scripts.helper import load_from_json


class RingDataset(Dataset):
    def __init__(self, data_dir, mode, low_prob=0.3, high_prob=0.8, negative_ratio=0.5):
        self.data_dir = data_dir
        self.mode = mode
        if self.mode == 'train':
            self.file_names = self.filter_for_training(os.listdir(os.path.join(self.data_dir, self.mode)))
        else:
            self.file_names = os.listdir(os.path.join(self.data_dir, self.mode))

        self.low_prob = low_prob
        self.high_prob = high_prob
        self.negative_ratio = negative_ratio

    def __len__(self):
        return len(self.file_names)

    def filter_for_training(self, file_names):
        filtered_file_names = []
        for file_name in file_names:
            # load the graph for each file name
            graph_path = os.path.join(self.data_dir, self.mode, file_name)
            graph = load_from_json(graph_path)

            # only take the graph if it has at least 3 nodes + adj
            if len(graph.keys()) > 3:
                filtered_file_names.append(file_name)

        return filtered_file_names

    @staticmethod
    def compute_feature_size(graph, node):
        # feature size is 2 times the size of the one hot encoding of categories + distance + iou
        feature_size = 2 * len(graph[node]['cat_vec']) + 1 + 1

        return feature_size

    @staticmethod
    def extract_edge_feature(graph, source_node, nb, edge_type):
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

    def draw_positive(self, graph, source_node, num_neighbours):
        ring = []
        ring_info = graph[source_node]['ring_info']
        for edge_type in ring_info.keys():
            node_cats = ring_info[edge_type]['cat']
            for node, _ in node_cats:
                ring.append((edge_type, node))

        # Randomly choose a subring. There are at least two neighbours and one source node. The minimum number of
        # neighbours that we sample is 1 (when there are 2 neighbours only).
        subring_size = int(np.floor(np.random.uniform(self.low_prob, self.high_prob) * num_neighbours))
        subring_size = np.maximum(subring_size, 1)
        rand_idx = np.random.choice(len(ring), subring_size)
        subring = np.asarray(ring)[rand_idx]

        # extract the features for the subring
        positives = torch.zeros((len(subring), self.compute_feature_size(graph, subring[0][1])))
        for i, (edge_type, nb) in enumerate(subring):
            feature = self.extract_edge_feature(graph, source_node, nb, edge_type)
            positives[i, :] = feature

        # print('Ring size: ', len(ring))
        # print('Positive shape: ', positives.shape)
        # print('*' * 50)
        return subring, positives

    @staticmethod
    def sample_by_closest_feature(sample_size, same_edge_same_cat, nb_features, pos_nb_feature, edge_type):
        # if the number of negatives is smaller than the sample size, take them all.
        neg_sample_edge = []
        if len(same_edge_same_cat) <= sample_size:
            for node in same_edge_same_cat:
                neg_sample_edge.append([edge_type, node])
        else:
            # Find the difference between features of the positive neighbour pos_nb and nodes of the same category as
            # pos_nb.
            feature_diffs = []
            for nb, f in nb_features:
                if nb in same_edge_same_cat:
                    feature_diffs.append((nb, np.abs(f - pos_nb_feature)))

            # sort the distance diffs from smallest to largest
            sorted_feature_diffs = sorted(feature_diffs, key=lambda x: x[1])
            neg_sample = sorted_feature_diffs[: sample_size]

            # package the negative sample as a list of edges
            for node, _ in neg_sample:
                neg_sample_edge.append([edge_type, node])

        return neg_sample_edge

    @staticmethod
    def extract_pos_nb_feature(ring_info, edge_type, pos_nb):
        # toss a coin to select the feature used for sorting the samples
        coin_prob = np.random.uniform(0, 1)
        if coin_prob > 0.5:
            feature = 'distance'
        else:
            feature = 'iou'
        pos_nb_feature = [f for n, f in ring_info[edge_type][feature] if n == pos_nb]

        return feature, pos_nb_feature[0]

    def neg_sample_same_edge_type(self, graph, ring_info, same_edge_nbs, edge_type, pos_nb, pos_nb_edge_type,
                                  sample_size, verbose=False, neg_type=None):
        feature, pos_nb_feature = self.extract_pos_nb_feature(ring_info, pos_nb_edge_type, pos_nb)
        neg_sample = self.sample_by_closest_feature(sample_size,
                                                    same_edge_nbs,
                                                    ring_info[edge_type][feature],
                                                    pos_nb_feature,
                                                    edge_type)
        if verbose:
            print('Sub-sample size: ', len(neg_sample))
            print('This is negative type: ', neg_type)
            print('Edge Type is: ', edge_type)
            print('Other edge type and neighbours: ', same_edge_nbs)
            print('Negative Samples: ', neg_sample, 1)
            print(graph[pos_nb]['category'][0], [graph[n]['category'][0] for _, n in neg_sample])
            print('*' * 50)

        return neg_sample

    def draw_negatives(self, graph, source_node, subring, subring_nbs, num_neighbours):
        # determine how many negatives you take for each positive edge
        # sample size is the minimum of remaining nodes and half of the neighbours.
        sample_size = int(np.minimum(self.negative_ratio * num_neighbours, num_neighbours - len(subring_nbs)))

        # the tensor that represents the negative features for each positive edge
        negatives = torch.zeros((len(subring), sample_size, self.compute_feature_size(graph, subring[0][1])))

        # select negatives for each positive edge starting from hard to soft negatives.
        for idx, (edge_type, pos_nb) in enumerate(subring):
            neg_samples = []
            ring_info = graph[source_node]['ring_info']
            same_edge_nbs = ring_info[edge_type]['cat']
            same_edge_same_cat, same_edge_other_cats = [], []

            # categorize the nodes with same edge type as the positive neighbour p into: 1) nodes with same category as
            # p, 2) nodes with category different than p.
            if len(same_edge_nbs) > 0:
                for nb, cat in same_edge_nbs:
                    # make sure the nb is not a positive
                    if nb not in subring_nbs:
                        if cat == graph[pos_nb]['category'][0]:
                            same_edge_same_cat.append(nb)
                        else:
                            same_edge_other_cats.append(nb)

            # categorize the nodes with different edge type as the current positive neighbour p into : 1) neighbours
            # with the same category as p, 2) neighbours with category different than p.
            other_edge_same_cat, other_edge_other_cats = {}, {}
            other_edges = [e for e in ring_info.keys() if e != edge_type]
            for other_edge in other_edges:
                edge_nbs = ring_info[other_edge]['cat']
                for nb, cat in edge_nbs:
                    if nb not in subring_nbs:
                        if cat == graph[pos_nb]['category'][0]:
                            if other_edge in other_edge_same_cat:
                                other_edge_same_cat[other_edge].append(nb)
                            else:
                                other_edge_same_cat[other_edge] = [nb]
                        else:
                            if other_edge in other_edge_other_cats:
                                other_edge_other_cats[other_edge].append(nb)
                            else:
                                other_edge_other_cats[other_edge] = [nb]

            # samples go from hard to soft negatives in the following order:
            # 1) sample objects of the same category and edge type.
            if len(same_edge_same_cat) > 0:
                neg_sub_sample = self.neg_sample_same_edge_type(graph, ring_info, same_edge_same_cat, edge_type, pos_nb,
                                                                edge_type, sample_size, verbose=False, neg_type=1)
                # append the negatives and update the sample size
                neg_samples += neg_sub_sample
                sample_size -= len(neg_sub_sample)

            # 2) sample objects of the same category but different edge type.
            if (sample_size > 0) and (len(other_edge_same_cat) > 0):
                for i, (other_edge, other_edge_nbs) in enumerate(other_edge_same_cat.items()):
                    # divide the sample size over number of other edge types that you will be exploring.
                    num_remaining_edges = len(other_edge_same_cat) - i
                    if sample_size > num_remaining_edges:
                        sample_size_per_edge = sample_size // num_remaining_edges
                    else:
                        sample_size_per_edge = sample_size
                    if sample_size_per_edge > 0:
                        neg_sub_sample = self.neg_sample_same_edge_type(graph, ring_info, other_edge_nbs, other_edge,
                                                                        pos_nb, edge_type, sample_size_per_edge,
                                                                        verbose=False, neg_type=2)

                        # append the negatives and update the sample size
                        neg_samples += neg_sub_sample
                        sample_size -= len(neg_sub_sample)

            # 3) sample objects of different category but same edge type.
            if (sample_size > 0) and (len(same_edge_other_cats) > 0):
                neg_sub_sample = self.neg_sample_same_edge_type(graph, ring_info, same_edge_other_cats, edge_type,
                                                                pos_nb, edge_type, sample_size, verbose=False,
                                                                neg_type=3)

                # append the negatives and update the sample size
                neg_samples += neg_sub_sample
                sample_size -= len(neg_sub_sample)

            # 4) sample objects of different category and different edge type.
            if (sample_size > 0) and (len(other_edge_other_cats)) > 0:
                for i, (other_edge, other_edge_nbs) in enumerate(other_edge_other_cats.items()):
                    # divide the sample size over number of other edge types that you will be exploring.
                    num_remaining_edges = len(other_edge_other_cats) - i
                    if sample_size > num_remaining_edges:
                        sample_size_per_edge = sample_size // num_remaining_edges
                    else:
                        sample_size_per_edge = sample_size
                    if sample_size_per_edge > 0:
                        neg_sub_sample = self.neg_sample_same_edge_type(graph, ring_info, other_edge_nbs, other_edge,
                                                                        pos_nb, edge_type, sample_size_per_edge,
                                                                        verbose=False, neg_type=4)

                        # append the negatives and update the sample size
                        neg_samples += neg_sub_sample
                        sample_size -= len(neg_sub_sample)

            # extract features for each negative example.
            for i, (e, nb) in enumerate(neg_samples):
                feature = self.extract_edge_feature(graph, source_node, nb, e)
                negatives[idx, i, :] = feature

        return negatives

    def extract_all_edge_features(self, graph, nodes):
        # treat each node a source node and derive the edge features for that
        features = torch.zeros(len(nodes), len(nodes) - 1, self.compute_feature_size(graph, nodes[0]))

        # handle the case of graph with one node first
        if len(nodes) == 1:
            source_feature = graph[nodes[0]]['cat_vec']
            nb_feature = [0] * len(source_feature) + [0] + [0]
            features = torch.zeros(1, 1, self.compute_feature_size(graph, nodes[0]))
            features[0, 0, :] = torch.from_numpy(np.asarray(source_feature + nb_feature))
        else:
            for i, source_node in enumerate(nodes):
                j = 0
                for nb in nodes:
                    if nb != source_node:
                        edge_type = graph[source_node]['neighbours'][nb][0]
                        features[i, j, :] = self.extract_edge_feature(graph, source_node, nb, edge_type)
                        j += 1

        return features

    def __getitem__(self, idx):
        # load the graph and its nodes
        graph_path = os.path.join(self.data_dir, self.mode, self.file_names[idx])
        graph = load_from_json(graph_path)
        source_nodes = [n for n in graph.keys() if n.isdigit()]

        # for training draw random positives and negatives
        if self.mode == 'train':
            # choose a random source node and its associate ring
            source_node = np.random.choice(source_nodes, 1)[0]

            # select a subring (positive edges)
            num_neighbours = len(source_nodes) - 1
            subring, positives = self.draw_positive(graph, source_node, num_neighbours)

            # make sure the ring offers a subring to begin with:
            # for each positive edge pick a number of negative examples
            subring_nbs = set([node for _, node in subring])
            negatives = self.draw_negatives(graph, source_node, subring, subring_nbs, num_neighbours)
            negatives = negatives.reshape(-1, negatives.shape[-1])

            # concat the positive and negative features
            positive_negatives = torch.cat([positives, negatives], dim=0)
            data = {'file_name': self.file_names[idx], 'positive_negatives': positive_negatives,
                    'num_positives': len(positives)}
        else:
            # for applying the model derive node features for all nodes
            features = self.extract_all_edge_features(graph, source_nodes)
            # TODO: Add the nodes ids   to get a sense of the order
            data = {'file_name': self.file_names[idx], 'features': features}

        return data

