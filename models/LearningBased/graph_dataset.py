import os
import torch
from torch.utils.data import Dataset
import numpy as np

from scripts.helper import load_from_json
from scripts.box import Box


class Scene(Dataset):
    def __init__(self, scene_graph_dir, latent_caps_dir, cat_to_idx, mode):
        self.scene_graph_dir = scene_graph_dir
        self.latent_caps_dir = latent_caps_dir
        self.cat_to_idx = cat_to_idx
        self.mode = mode

        self.file_names = os.listdir(os.path.join(self.scene_graph_dir, self.mode))
        self.relation_dic = {'supports': 0, 'supported': 1, 'encloses': 2, 'enclosed': 3, 'contact': 4, 'fc': 5}

    def __len__(self):
        return len(self.file_names)

    def prepare_latent_caps(self, scene_name, obj):
        file_name = scene_name + '-' + obj + '.npy'
        latent_caps = np.load(os.path.join(self.latent_caps_dir, file_name))
        latent_caps = latent_caps.reshape(-1)

        return latent_caps

    @staticmethod
    def prepare_volumes(graph, obj):
        obbox_vertices = np.asarray(graph[obj]['obbox'])
        obbox = Box(obbox_vertices)

        return obbox.volume

    def prepare_labels(self, graph, obj):
        cat = graph[obj]['category'][0]

        return self.cat_to_idx[cat]

    def build_adj_matrix(self, graph, objects, obj_to_index):
        # initialize the adj matrix
        N = len(objects)
        adj = torch.zeros((len(self.relation_dic), N, N), dtype=torch.float32)

        # populate the adj matrix
        for obj in objects:
            for nb, relations in graph[obj]['neighbours'].items():
                # populate source-to-nb
                for relation in relations:
                    idx = self.relation_dic[relation]
                    adj[idx, obj_to_index[obj], obj_to_index[nb]] = 1
                # populate nb-to-source
                reverse_relations = graph[nb]['neighbours'][obj]
                for relation in reverse_relations:
                    idx = self.relation_dic[relation]
                    adj[idx, obj_to_index[nb], obj_to_index[obj]] = 1

        # test the adj is symmetric for contact and fc relations.
        assert torch.all(adj[4, :, :] == adj[4, :, :].t())
        assert torch.all(adj[5, :, :] == adj[5, :, :].t())

        # test if the graph is fully connected.
        assert torch.all(torch.ones(N, N) - adj[-1, ...] - torch.sum(adj[:-1, ...], dim=0) == torch.eye(N, N))

        return adj

    @staticmethod
    def normalize_adj_mp(adj, num_nodes):
        self_loop = torch.eye(num_nodes, num_nodes, dtype=torch.float32)
        degree = torch.zeros_like(adj)
        for j in range(adj.shape[0]):
            # check the adj has no self loops to begin with
            assert adj[j, :, :].diagonal().sum() == 0

            adj[j, :, :] += self_loop
            row_sum = torch.sum(adj[j, :, :], dim=1)
            d_inv = torch.pow(row_sum, -1)
            d_inv[torch.isinf(d_inv)] = 0.
            ind = np.diag_indices(num_nodes)
            degree[j, ind[0], ind[1]] = d_inv
            adj[j, :, :] = torch.mm(degree[j, :, :], adj[j, :, :])
        return adj

    def __getitem__(self, idx):
        # load the graph
        graph_path = os.path.join(self.scene_graph_dir, self.mode, self.file_names[idx])
        graph = load_from_json(graph_path)
        scene_name = self.file_names[idx].split('.')[0]
        objects = list(graph.keys())

        # sort the objects consistently
        objects = sorted(objects, key=int)
        obj_to_index = {obj: idx for idx, obj in enumerate(objects)}

        data = {'file_name': self.file_names[idx]}
        # prepare features and labels.
        graph_latent_caps = []
        volumes = []
        labels = []
        for i, obj in enumerate(objects):
            # read and prepare latent caps.
            graph_latent_caps.append(self.prepare_latent_caps(scene_name, obj))
            volumes.append(self.prepare_volumes(graph, obj))
            labels.append(self.prepare_labels(graph, obj))

        # combine features and lablels into a tensor
        graph_latent_caps = torch.from_numpy(np.asarray(graph_latent_caps))
        volumes = torch.from_numpy(np.asarray(volumes)).unsqueeze(dim=1)
        labels = torch.from_numpy(np.asarray(labels))

        data['features'] = torch.cat([graph_latent_caps, volumes], dim=1)
        data['labels'] = labels

        # build and normalize the adj matrix
        adj = self.build_adj_matrix(graph, objects, obj_to_index)
        data['adj'] = self.normalize_adj_mp(adj, len(objects))

        return data

