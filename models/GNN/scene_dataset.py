import os
import torch
from torch.utils.data import Dataset
import numpy as np
import heapq

from scripts.helper import load_from_json
from scripts.box import Box
from scripts.iou import IoU


class Scene(Dataset):
    def __init__(self, data_dir, mode, query_scene_name=None, q_context_objects=None, query_node=None, min_num_nbs=5,
                 max_num_nbs=10):
        self.data_dir = data_dir
        self.mode = mode
        self.query_scene_name = query_scene_name

        if self.query_scene_name is not None:
            self.query_graph = load_from_json(os.path.join(self.data_dir, self.mode, query_scene_name))
            self.q_context_objects = q_context_objects
            self.query_node = query_node
            self.file_names = self.filter_by_query_cat(os.listdir(os.path.join(self.data_dir, self.mode)),
                                                       self.query_graph[query_node]['category'][0])
        else:
            self.file_names = os.listdir(os.path.join(self.data_dir, self.mode))

        self.min_num_nbs = min_num_nbs
        self.max_num_nbs = max_num_nbs

    def __len__(self):
        return len(self.file_names)

    def filter_by_query_cat(self, file_names, query_cat):
        filtered_file_names = []
        for file_name in file_names:
            graph = load_from_json(os.path.join(self.data_dir, self.mode, file_name))
            for _, node_info in graph.items():
                if node_info['category'][0] == query_cat:
                    filtered_file_names.append(file_name)
                    break
        return filtered_file_names

    def sample_context_objects(self, graph, source_node, num_neighbours):
        # ring consists of the id nof the neighbour nodes
        ring = list(graph.keys())
        ring.remove(source_node)

        # randomly choose a subring.
        rand_idx = np.random.choice(len(ring), num_neighbours, replace=False)
        query_subring = np.asarray(ring)[rand_idx]

        # print('Ring size: ', len(ring))
        # print('Subring size: ', len(query_subring))
        # print('*' * 50)

        return query_subring

    @staticmethod
    def map_cats_to_obj(cats, graph, source_node):
        cat_to_objects = {cat: [] for cat in cats}
        for node, node_info in graph.items():
            if node != source_node:
                cat = node_info['category'][0]
                if cat in cats:
                    cat_to_objects[cat].append(node)

        return cat_to_objects

    @staticmethod
    def compute_angle(graph, source_node, nb):
        # compute the vector connecting the centroids of the nb to the source node.
        source_nb_vec = np.asarray(graph[nb]['obbox'][0]) - np.asarray(graph[source_node]['obbox'][0])
        source_nb_vec_xy = source_nb_vec[:-1]

        x_unit = np.asarray([1, 0])
        cos_angle = np.dot(source_nb_vec_xy, x_unit) / (np.linalg.norm(source_nb_vec_xy) + 0.00001)
        angle = np.arccos(cos_angle)
        if source_nb_vec_xy[1] < 0:
            angle = 2 * np.pi - angle

        return angle

    def build_bipartite_graph(self, query_graph, target_graph, query_subring, query_node, target_node, cat_to_objects):
        bp_graph = {}
        # find the translation from the query and target nodes to origin
        q_to_origin = -np.asarray(query_graph[query_node]['obbox'][0])
        t_to_origin = -np.asarray(target_graph[target_node]['obbox'][0])

        # for each context object find its neighbours and translate their obbox
        for context_obj in query_subring:
            # add the neighbours
            context_obj_cat = query_graph[context_obj]['category'][0]
            bp_graph[context_obj] = {'neighbours': cat_to_objects[context_obj_cat]}

            # translate the obbox for the context object and add the angle
            bp_graph[context_obj]['obbox'] = np.asarray(query_graph[context_obj]['obbox']) + q_to_origin
            context_obj_angle = self.compute_angle(query_graph, query_node, context_obj)
            bp_graph[context_obj]['angle'] = context_obj_angle

            # translate the neighbours obboxes and add their angles
            bp_graph[context_obj]['obboxes'] = []
            bp_graph[context_obj]['angles'] = []
            for nb in bp_graph[context_obj]['neighbours']:
                bp_graph[context_obj]['obboxes'] += [np.asarray(target_graph[nb]['obbox']) + t_to_origin]
                bp_graph[context_obj]['angles'] += [self.compute_angle(target_graph, target_node, nb)]

        return bp_graph

    def update_matches(self, iou_nodes, curr_q_context, bp_graph, visited):
        match = (-1, 0)
        while len(iou_nodes) > 0:
            max_iou, t_context = heapq.heappop(iou_nodes)
            max_iou = -max_iou
            if t_context not in visited:
                match = (t_context, max_iou)
                visited[t_context] = (curr_q_context, max_iou)
                break
            else:
                prev_q_context, prev_iou = visited[t_context]
                if max_iou > prev_iou:
                    # the current match is better than previous.
                    visited[t_context] = (curr_q_context, max_iou)
                    match = (t_context, max_iou)
                    # find a replacement for the previous match
                    new_match = self.update_matches(bp_graph[prev_q_context]['IoUs'], prev_q_context, bp_graph, visited)
                    bp_graph[prev_q_context]['match'] = new_match
                    break

        return match

    def find_correspondence(self, bp_graph, query_obbox, target_obbox):
        q_obbox = Box(query_obbox)
        visited = {}
        for q_context, node_info in bp_graph.items():
            # read the angle and obbox for the context object in query scene
            q_context_angle = node_info['angle']
            q_c_obbox = Box(node_info['obbox'])

            # find the rotation angle from a t_context to the q_context
            node_info['IoUs'] = []
            for i, t_context in enumerate(node_info['neighbours']):
                # find the rotation angle and extract the obbox
                t_context_angle = node_info['angles'][i]
                delta_theta = q_context_angle - t_context_angle

                # create box objects for the target node and t_context
                t_obbox = Box(target_obbox.copy())
                t_c_obbox = Box(node_info['obboxes'][i].copy())

                # apply rotation to the pair of obboxes (t, t_c_context)
                transformation = np.eye(4)
                rotation = np.asarray([[np.cos(delta_theta), -np.sin(delta_theta), 0],
                                       [np.sin(delta_theta), np.cos(delta_theta), 0],
                                       [0, 0, 1]])
                transformation[:3, :3] = rotation

                # beofre rotation
                # if delta_theta > 0:
                #     import trimesh
                #     t = trimesh.creation.box(t_obbox.scale, transform=t_obbox.transformation)
                #     t.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#0000ff")
                #     q_c = trimesh.creation.box(q_c_obbox.scale, transform=q_c_obbox.transformation)
                #     q_c.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#FF0000")
                #     t_c = trimesh.creation.box(t_c_obbox.scale, transform=t_c_obbox.transformation)
                #     trimesh.Scene([t, q_c, t_c]).show()

                t_obbox = t_obbox.apply_transformation(transformation)
                t_c_obbox = t_c_obbox.apply_transformation(transformation)

                    # # after rotation
                    # t = trimesh.creation.box(t_obbox.scale, transform=t_obbox.transformation)
                    # t_c = trimesh.creation.box(t_c_obbox.scale, transform=t_c_obbox.transformation)
                    # trimesh.Scene([t, t_c]).show()

                # compute the iou between the pair (t, c_1) and (q, c_2)
                iou = IoU(t_obbox, q_obbox).iou() + IoU(t_c_obbox, q_c_obbox).iou()
                node_info['IoUs'].append((-iou, t_context))

            # update matches
            heapq.heapify(node_info['IoUs'])
            match = self.update_matches(node_info['IoUs'], q_context, bp_graph, visited)
            node_info['match'] = match

    def __getitem__(self, idx):
        # load the graph
        graph_path = os.path.join(self.data_dir, self.mode, self.file_names[idx])
        graph = load_from_json(graph_path)
        objects = list(graph.keys())

        data = {'file_name': self.file_names[idx]}
        # if a query ring is not given draw random positives by randomly selecting a subring.
        if self.query_scene_name is None:
            # choose a random source node and its associate ring
            query_node = np.random.choice(objects, 1)[0]

            # choose a query subring.
            num_neighbours = np.random.randint(self.min_num_nbs, self.max_num_nbs+1)
            num_neighbours = np.minimum(len(objects) - 1, num_neighbours)
            q_cotnext_objects = self.sample_context_objects(graph, query_node, num_neighbours)

            # find the categories of the context objects in the query subring.
            query_cats = [graph[c]['category'][0] for c in q_cotnext_objects]

            # map the categories found to the nodes in the entire scene
            cat_to_objects = self.map_cats_to_obj(query_cats, graph, query_node)

            # build a bipartite graph connecting each context object in the query to a candidate in the target graph.
            bp_graph = self.build_bipartite_graph(graph, graph, q_cotnext_objects, query_node, query_node,
                                                  cat_to_objects)

            # extract the query and target obboxes and translate them to the origin
            query_obbox = np.asarray(graph[query_node]['obbox'])
            query_obbox -= query_obbox[0]

            target_obbox = np.asarray(graph[query_node]['obbox'])
            target_obbox -= target_obbox[0]

            bp_graphs = [bp_graph]
            translated_target_obboxes = [target_obbox]
        else:
            # determine the potential target nodes; i.e nodes with the same category as query node
            query_cat = self.query_graph[self.query_node]['category'][0]
            target_nodes = [node for node in graph.keys() if graph[node]['category'][0] == query_cat]
            data['target_nodes'] = target_nodes

            # extract the query obbox and translate it to the origin
            query_obbox = np.asarray(self.query_graph[self.query_node]['obbox'])
            query_obbox -= query_obbox[0]

            # find the correspondences for each bipartite graph
            bp_graphs = []
            translated_target_obboxes = []
            for target_node in target_nodes:
                # find the categories of the context objects in the query subring.
                query_cats = [self.query_graph[c]['category'][0] for c in self.q_context_objects]

                # map the categories found to the nodes in the entire scene
                cat_to_objects = self.map_cats_to_obj(query_cats, graph, target_node)

                # build a bipartite graph connecting each context object in the query to a candidate in the target
                # graph.
                bp_graph = self.build_bipartite_graph(self.query_graph, graph, self.q_context_objects, self.query_node,
                                                      target_node, cat_to_objects)
                bp_graphs.append(bp_graph)

                # extract the target obbox and translate it to the origin
                target_obbox = np.asarray(graph[target_node]['obbox'])
                target_obbox -= target_obbox[0]
                translated_target_obboxes.append(target_obbox)

        # find the correspondence by comparing IoU of each pair (t, c_1) and (q, c_2). c_1 and c_2 are on each side
        # of the bipartite graph and connect to each other.
        for i in range(len(bp_graphs)):
            # find the correspondence by comparing IoU of each pair (t, c_1) and (q, c_2)
            self.find_correspondence(bp_graphs[i], query_obbox, translated_target_obboxes[i])
            filtered_bp = {}
            for q_context, q_context_info in bp_graphs[i].items():
                filtered_bp[q_context] = {'match': q_context_info['match']}
        data['bp_graphs'] = bp_graphs
        # print(filtered_bp)

        return data

