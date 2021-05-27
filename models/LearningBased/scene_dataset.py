import os
import torch
from torch.utils.data import Dataset
import numpy as np

from scripts.helper import load_from_json
from scripts.box import Box
from scripts.iou import IoU


class Scene(Dataset):
    def __init__(self, data_dir, mode, query_scene_name=None, q_context_objects=None, query_node=None,
                 with_cat_predictions=False, min_num_nbs=5, max_num_nbs=10, q_theta=0):
        self.data_dir = data_dir
        self.mode = mode
        self.query_scene_name = query_scene_name
        self.with_cat_predictions = with_cat_predictions

        if self.query_scene_name is not None:
            self.query_graph = load_from_json(os.path.join(self.data_dir, self.mode, query_scene_name))
            self.q_context_objects = q_context_objects
            self.query_node = query_node
            self.with_cat_predictions = with_cat_predictions
            self.file_names = self.filter_by_query_node(os.listdir(os.path.join(self.data_dir, self.mode)),
                                                        query_cat=self.query_graph[query_node]['category'][0])

            # if q_theta is greater than 0 rotate it
            if q_theta > 0:
                self.rotate_query(q_theta)
        else:
            self.file_names = os.listdir(os.path.join(self.data_dir, self.mode))

        self.min_num_nbs = min_num_nbs
        self.max_num_nbs = max_num_nbs
        self.feature_names = ['volumes', 'distances', 'cosines', 'sines', 'IoUs', 'matches']

    def __len__(self):
        return len(self.file_names)

    def filter_by_query_node(self, file_names, query_cat=None):
        # allowing methods relying on oracle or predicted categories.
        filtered_file_names = []
        for file_name in file_names:
            graph = load_from_json(os.path.join(self.data_dir, self.mode, file_name))
            for _, node_info in graph.items():
                if self.with_cat_predictions:
                    if query_cat == node_info['category_predicted'][0]:
                        filtered_file_names.append(file_name)
                        break
                else:
                    if query_cat == node_info['category'][0]:
                        filtered_file_names.append(file_name)
                        break

        return filtered_file_names

    def rotate_query(self, theta):
        # create a box for each query and context object. translate the query subscene to the origin.
        obj_to_obbox = {}
        vertices = np.asarray(self.query_graph[self.query_node]['obbox'])
        obj_to_obbox[self.query_node] = Box(vertices)
        q_translation = -obj_to_obbox[self.query_node].translation
        obj_to_obbox[self.query_node] = self.translate_obbox(obj_to_obbox[self.query_node], q_translation)

        for c_obj in self.q_context_objects:
            vertices = np.asarray(self.query_graph[c_obj]['obbox'])
            obj_to_obbox[c_obj] = Box(vertices)
            obj_to_obbox[c_obj] = self.translate_obbox(obj_to_obbox[c_obj], q_translation)

        # rotate the query subscene
        transformation = np.eye(4)
        rotation = np.asarray([[np.cos(theta), -np.sin(theta), 0],
                               [np.sin(theta), np.cos(theta), 0],
                               [0, 0, 1]])
        transformation[:3, :3] = rotation
        for obj in obj_to_obbox.keys():
            obj_to_obbox[obj] = obj_to_obbox[obj].apply_transformation(transformation)

            # translate the query subscne back to where it was
            obj_to_obbox[obj] = self.translate_obbox(obj_to_obbox[obj], -q_translation)

            # record the obboxes for the rotated subscene.
            self.query_graph[obj]['obbox'] = obj_to_obbox[obj].vertices.tolist()

    @staticmethod
    def sample_context_objects(graph, source_node, num_neighbours):
        # ring consists of the id of the neighbour nodes
        ring = list(graph.keys())
        ring.remove(source_node)

        # randomly choose a subring.
        rand_idx = np.random.choice(len(ring), num_neighbours, replace=False)
        query_subring = np.asarray(ring)[rand_idx]

        return query_subring.tolist()

    def map_cat_to_objects(self, query_cats, graph, source_node):
        cat_to_objects = {cat: [] for cat in query_cats}
        for node, node_info in graph.items():
            if node != source_node:
                if self.with_cat_predictions:
                    target_cat = node_info['category_predicted'][0]
                else:
                    target_cat = node_info['category'][0]
                if target_cat in cat_to_objects:
                    cat_to_objects[target_cat].append(node)

        return cat_to_objects

    @staticmethod
    def compute_angle(source_obbox, nb_obbox, source_nb_vec=None):
        if source_nb_vec is None:
            # compute the vector connecting the centroids of the nb to the source node.
            source_nb_vec = nb_obbox.translation - source_obbox.translation
        source_nb_vec_xy = source_nb_vec[:-1]

        x_unit = np.asarray([1, 0])
        cos_angle = np.dot(source_nb_vec_xy, x_unit) / (np.linalg.norm(source_nb_vec_xy))
        angle = np.arccos(cos_angle)
        if source_nb_vec_xy[1] < 0:
            angle = 2 * np.pi - angle

        return angle

    @staticmethod
    def translate_obbox(obbox, translation):
        # build the transformation matrix
        transformation = np.eye(4)
        transformation[:3, 3] = translation

        # apply tranlsation to the obbox
        obbox = obbox.apply_transformation(transformation)

        return obbox

    def build_bipartite_graph(self, query_graph, target_graph, context_objects, query_node, target_node,
                              cat_to_objects, clean_objects=[]):
        bp_graph = {}
        # build obbox for the query object and translate it to the origin
        query_obbox_vertices = np.asarray(query_graph[query_node]['obbox'])
        q_obbox = Box(query_obbox_vertices)
        q_translation = -q_obbox.translation
        q_obbox = self.translate_obbox(q_obbox, q_translation)

        # build obbox for the target object and find its translation to the origin
        target_obbox_vertices = np.asarray(target_graph[target_node]['obbox'])
        t_obbox = Box(target_obbox_vertices)
        t_translation = -t_obbox.translation
        t_obbox = self.translate_obbox(t_obbox, t_translation)

        # for each context object populate features for its neighbours of the same category
        for context_obj in context_objects:
            # add the neighbours
            context_obj_cat = query_graph[context_obj]['category'][0]
            bp_graph[context_obj] = {'candidates': cat_to_objects[context_obj_cat]}

            # translate the obbox of the context object using the vector that translates the query object to the origin.
            q_context_obbox_vertices = np.asarray(query_graph[context_obj]['obbox'])
            q_c_obbox = Box(q_context_obbox_vertices)
            q_c_obbox = self.translate_obbox(q_c_obbox, q_translation)

            # compute the angle for the context object centered around the query object.
            context_obj_angle = self.compute_angle(q_obbox, q_c_obbox)

            # populate features for each corresponding candidate
            bp_graph[context_obj]['volumes'] = []
            bp_graph[context_obj]['distances'] = []
            bp_graph[context_obj]['cosines'] = []
            bp_graph[context_obj]['sines'] = []
            bp_graph[context_obj]['IoUs'] = []
            bp_graph[context_obj]['matches'] = []
            for candidate in bp_graph[context_obj]['candidates']:
                if candidate in clean_objects and candidate == context_obj:
                    bp_graph[context_obj]['matches'] += [1.0]
                else:
                    bp_graph[context_obj]['matches'] += [0.0]
                # translate the obbox of the candidate object using a vector that translates the target object to the
                # origin
                t_candidate_obbox_vertices = np.asarray(target_graph[candidate]['obbox'])
                t_c_obbox = Box(t_candidate_obbox_vertices)
                t_c_obbox = self.translate_obbox(t_c_obbox, t_translation)

                # add the volume of each candidate
                bp_graph[context_obj]['volumes'] += [t_c_obbox.volume]

                # add the distances
                distance = np.linalg.norm(t_c_obbox.translation - t_obbox.translation)
                bp_graph[context_obj]['distances'] += [distance]

                # add the cosines and sines for the delta angle between the context object and the candidate
                candidate_obj_angle = self.compute_angle(t_obbox, t_c_obbox)
                delta_angle = context_obj_angle - candidate_obj_angle
                bp_graph[context_obj]['cosines'] += [np.cos(delta_angle)]
                bp_graph[context_obj]['sines'] += [np.sin(delta_angle)]

                # create a new target obbox that you will rotate
                target_obbox_vertices = np.asarray(target_graph[target_node]['obbox'])
                t_obbox = Box(target_obbox_vertices)
                t_obbox = self.translate_obbox(t_obbox, -t_obbox.translation)

                # rotate the pair (candidate, target) based on the angle between that pair and (context, query)
                transformation = np.eye(4)
                rotation = np.asarray([[np.cos(delta_angle), -np.sin(delta_angle), 0],
                                       [np.sin(delta_angle), np.cos(delta_angle), 0],
                                       [0, 0, 1]])
                transformation[:3, :3] = rotation
                t_obbox = t_obbox.apply_transformation(transformation)
                t_c_obbox = t_c_obbox.apply_transformation(transformation)

                # compute the iou between the pair (candidate, target) and (context, query)
                iou1 = IoU(t_obbox, q_obbox).iou()
                iou2 = IoU(t_c_obbox, q_c_obbox).iou()
                bp_graph[context_obj]['IoUs'] += [iou1 + iou2]

        return bp_graph

    @staticmethod
    def combine_features(bp_graph, feature_names):
        features = []
        for q_context, q_context_info in bp_graph.items():
            curr_candidate_features = []
            for feature_name in feature_names:
                feature = torch.from_numpy(np.asarray(q_context_info[feature_name]))
                feature.unsqueeze_(dim=1)
                curr_candidate_features.append(feature)

            # combine the features for the candidates of the current context object
            curr_candidate_features = torch.cat(curr_candidate_features, dim=1)
            features.append(curr_candidate_features)

        # combine features across the candidates of all context objects
        features = torch.cat(features, dim=0)

        return features

    def perturb_centroid(self, obbox, direction, direction_fraction=0.2, angle_epsilon=5):
        # find the min and max scale of the obbox
        scale = obbox.scale
        min_scale = np.min(scale)

        # perturb the distance of the centroid along the positive or negative direction by using a fraction of the min
        # scale.
        if np.random.uniform(0, 1) > 0.5:
            direction = -direction
        perturbation = np.random.uniform(0, min_scale * direction_fraction)

        # find the new centroid
        new_centroid = obbox.translation + perturbation * direction

        # perturb the angle of the centroid by a small amount
        angle_epsilon = np.random.uniform(0, angle_epsilon)
        if np.random.uniform(0, 1) > 0.5:
            angle_epsilon = -angle_epsilon
        angle_epsilon = angle_epsilon * np.pi/180
        rotation = np.asarray([[np.cos(angle_epsilon), -np.sin(angle_epsilon), 0],
                               [np.sin(angle_epsilon), np.cos(angle_epsilon), 0],
                               [0, 0, 1]])
        new_centroid = np.dot(rotation, new_centroid)

        # translate the centroid to its new location
        obbox = self.translate_obbox(obbox, new_centroid - obbox.translation)

        return obbox

    @staticmethod
    def find_different_angle(curr_angles):
        # add pi/4 and 7pi/4 to the list of angles
        angle_list = curr_angles.copy()
        if np.pi/4 not in angle_list:
            angle_list.append(np.pi/4)
        if 7*np.pi/4 not in angle_list:
            angle_list.append(7*np.pi/4)

        # sort the angles
        sorted_angles = sorted(angle_list)

        # find the largest segment
        largest_segment = []
        largest_length = 0
        for i in range(1, len(sorted_angles)):
            curr_length = sorted_angles[i] - sorted_angles[i-1]
            if curr_length > largest_length:
                largest_segment = [sorted_angles[i-1], sorted_angles[i]]
                largest_length = curr_length

        # sample an angle approximately in the middle of the largest segment
        middle_angle = (largest_segment[0] + largest_segment[1]) / 2
        if np.random.uniform(0, 1) > 0.5:
            new_angle = middle_angle + largest_length / 8
        else:
            new_angle = middle_angle - largest_length / 8

        return new_angle

    def build_query(self, target_graph, query_node, context_objects, rotation_angle):
        query_and_context = [query_node] + context_objects
        query_graph = {n: target_graph[n].copy() for n in query_and_context}

        # pick the majority of the context objects to corrupt
        if len(context_objects) == 3:
            clean_objects = context_objects[:2]
        elif len(context_objects) > 3:
            # pick less than half of the context objects as corresponding objects
            num_clean = int(np.floor(len(context_objects) / 2))
            # make sure to pick at least two clean objects
            num_clean = np.random.randint(2, num_clean+1)
            clean_objects = context_objects[:num_clean]
        else:
            clean_objects = context_objects

        # translate the query obbox to the global origin and save it
        obj_to_obbox = {}
        vertices = np.asarray(query_graph[query_node]['obbox'])
        obj_to_obbox[query_node] = Box(vertices)
        q_translation = -obj_to_obbox[query_node].translation
        obj_to_obbox[query_node] = self.translate_obbox(obj_to_obbox[query_node], q_translation)

        # create obbox for context objects
        obj_to_old_obbox = {}
        for obj in context_objects:
            vertices = np.asarray(query_graph[obj]['obbox'])
            obj_to_obbox[obj] = Box(vertices)
            # translate the obbox according to the query translation.
            obj_to_obbox[obj] = self.translate_obbox(obj_to_obbox[obj], q_translation)

            # perturb the volume of all obboxes
            old_obbox_vertices = obj_to_obbox[obj].vertices.copy()
            old_obbox = Box(old_obbox_vertices)
            obj_to_old_obbox[obj] = old_obbox

            # perturb the centroid of the context object along the vector that connects it to the query object.
            direction = obj_to_obbox[obj].translation - obj_to_obbox[query_node].translation
            direction = direction / np.linalg.norm(direction)
            obj_to_obbox[obj] = self.perturb_centroid(obj_to_obbox[obj], direction)

        # rotate the corrupt objects by an angle different from the rotation_angle. rotate maximum n-1 of the corrupt
        # objects by the exact same angle where n is the number of clean objects
        default_rotation = np.random.uniform(np.pi / 4, 7 * np.pi / 4)
        num_clean = np.maximum(len(clean_objects), 2)
        num_same_angle = np.random.randint(1, num_clean)
        same_angle_count = 0
        corrupt_angles = [default_rotation]
        for obj in context_objects:
            if obj not in clean_objects:
                if same_angle_count < num_same_angle:
                    rotation_angle_corrupt = default_rotation
                    same_angle_count += 1
                else:
                    rotation_angle_corrupt = self.find_different_angle(corrupt_angles)
                    corrupt_angles.append(rotation_angle_corrupt)
                transformation = np.eye(4)
                rotation = np.asarray([[np.cos(rotation_angle_corrupt), -np.sin(rotation_angle_corrupt), 0],
                                       [np.sin(rotation_angle_corrupt), np.cos(rotation_angle_corrupt), 0],
                                       [0, 0, 1]])
                transformation[:3, :3] = rotation
                obj_to_obbox[obj] = obj_to_obbox[obj].apply_transformation(transformation)

        # rotate all the perturbed and corrupted context objects by the rotation_angle
        transformation = np.eye(4, dtype=np.float)
        rotation = np.asarray([[np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                               [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                               [0, 0, 1]])
        transformation[:3, :3] = rotation
        for obj in query_and_context:
            obj_to_obbox[obj] = obj_to_obbox[obj].apply_transformation(transformation)

        # update the obboxes in the query graph
        for node, node_info in query_graph.items():
            node_info['obbox'] = obj_to_obbox[node].vertices

        return query_graph, clean_objects

    def __getitem__(self, idx):
        # load the graph
        graph_path = os.path.join(self.data_dir, self.mode, self.file_names[idx])
        target_graph = load_from_json(graph_path)
        objects = list(target_graph.keys())

        data = {'file_name': self.file_names[idx]}
        # if a query ring is not given draw random positives by randomly selecting a subring.
        if self.query_scene_name is None:
            # choose a random query object
            query_node = np.random.choice(objects, 1)[0]

            # choose a random set of context objects for the query.
            num_neighbours = np.random.randint(self.min_num_nbs, self.max_num_nbs+1)
            num_neighbours = np.minimum(len(objects) - 1, num_neighbours)
            q_context_objects = self.sample_context_objects(target_graph, query_node, num_neighbours)

            # find the categories of the context objects in the query subring.
            query_cats = [target_graph[c]['category'][0] for c in q_context_objects]
            # map the categories found to the nodes in the entire scene
            cat_to_objects = self.map_cat_to_objects(query_cats, target_graph, query_node)

            # perturb the context objects in the query and rotate them by a random angle.
            data['theta'] = np.random.uniform(0, 2*np.pi)
            query_graph, clean_objects = self.build_query(target_graph, query_node, q_context_objects, data['theta'])

            # build a bipartite graph connecting each context object in the query to a candidate in the target
            # graph.
            bp_graph = self.build_bipartite_graph(query_graph, target_graph, q_context_objects, query_node,
                                                  query_node, cat_to_objects, clean_objects)
            # combine the features recorded in the bp_graph into a tensor
            features = self.combine_features(bp_graph, self.feature_names)
            features.unsqueeze_(dim=0)
        else:
            # determine the potential target nodes; i.e nodes with the same category as query node
            query_cat = self.query_graph[self.query_node]['category'][0]

            # find all target nodes with the same category as the query object.
            if self.with_cat_predictions:
                target_nodes = [n for n in target_graph.keys() if target_graph[n]['category_predicted'][0] == query_cat]
            else:
                target_nodes = [n for n in target_graph.keys() if target_graph[n]['category'][0] == query_cat]
            data['target_nodes'] = target_nodes

            # find the categories of the context objects in the query subring.
            query_cats = [self.query_graph[c]['category'][0] for c in self.q_context_objects]

            # build the features for each target node
            features = []
            data['bp_graphs'] = {}
            for target_node in target_nodes:
                # map each category to the objects in the scene with that category.
                cat_to_objects = self.map_cat_to_objects(query_cats, target_graph, target_node)

                # build a bipartite graph connecting each context object in the query to a candidate in the target
                # graph.
                bp_graph = self.build_bipartite_graph(self.query_graph, target_graph, self.q_context_objects,
                                                      self.query_node, target_node, cat_to_objects)

                # record the context candidates for each target node
                data['bp_graphs'][target_node] = {}
                for context_object in bp_graph.keys():
                    data['bp_graphs'][target_node][context_object] = bp_graph[context_object]['candidates']

                # combine the features recorded in the bp_graph into a tensor
                target_features = self.combine_features(bp_graph, self.feature_names)
                features.append(target_features)

        data['features'] = features

        return data

