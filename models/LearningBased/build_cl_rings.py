import os
import sys
import numpy as np
from time import time
import trimesh

from scripts.helper import load_from_json, write_to_json, visualize_scene, create_train_val_test
from scripts.box import Box
from scripts.iou import IoU


class Scene:
    def __init__(self, scene_graph_dir, scene_name, edge_types, accepted_cats):
        self.graph = load_from_json(os.path.join(scene_graph_dir, scene_name))
        self.edge_types = edge_types
        self.accepted_cats = accepted_cats

    def find_cat(self, node):
        return self.graph[node]['category'][0]

    @staticmethod
    def compute_dist(x, y):
        return np.linalg.norm(x - y)

    def filter_by_accepted_cats(self):
        filtered_graph = {}
        for node, node_info in self.graph.items():
            if node_info['category'][0] in self.accepted_cats:
                filtered_graph[node] = node_info
        self.graph = filtered_graph

    @staticmethod
    def translate_obbox(obbox, translation):
        # build the transformation matrix
        transformation = np.eye(4)
        transformation[:3, 3] = translation

        # apply tranlsation to the obbox
        obbox = obbox.apply_transformation(transformation)

        return obbox

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

    def test_sort_features(self, source_node, nb):
        # load the source obbox and centroid
        source_obbox_vertices = np.asarray(self.graph[source_node]['obbox'])
        obbox_source = Box(source_obbox_vertices)
        source_translation = -obbox_source.translation
        obbox_source = self.translate_obbox(obbox_source, source_translation)

        # load the obbox and centroid of the neighbour
        nb_obbox_vertices = np.asarray(self.graph[nb]['obbox'])
        obbox_nb = Box(nb_obbox_vertices)
        obbox_nb = self.translate_obbox(obbox_nb, source_translation)

        transformation = np.eye(4)
        # compute translation
        # transformation[:3, 3] = obbox_source.translation - obbox_nb.translation

        # compute rotation
        theta1 = self.compute_angle(self.graph, source_node, nb)
        print(theta1)
        theta2 = np.pi * 3/ 2
        delta_theta = theta2 - theta1
        rotation = np.asarray([[np.cos(delta_theta), -np.sin(delta_theta), 0],
                               [np.sin(delta_theta), np.cos(delta_theta), 0],
                               [0, 0, 1]])
        transformation[:3, :3] = rotation
        # nb_obbox_vertices_rot = np.dot(rotation, nb_obbox_vertices.transpose())
        # nb_obbox_vertices_rot = nb_obbox_vertices_rot.transpose()
        # obbox_nb = Box(nb_obbox_vertices_rot)
        # t=y

        # apply transformation
        obbox_nb = obbox_nb.apply_transformation(transformation)

        # print(self.compute_dist(centroid_source, centroid_nb))
        # print(centroid_source, centroid_nb)
        # print(self.find_cat(source_node), self.find_cat(nb))
        iou_computer = IoU(obbox_source, obbox_nb)
        print('IoU: ', iou_computer.iou())
        print('Source: {}'.format(self.graph[source_node]['category'][0]))
        s = trimesh.creation.box(obbox_source.scale, transform=obbox_source.transformation)
        nb = trimesh.creation.box(obbox_nb.scale, transform=obbox_nb.transformation)
        trimesh.Scene([s, nb]).show()

    def sort_features(self):
        # treat each node as the source of a ring
        for source_node in self.graph.keys():
            # build the aabbox for the source node
            obbox_source_vertices = np.asarray(self.graph[source_node]['obbox'])
            obbox_source = Box(obbox_source_vertices)
            aabbox_source_vertices = obbox_source.scaled_axis_aligned_vertices(obbox_source.scale)
            aabbox_source = Box(aabbox_source_vertices)

            # collect the ring info for each neighbour of the source node.
            ring_info = {edge_type: {'cat': [], 'distance': [], 'iou': []} for edge_type in self.edge_types}
            for nb, relations in self.graph[source_node]['neighbours'].items():
                # find the aabbox of the neighbour object
                obbox_nb_vertices = np.asarray(self.graph[nb]['obbox'])
                obbox_nb = Box(obbox_nb_vertices)
                aabbox_nb_vertices = obbox_nb.scaled_axis_aligned_vertices(obbox_nb.scale)
                aabbox_nb = Box(aabbox_nb_vertices)

                # for each edge type of the neighbour collect info about the neighbour
                for relation in relations:
                    # extract the category of the neighbour
                    ring_info[relation]['cat'].append((nb, self.find_cat(nb)))

                    # compute the distance of the neighbour to the source node
                    ring_info[relation]['distance'].append((nb, self.compute_dist(obbox_source.translation,
                                                                                  obbox_nb.translation)))

                    # compute the iou of the obbox translated to the source node
                    iou = IoU(aabbox_source, aabbox_nb).iou()
                    ring_info[relation]['iou'].append((nb, iou))

            # sort the features for distance and iou from closest to furthest
            for edge_type in self.edge_types:
                # closest to furthest distnace
                ring_info[edge_type]['distance'] = sorted(ring_info[edge_type]['distance'], key=lambda x: x[1])
                # highest to lowes iou
                ring_info[edge_type]['iou'] = sorted(ring_info[edge_type]['iou'], reverse=True, key=lambda x: x[1])

            # save the ring info
            self.graph[source_node]['ring_info'] = ring_info


def process_scenes(scene_names, scene_graph_dir_input, models_dir, edge_types, accepted_cats):
    t = time()
    idx = 0
    for scene_name in scene_names:
        # for each scene build scene graphs
        idx += 1
        print('Processing scene {} ... '.format(scene_name))
        print('Iteration {}/{}'.format(idx, len(scene_names)))
        t2 = time()
        if scene_name in visited:
            continue

        # create an instance of a scene and filter it by accepted cats.
        scene_graph = Scene(scene_graph_dir_input, scene_name, edge_types, accepted_cats)
        scene_graph.filter_by_accepted_cats()
        # after filtering, only save graphs with at least two elements (source and neighbour).
        if len(scene_graph.graph) > 1:
            # sort the ring objects based on distance and obbox iou relative to the source node.
            scene_graph.sort_features()

            # visualize the scene and test sort features if necessary
            # objects = ['10', '21']
            # scene_graph.test_sort_features(objects[0], objects[1])
            # visualize_scene(scene_graph_dir_input, models_dir, scene_name, accepted_cats=accepted_cats,
            #                 objects=objects, with_backbone=True, as_obbox=False)

            # save the graph
            scene_graph_path = os.path.join(scene_graph_dir_output, scene_name)
            write_to_json(scene_graph.graph, scene_graph_path)

            duration_house = (time() - t2) / 60
            print('Scene {} took {} minutes to process'.format(scene_name, round(duration_house, 2)))
            print('-' * 50)

    duration_all = (time() - t) / 60
    print('Processing {} scenes took {} minutes'.format(idx, round(duration_all, 2)))


def main(num_chunks, chunk_idx):
    build_rings = False
    split_train_test_val = False

    # define paths and params
    edge_types = ['supported', 'supports', 'contact', 'enclosed', 'encloses', 'fc']
    scene_graph_dir_input = '../../results/matterport3d/LearningBased/scene_graphs/all'
    models_dir = '../../data/matterport3d/models'
    accepted_cats = set(load_from_json('../../data/matterport3d/accepted_cats.json'))

    if build_rings:
        scene_names = os.listdir(scene_graph_dir_input)
        # scene_names = ['1pXnuDYAj8r_room18.json']
        # scene_names = ['8194nk5LbLH_room3.json']
        chunk_size = int(np.ceil(len(scene_names) / num_chunks))
        process_scenes(scene_names[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size], scene_graph_dir_input,
                       models_dir, edge_types, accepted_cats)

    if split_train_test_val:
        data_dir = '../../data/matterport3d'
        scene_graph_dir = '../../results/matterport3d/LearningBased/scene_graphs_cl'
        train_path = os.path.join(data_dir, 'scenes_train.txt')
        val_path = os.path.join(data_dir, 'scenes_val.txt')
        test_path = os.path.join(data_dir, 'scenes_test.txt')
        create_train_val_test(scene_graph_dir, train_path, val_path, test_path)


if __name__ == '__main__':
    scene_graph_dir_output = '../../results/matterport3d/LearningBased/scene_graphs_cl/all'
    if not os.path.exists(scene_graph_dir_output):
        os.makedirs(scene_graph_dir_output)
        visited = set()
    else:
        visited = set(os.listdir(scene_graph_dir_output))
    if len(sys.argv) == 1:
        main(1, 0)
    else:
        # To run in parallel you can use the command:
        # export PYTHONPATH="${PYTHONPATH}:/home/reza/Documents/research/3DSSR"
        # parallel -j5 "python3 -u build_cl_rings.py main {1} {2}" ::: 5 ::: 0 1 2 3 4
        main(int(sys.argv[2]), int(sys.argv[3]))
