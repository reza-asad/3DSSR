import os
import json
import numpy as np
import pandas as pd
import ast

from obj_3d import Mesh


class BaseScene (object):
    def __init__(self, models_dir, scene_graph_dir, scene_name, accepted_cats):
        """
        Initialize an instance of a scene
        :param models_dir: The path to the models directory
        :param scene_name: Name of the scene
        """
        self.models_dir = models_dir

        self.scene_graph_dir = scene_graph_dir
        self.scene_name = scene_name
        self.accepted_cats = accepted_cats
        self.graph = {}

    def filter_by_accepted_cats(self):
        filtered_graph = {}
        for node, node_info in self.graph.items():
            if node_info['category'][0] in self.accepted_cats:
                filtered_graph[node] = node_info
        self.graph = filtered_graph

    def prepare_mesh_for_scene(self, obj, graph):
        """
        This load and transforms a mesh according to the scene.
        :param obj: String id for the mesh to be loaded
        :return: Mesh representing the object
        """
        model_path = os.path.join(self.models_dir, graph[obj]['file_name'])
        mesh_obj = Mesh(model_path)
        transform = np.asarray(graph[obj]['transform'], dtype=np.float).reshape(4, 4).transpose()
        return mesh_obj.load(transform)

    def build_from_example_based(self, obj_to_category, scene_recipe_dir):
        """
        Build a scene graph based on hierarchy. The format is {'obj_Id': {'neighbours': {}, , 'transform':[],
        'mesh': None}}
        :param obj_to_category: Dictionary that maps an obj file to its hierarchical category
        :param scene_recipe_dir: The path to the text file containing scene recipe
        :return: Populates the graph dictionary with the items like described above
        """
        # read the recipe
        scene_recipe = open(scene_recipe_dir, 'r')

        for line in scene_recipe.readlines():
            words = line.split()
            # e.g newModel 0 room2. O is the id and room2 is the obj file name
            if words[0] == 'newModel':
                obj_file = words[-1] + '.obj'
                model_path = os.path.join(self.models_dir, obj_file)
                curr_object = words[1]
                mesh_obj = Mesh(model_path, obj_to_category)
                self.graph[curr_object] = {'category': mesh_obj.category,
                                           'neighbours': {},
                                           'scale': None,
                                           'transform': [],
                                           'file_name': obj_file}
            elif words[0] == 'children':
                for i in range(1, len(words)):
                    self.graph[curr_object]['neighbours'][words[i]] = ['parent']
            elif words[0] == 'scale':
                self.graph[curr_object]['scale'] = float(words[1])
            elif words[0] == 'transform':
                transform = [float(e) for e in words[1:]]
                self.graph[curr_object]['transform'] = transform

    @staticmethod
    def compute_transformation(translation):
        t_x, t_y, t_z = translation
        R_t = np.eye(3, 3, dtype=np.float)

        # package the transformation matrix
        transformation = np.zeros((4, 4), dtype=np.float)
        transformation[:3, :3] = R_t
        transformation[0, 3] = t_x
        transformation[1, 3] = t_y
        transformation[2, 3] = t_z
        transformation[3, 3] = 1.0

        return transformation.T.reshape(16)

    def build_from_matterport(self, scene_name, csv_path):
        # read the metadata and filter it to the current scene
        df = pd.read_csv(csv_path)
        df = df[df['room_name'] == scene_name]
        if len(df) == 0:
            return
        df['key'] = df.apply(lambda x: '-'.join([scene_name, str(x['objectId'])]), axis=1)

        # build the initial scene graph
        key_to_translation = dict(zip(df['key'], df['translation']))
        key_to_cat = dict(zip(df['key'], df['mpcat40']))
        for key in key_to_cat.keys():
            category = key_to_cat[key]
            # derive the transformation for mapping the mesh to the scene
            translation = ast.literal_eval(key_to_translation[key])
            translation = [float(e) for e in translation]
            transformation = self.compute_transformation(translation)
            transformation = list(transformation)

            _, object_index = key.split('-')
            self.graph[object_index] = {'category': [category],
                                        'neighbours': {},
                                        'scale': 1,
                                        'transform': transformation,
                                        'file_name': key+'.ply'}

    def to_json(self):
        """
        Write the scene graph built into json.
        :return:
        """
        file_path = os.path.join(self.scene_graph_dir, self.scene_name + '.json')
        with open(file_path, 'w') as f:
            json.dump(self.graph, f, indent=4)
