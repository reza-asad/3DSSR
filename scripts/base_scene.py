import os
import numpy as np
import pandas as pd
import ast
import trimesh

from mesh import Mesh


class BaseScene:
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.graph = {}

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

    def compute_box(self, obj, centroid):
        mesh = self.prepare_mesh_for_scene(obj)
        obbox = [centroid] + mesh.bounding_box_oriented.vertices.tolist()
        return obbox

    def build_from_matterport(self, scene_name, csv_path):
        # read the metadata and filter it to the current scene
        df_metadata = pd.read_csv(csv_path)
        df_metadata = df_metadata[df_metadata['room_name'] == scene_name]
        if len(df_metadata) == 0:
            return
        df_metadata['key'] = df_metadata.apply(lambda x: '-'.join([scene_name, str(x['objectId'])]), axis=1)

        # build the initial scene graph
        key_to_translation = dict(zip(df_metadata['key'], df_metadata['translation']))
        key_to_cat = dict(zip(df_metadata['key'], df_metadata['mpcat40']))
        for key in key_to_cat.keys():
            category = key_to_cat[key]
            # derive the transformation for mapping the mesh to the scene
            translation = ast.literal_eval(key_to_translation[key])
            translation = [float(e) for e in translation]
            transformation = self.compute_transformation(translation)
            transformation = list(transformation)

            _, object_index = key.split('-')
            self.graph[object_index] = {'category': [category],
                                        'transform': transformation,
                                        'file_name': key+'.ply'}
            # save the centroid and vertices of the object's obbox
            self.graph[object_index]['obbox'] = self.compute_box(object_index, translation)

    def prepare_mesh_for_scene(self, node):
        model_path = os.path.join(self.models_dir, self.graph[node]['file_name'])
        mesh_obj = Mesh(model_path, self.graph[node]['transform'])
        return mesh_obj.load(with_transform=True)

    def visualize(self):
        scene = []
        for node, node_info in self.graph.items():
            mesh = self.prepare_mesh_for_scene(node)
            scene += [mesh]
        if len(self.graph) == 0:
            raise ValueError('You need to populate the graph first')
        scene = trimesh.Scene(scene)
        scene.show()



