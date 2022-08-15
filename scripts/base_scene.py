import numpy as np
import ast


class BaseScene:
    def __init__(self):
        self.graph = {}

    @staticmethod
    def compute_transformation(translation):
        R_t = np.eye(3, 3, dtype=np.float)

        # package the transformation matrix
        transformation = np.ones((4, 4), dtype=np.float)
        transformation[:3, :3] = R_t
        transformation[:3, 3] = translation
        transformation[3, 3] = 1.0

        return transformation.T.reshape(16)

    def build_from_metadata(self, scene_name, df_metadata):
        df_metadata_scene = df_metadata[df_metadata['room_name'] == scene_name]

        if len(df_metadata_scene) == 0:
            return

        # build the initial scene graph
        key_to_box = dict(zip(df_metadata_scene['key'], df_metadata_scene['aabb']))
        key_to_cat = dict(zip(df_metadata_scene['key'], df_metadata_scene['nyu40_category']))

        for key in key_to_cat.keys():
            # load the cat
            category = key_to_cat[key]

            # derive the transformation for mapping the box to the scene
            box = ast.literal_eval(key_to_box[key])
            translation = [float(e) for e in box[:3]]
            transformation = self.compute_transformation(translation)
            transformation = list(transformation)

            _, object_index = key.split('-')
            self.graph[object_index] = {
                'category': [category],
                'transform': transformation,
                'aabb': box
            }



