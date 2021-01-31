import trimesh
import numpy as np


class Mesh(object):
    def __init__(self, model_path, obj_to_category=None, obj_to_front=None, obj_to_up=None, default_front=None,
                 default_up=None):
        """
        Initialize a mesh object
        :param model_path: The path to the obj file
        :param obj_to_category: Optional dictionary with obj file as key and category as value
        """
        self.model_path = model_path
        self.obj_file = self.model_path.split('/')[-1]

        self.default_front = default_front
        self.default_up = default_up

        if obj_to_category is not None:
            self.category = self.add_category(obj_to_category)
        if obj_to_front is not None:
            self.front_dir = self.find_front_direction(obj_to_front)
        if obj_to_up is not None:
            self.up_dir = self.find_up_direction(obj_to_up)

    def load(self, transform=None):
        """
        :param transform: Optional 4x4 transformation matrix for the mesh
        :return:
        """
        mesh = trimesh.exchange.load.load(self.model_path)
        if transform is not None:
            mesh.apply_transform(transform)
        return mesh

    def add_category(self, obj_to_category):
        if self.obj_file in obj_to_category:
            return obj_to_category[self.obj_file]
        return []

    def find_front_direction(self, obj_to_front):
        if self.obj_file in obj_to_front:
            front_dir = np.asarray(obj_to_front[self.obj_file])
        elif self.default_front is not None:
            front_dir = self.default_front
            print('using default front direction')
        else:
            raise ValueError('Front direction is missing for object {}'.format(self.obj_file))
        return front_dir / np.linalg.norm(front_dir)

    def find_up_direction(self, obj_to_up):
        if self.obj_file in obj_to_up:
            up_dir = np.asarray(obj_to_up[self.obj_file])
        elif self.default_up is not None:
            up_dir = self.default_up
            print('using default up direction')
        else:
            raise ValueError('Up direction is missing for object {}'.format(self.obj_file))
        return up_dir / np.linalg.norm(up_dir)

    def compute_coordinate_frame(self):
        coordinate_frame = np.zeros((3, 3))
        coordinate_frame[:, 1] = self.front_dir
        coordinate_frame[:, 2] = self.up_dir
        side_dir = np.cross(self.front_dir, self.up_dir)
        coordinate_frame[:, 0] = side_dir / np.linalg.norm(side_dir)
        return coordinate_frame
