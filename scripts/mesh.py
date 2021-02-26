import trimesh
import numpy as np


class Mesh(object):
    def __init__(self, model_path, transform=None):
        self.model_path = model_path
        self.file_name = self.model_path.split('/')[-1]
        self.transform = transform

    def load(self, with_transform=False):
        mesh = trimesh.load(self.model_path)
        if with_transform and self.transform is not None:
            self.transform = np.asarray(self.transform, dtype=np.float).reshape(4, 4).transpose()
            mesh.apply_transform(self.transform)
        return mesh
