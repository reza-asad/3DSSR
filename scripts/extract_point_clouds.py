import os
import sys
import numpy as np
import pandas as pd
import trimesh

from scripts.mesh import Mesh
from scripts.helper import sample_mesh, load_from_json, create_train_val_test


class PointCloud:
    def __init__(self, models_dir, model_name, results_dir, num_points=1024):
        # load the mesh
        self.model_name = model_name
        mesh_path = os.path.join(models_dir, model_name)
        mesh_obj = Mesh(mesh_path)
        self.mesh = mesh_obj.load()

        self.results_dir = results_dir

        # sample points on the mesh
        self.num_points = num_points
        self.pc, _ = sample_mesh(self.mesh, count=self.num_points)

    def save(self):
        file_name = self.model_name.split('.')[0]
        path = os.path.join(self.results_dir, file_name)
        np.save(path, self.pc)

    def visualize(self):
        trimesh.points.PointCloud(self.pc).show()


def derive_pc(models_dir, file_names, results_dir, num_points):
    for file_name in file_names:
        pc_object = PointCloud(models_dir, file_name, results_dir, num_points)
        pc_object.save()


def main(num_chunks, chunk_idx, action='extract_pc'):
    # define some params
    models_dir = '../data/matterport3d/models'
    accepted_cats = load_from_json('../data/matterport3d/accepted_cats.json')
    metadata_path = '../data/matterport3d/metadata.csv'
    results_dir = '../data/matterport3d/point_clouds/all'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if action == 'extract_pc':
        # filter models to only include accepted cats.
        df_metadata = pd.read_csv(metadata_path)
        is_accepted = df_metadata['mpcat40'].apply(lambda x: x in accepted_cats)
        df_metadata = df_metadata.loc[is_accepted]
        model_names = df_metadata[['room_name', 'objectId']].apply(lambda x: '-'.join([x['room_name'],
                                                                                       str(x['objectId'])]) + '.ply', axis=1)
        model_names = model_names.tolist()
        chunk_size = int(np.ceil(len(model_names) / num_chunks))
        derive_pc(models_dir=models_dir, file_names=model_names[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size],
                  results_dir=results_dir, num_points=2048)

    if action == 'split_train_test_val':
        data_dir = '../data/matterport3d'
        results_dir = '../data/matterport3d/point_clouds'
        train_path = os.path.join(data_dir, 'scenes_train.txt')
        val_path = os.path.join(data_dir, 'scenes_val.txt')
        test_path = os.path.join(data_dir, 'scenes_test.txt')
        create_train_val_test(results_dir, train_path, val_path, test_path)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main(1, 0, 'extract_pc')
    elif len(sys.argv) == 2:
        main(1, 0, sys.argv[1])
    else:
        # To run in parallel you can use the command:
        # parallel -j5 "python3 -u extract_point_clouds.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: extract_pc
        main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
