import os
import sys
import trimesh
import numpy as np
import pandas as pd

from scripts.helper import write_to_json


def main(num_chunks, chunk_idx):
    visited = set(os.listdir(scene_dir))
    chunk_size = int(np.ceil(len(obj_file_names) / num_chunks))
    obj_file_names_chunk = obj_file_names[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size]
    for file_name in obj_file_names_chunk:
        scene_file_name = file_name.split('.')[0] + '.json'
        if scene_file_name in visited:
            continue
        cat = df_metadata.loc[df_metadata['objectId'] == file_name.split('.')[0] + '.obj', 'mpcat40'].values[0]
        pc = np.load(os.path.join(obj_dir, file_name))
        centroid = np.mean(pc, axis=0)
        centroid = [float(e) for e in centroid]
        pc = trimesh.points.PointCloud(pc)
        try:
            vertices = pc.bounding_box_oriented.vertices.tolist()
        except Exception:
            vertices = pc.bounding_box.vertices.tolist()
        obbox = [centroid] + vertices
        scene = {'0': {'category': [cat], 'obbox': obbox}}
        write_to_json(scene, os.path.join(scene_dir, scene_file_name))


if __name__ == '__main__':
    # define the paths
    mode = 'train'
    data_dir = '../data/shapenetsem'
    obj_dir = '/media/reza/Large/shapenetsem/objects_pc/{}'.format(mode)#os.path.join(data_dir, 'objects_pc', mode)
    obj_file_names = os.listdir(obj_dir)
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    df_metadata = pd.read_csv(metadata_path)
    scene_dir = os.path.join(data_dir, 'scenes', mode)
    if not os.path.exists(scene_dir):
        try:
            os.makedirs(scene_dir)
        except FileExistsError:
            pass

    if len(sys.argv) == 1:
        main(1, 0)
    else:
        # To run in parallel you can use the command:
        # parallel -j5 "python3 -u build_shapenetsem_scenes.py {1} {2}" ::: 5 ::: 0 1 2 3 4
        main(int(sys.argv[1]), int(sys.argv[2]))
