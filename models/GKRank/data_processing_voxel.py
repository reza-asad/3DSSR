import os
import sys
import numpy as np
from scipy import ndimage
import struct
from time import time
import trimesh.voxel.creation as creation
import pandas as pd

from obj_3d import Mesh
from helper import nth_closest_descriptor, compute_descriptors, load_from_json, write_to_json


class Voxel:
    def __init__(self, models_dir, voxel_dir, res=128, added_thickness=4):
        self.models_dir = models_dir
        self.voxel_dir = voxel_dir

        self.encoding = []
        self.res = res
        self.added_thickness = added_thickness

    @staticmethod
    def find_padding(H, res):
        pad_left = (res - H) // 2
        pad_right = res - pad_left - H
        return pad_left, pad_right

    @staticmethod
    def thicken_voxel(voxel_encoding, res, kernel_size=5):
        # pad the encoding to the resolution
        H, W, L = voxel_encoding.shape
        pad_h = Voxel.find_padding(H, res)
        pad_w = Voxel.find_padding(W, res)
        pad_l = Voxel.find_padding(L, res)
        voxel_encoding = np.pad(voxel_encoding, (pad_h, pad_w, pad_l), mode='constant', constant_values=False)
        return ndimage.convolve(voxel_encoding, np.ones((kernel_size, kernel_size, kernel_size)), mode='constant',
                                cval=False)

    def from_mesh(self, model_name, thicken=True):
        # load the mesh
        model_path = os.path.join(self.models_dir, model_name)
        mesh_obj = Mesh(model_path)
        mesh = mesh_obj.load()

        min_vals = np.min(mesh.vertices, 0)
        max_vals = np.max(mesh.vertices, 0)
        idx = np.argmax(max_vals - min_vals)
        mesh.vertices = (mesh.vertices - min_vals[idx]) / (max_vals[idx] - min_vals[idx]) - 1./2
        voxel = creation.voxelize_binvox(mesh, pitch=1./(self.res-2), exact=True, binvox_path='../GKRank/binvox')
        # in case the mesh was too thin use the subdivision apporach for voxelization
        if voxel.matrix.sum() == 0:
            voxel = creation.voxelize_subdivide(mesh, pitch=1./(self.res-2))
        if voxel.matrix.sum() == 0:
            raise ValueError('Created an empty voxel!')
        if thicken:
            self.encoding = self.thicken_voxel(voxel.matrix, self.res+self.added_thickness)

    def save(self, file_name):
        # save the voxels into a file
        voxel_path = os.path.join(self.voxel_dir, file_name)
        with open(voxel_path, 'wb') as file:
            for x in range(self.res):
                for y in range(self.res):
                    for z in range(self.res):
                        value = bytearray(struct.pack("f", float(self.encoding[x, y, z])))
                        file.write(value)
        file.close()


def process_meshes(models_dir, voxel_dir, model_names):
    t = time()
    for i, model_name in enumerate(model_names):
        print('Processing mesh {} ... '.format(model_name))
        print('Iteration {}/{}'.format(i+1, len(model_names)))
        if model_name in visited:
            continue
        t2 = time()
        # derive the voxel and save it
        v = Voxel(models_dir, voxel_dir, res=128)
        v.from_mesh(model_name)
        voxel_name = model_name.split('.')[0] + '.txt'
        v.save(voxel_name)
        # derive the zernike descriptors and save it
        command = './ZernikeMoments-master/examples/zernike3d {}/{} 20'.format(voxel_dir, voxel_name)
        compute_descriptors(command)
        # delete the voxel file
        os.remove(os.path.join(voxel_dir, voxel_name))
        duration = (time() - t2) / 60
        print('Processing mesh {} took {} minutes'.format(model_name, round(duration, 2)))
        print('-' * 50)
    duration_all = (time() - t) / 60
    print('Processing {} objects took {} minutes'.format(i+1, round(duration_all, 2)))


def main(num_chunks, chunk_idx, action='derive_zernike_features'):
    data_dir = '../../data/matterport3d'
    models_dir = os.path.join(data_dir, 'models')
    accepted_cats = load_from_json(os.path.join(data_dir, 'accepted_cats.json'))
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    voxel_dir = os.path.join(data_dir, 'voxels')
    if not os.path.exists(voxel_dir):
        os.mkdir(voxel_dir)

    if action == 'derive_zernike_features':
        # filter models to only include accepted cats.
        df_metadata = pd.read_csv(metadata_path)
        is_accepted = df_metadata['mpcat40'].apply(lambda x: x in accepted_cats)
        df_metadata = df_metadata.loc[is_accepted]
        model_names = df_metadata[['room_name', 'objectId']].apply(lambda x: '-'.join([x['room_name'],
                                                                                       str(x['objectId'])]) + '.ply', axis=1)
        model_names = model_names.tolist()
        chunk_size = int(np.ceil(len(model_names) / num_chunks))
        process_meshes(models_dir, voxel_dir, model_names[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size])

    if action == 'find_nth_closest_model':
        nth_closest_dict = nth_closest_descriptor(voxel_dir, subset_size=1000, metadata_path=metadata_path, n=100)
        # save the nth_closest dict
        write_to_json(nth_closest_dict, os.path.join(data_dir, 'nth_closest_obj.json'))

        data = load_from_json(os.path.join(data_dir, 'nth_closest_obj.json'))
        for k, v in data.items():
            if pd.isna(v[1]):
                print(k, v)


if __name__ == '__main__':
    visited = set()
    if len(sys.argv) == 1:
        main(1, 0, 'derive_zernike_features')
    elif len(sys.argv) == 2:
        main(1, 0, sys.argv[1])
    else:
        # To run in parallel you can use the command:
        # parallel -j5 "python3 -u data_processing_voxel.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: derive_zernike_features
        main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
