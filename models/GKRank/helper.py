import os
import shutil
import struct
import numpy as np
import json
import pandas as pd


def load_from_json(path, mode='r'):
    with open(path, mode) as f:
        return json.load(f)


def write_to_json(dictionary, path, mode='w', indent=4):
    with open(path, mode) as f:
        json.dump(dictionary, f, indent=indent)


def compute_descriptors(command):
    stream = os.popen(command)
    stream.read()


def read_zernike_descriptors(file_name):
    f = open(file_name, 'rb')
    dim = struct.unpack('i', f.read(4))[0]
    if dim != 121:
        raise ValueError('You Must Use 20 Moments to Get 121 Descriptors')
    data = np.asarray(struct.unpack('f'*121, f.read(4*121)))
    return data


def find_diverse_subset(subset_size, df):
    # filter the metadata to train data only
    df = df.loc[df['split'] == 'train'].copy()
    # add unique key for each object
    df['scene_object'] = df[['room_name', 'objectId']].apply(lambda x: '-'.join([x['room_name'], str(x['objectId'])]),
                                                             axis=1)
    # map each category to the scene object keys
    cat_to_scene_objects = {}

    def map_cat_to_scene_objects(x):
        if x['mpcat40'] in cat_to_scene_objects:
            cat_to_scene_objects[x['mpcat40']].append(x['scene_object'])
        else:
            cat_to_scene_objects[x['mpcat40']] = [x['scene_object']]
    df[['mpcat40', 'scene_object']].apply(map_cat_to_scene_objects, axis=1)

    # sample from each category
    subset = []
    cat_to_num_objects = [(cat, len(cat_to_scene_objects[cat])) for cat in cat_to_scene_objects.keys()]
    cat_to_num_objects = sorted(cat_to_num_objects, key=lambda x: x[1])
    min_sample = subset_size // len(cat_to_scene_objects)
    counter = 0
    for cat, num_objects in cat_to_num_objects:
        counter += 1
        if num_objects <= min_sample:
            subset += cat_to_scene_objects[cat]
        else:
            subset += np.random.choice(cat_to_scene_objects[cat], min_sample, replace=False).tolist()
        remaining = subset_size - len(subset)
        if remaining > 0:
            min_sample = remaining // (len(cat_to_num_objects) - counter)

    return subset


def nth_closest_descriptor(voxel_dir, subset_size, metadata_path, n=2):
    voxel_names = [voxel for voxel in os.listdir(voxel_dir) if voxel.endswith('.inv')]
    # take a subset of the voxels for normalizing the geometric kernel.
    df_metadata = pd.read_csv(metadata_path)
    voxel_names_subset = find_diverse_subset(subset_size, df_metadata)
    voxel_names_subset = {voxel_name+'.inv' for voxel_name in voxel_names_subset}

    nth_closest_dict = {}
    for i, voxel_name1 in enumerate(voxel_names):
        inv_file_path = os.path.join(voxel_dir, voxel_name1)
        voxel1_data = read_zernike_descriptors(inv_file_path)
        dist = []
        candidates = []
        for voxel_name2 in voxel_names_subset:
            inv_file_path = os.path.join(voxel_dir, voxel_name2)
            voxel2_data = read_zernike_descriptors(inv_file_path)
            dist.append(np.linalg.norm((voxel1_data-voxel2_data)**2))
            candidates.append(voxel_name2)
        nth_closest_idx = np.argsort(dist)[n-1]
        nth_closest_dict[voxel_name1] = (candidates[nth_closest_idx], dist[nth_closest_idx])
        print('Finished processing {}/{} voxels'.format(i, len(voxel_names)))
    return nth_closest_dict


def sample_mesh(mesh, count=1000):
    """
    Sample points from the mesh.
    :param mesh: Mesh representing the 3d object.
    :param count: Number of query points/
    :return: Sample points on the mesh and the face index corresponding to them.
    """
    faces_idx = np.zeros(count, dtype=int)
    points = np.zeros((count, 3), dtype=float)
    # pick a triangle randomly promotional to its area
    cum_area = np.cumsum(mesh.area_faces)
    random_areas = np.random.uniform(0, cum_area[-1]+1, count)
    for i in range(count):
        face_idx = np.argmin(np.abs(cum_area - random_areas[i]))
        faces_idx[i] = face_idx
        r1, r2, = np.random.uniform(0, 1), np.random.uniform(0, 1)
        triangle = mesh.triangles[face_idx, ...]
        point = (1 - np.sqrt(r1)) * triangle[0, ...] + \
            np.sqrt(r1) * (1 - r2) * triangle[1, ...] + \
            np.sqrt(r1) * r2 * triangle[2, ...]
        points[i, :] = point
    return points, faces_idx


def create_train_val_test(scene_graph_dir, train_path, val_path, test_path):
    # make sure the 3 folders exist
    folder_to_path = {'train': train_path, 'val': val_path, 'test': test_path}
    for folder in folder_to_path.keys():
        path = os.path.join(scene_graph_dir, folder)
        if not os.path.exists(path):
            os.mkdir(path)

    # for each house find out which folder (train, val and test) it belongs to
    house_to_folder = {}
    for folder, path in folder_to_path.items():
        with open(path, 'r') as f:
            house_names = f.readlines()
        for house_name in house_names:
            house_name = house_name.strip()
            house_to_folder[house_name] = folder

    # for each scene find out which folder it belongs to and copy it there
    scene_names = os.listdir(os.path.join(scene_graph_dir, 'all'))
    for scene_name in scene_names:
        house_name = scene_name.split('_')[0]
        folder = house_to_folder[house_name]
        d1 = os.path.join(scene_graph_dir, 'all', scene_name)
        d2 = os.path.join(scene_graph_dir, folder, scene_name)
        shutil.copy(d1, d2)
