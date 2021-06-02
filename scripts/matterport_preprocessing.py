import os
import sys
import numpy as np
from plyfile import PlyData, PlyElement
import trimesh
import pandas as pd

from scripts.helper import load_from_json, write_to_json


def compute_transformation(translation):
    t_x, t_y, t_z = translation
    R_t = np.eye(3, 3, dtype=np.float)

    # package the transformation matrix
    transformation = np.zeros((4, 4), dtype=np.float)
    transformation[:3, :3] = R_t
    transformation[0, 3] = -t_x
    transformation[1, 3] = -t_y
    transformation[2, 3] = -t_z
    transformation[3, 3] = 1.0

    return transformation


def extract_meshes(room_name, ply_path, models_dir, obj_properties_dict):
    # read the file into ascii
    with open(ply_path, 'rb') as f:
        ply_ascii = PlyData.read(f)

    # find a mapping from objectIds to vertex indices and faces
    object_id_to_vertices = {}
    object_id_to_faces = {}
    for face in ply_ascii['face'].data:
        face = face[0]
        # find all the unique objectIds associated with each face
        object_ids = set([ply_ascii['vertex']['objectId'][v] for v in face])

        # each object_id will contain each vertex
        for object_id in object_ids:
            if object_id not in object_id_to_vertices:
                object_id_to_vertices[object_id] = set(face)
            else:
                for v in face:
                    object_id_to_vertices[object_id].add(v)

            # similarly each object id will contain the face
            if object_id not in object_id_to_faces:
                object_id_to_faces[object_id] = [(face, )]
            else:
                object_id_to_faces[object_id].append((face, ))

    # for each object id filter the room to each object
    for object_id, vertices in object_id_to_vertices.items():
        vertex_list = list(vertices)
        mesh_vertices_full_data = ply_ascii['vertex'].data[vertex_list]

        # extract the x, y and z of for the vertices
        mesh_vertices = np.zeros((len(vertices), 4), dtype=float)
        mesh_vertices[:, 0] = mesh_vertices_full_data['x']
        mesh_vertices[:, 1] = mesh_vertices_full_data['y']
        mesh_vertices[:, 2] = mesh_vertices_full_data['z']
        mesh_vertices[:, 3] = 1.0

        # map the index of old vertices to their new index
        vertex_index_map = dict(zip(vertex_list, range(len(mesh_vertices_full_data))))

        # map the vertices in the object's face to their new value
        mesh_faces = []
        for face in object_id_to_faces[object_id]:
            face = face[0]
            new_face = []
            for v in face:
                new_face.append(vertex_index_map[v])
            mesh_faces += [(new_face, )]
        mesh_faces = np.asarray(mesh_faces, dtype=ply_ascii['face'].data.dtype)

        # translate the mesh to the origin
        centroid = np.mean(mesh_vertices, axis=0)[:3]
        transformation = compute_transformation(centroid)
        transformed_mesh_vertices = transformation.dot(mesh_vertices.T).T
        mesh_vertices_full_data['x'] = transformed_mesh_vertices[:, 0]
        mesh_vertices_full_data['y'] = transformed_mesh_vertices[:, 1]
        mesh_vertices_full_data['z'] = transformed_mesh_vertices[:, 2]

        # extract the properties for this object and add it to the obj_properties_dict
        obj_properties_dict['room_name'].append(room_name)
        obj_properties_dict['translation'].append(list(centroid))
        curr_object = ply_ascii['vertex']['objectId'] == object_id
        property_values = list(ply_ascii['vertex'][curr_object][0])[6:]
        property_keys = ply_ascii['vertex'].properties[6:]
        for i, k in enumerate(property_keys):
            obj_properties_dict[k.name].append(int(property_values[i]))

        # convert the mesh and vertex arrays to a .ply file
        vertex_elm = PlyElement.describe(mesh_vertices_full_data, 'vertex')
        face_elm = PlyElement.describe(mesh_faces, 'face')
        key = '-'.join([room_name, str(object_id)])
        output_path = os.path.join(models_dir, key + '.ply')
        PlyData([vertex_elm, face_elm]).write(output_path)
    return obj_properties_dict


def read_houses(path):
    with open(path, 'r') as f:
        houses = f.readlines()
        houses = set([house.strip() for house in houses])
    return houses


def main(num_chunks, chunk_idx, action='extract_mesh'):
    # define paths and set up folders for the extracted object meshes
    root_dir = '../data/matterport3d'
    models_dir = os.path.join(root_dir, 'models')
    if not os.path.exists(models_dir):
        try:
            os.mkdir(models_dir)
        except FileExistsError:
            pass

    if action == 'extract_mesh':
        rooms_dir = os.path.join(root_dir, 'rooms')
        room_names = os.listdir(rooms_dir)
        chunk_size = int(np.ceil(len(room_names) / num_chunks))
        room_names = room_names[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size]
        obj_properties_dict = {'room_name': [], 'objectId': [], 'categoryId': [], 'NYU40': [], 'mpr40': [],
                               'translation': []}
        # extract meshes from each room
        for room_name in room_names:
            if room_name in visited:
                continue
            visited.add(room_name)
            # extract the meshes from each room
            print('Processing room {}'.format(room_name))
            ply_path = os.path.join(rooms_dir, room_name, room_name+'.annotated.ply')
            if not os.path.exists(ply_path):
                continue
            obj_properties_dict = extract_meshes(room_name, ply_path, models_dir, obj_properties_dict)

        # save the extracted mesh properties
        write_to_json(obj_properties_dict, '../data/matterport3d/mesh_properties{}.json'.format(chunk_idx))

    # save the extracted properties
    if action == 'save_metadata':
        file_names = os.listdir(root_dir)
        property_dicts = []
        for file_name in file_names:
            if file_name.split('.')[0][:-1] == 'mesh_properties':
                obj_properties_dict = load_from_json(os.path.join(root_dir, file_name))
                property_dicts.append(obj_properties_dict)
        dict0 = property_dicts[0]
        for property_dict in property_dicts[1:]:
            for k, v in property_dict.items():
                dict0[k] += v
        # create a metadata dataframe
        csv_path = os.path.join(root_dir, 'metadata.csv')
        df_metadata = pd.DataFrame.from_dict(dict0)

        # read the dataframe mapping category ids to categories
        df_cats = pd.read_csv(os.path.join(root_dir, 'category_mapping.tsv'), delimiter='\t')
        df_metadata = df_metadata.merge(df_cats[['index', 'mpcat40']], how='inner', left_on='categoryId', right_on='index')
        df_metadata = df_metadata.sort_values(by='room_name')

        # map each house to train, test and val
        train_houses = read_houses('../data/matterport3d/scenes_train.txt')
        val_houses = read_houses('../data/matterport3d/scenes_val.txt')
        test_houses = read_houses('../data/matterport3d/scenes_test.txt')
        house_to_folder = {}
        folder_to_houses = {'train': train_houses, 'val': val_houses, 'test': test_houses}
        for folder, houses in folder_to_houses.items():
            for house in houses:
                house_to_folder[house] = folder

        # map each room object to the folder they belong
        df_metadata['split'] = df_metadata['room_name'].apply(lambda x: house_to_folder[x.split('_')[0]])

        # save the metadata
        df_metadata.to_csv(csv_path, index=False)


if __name__ == '__main__':
    visited = set()
    if len(sys.argv) == 1:
        main(1, 0, 'extract_mesh')
    elif len(sys.argv) == 2:
        main(1, 0, sys.argv[1])
    else:
        # parallel -j5 "python3 -u matterport_preprocessing.py {1} {2} {3}" ::: 5 ::: 0 1 2 3 4 ::: extract_mesh
        main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
