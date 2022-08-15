import os
import numpy as np
import pandas as pd


def map_nuy40Id_to_cat():
    # load the combined label df.
    df = pd.read_csv(os.path.join(data_root, 'meta_data', 'scannetv2-labels.combined.tsv'), delimiter='\t')
    id_to_cat = {}
    for i in range(1, 41):
        cat = df.loc[df['nyu40id'] == i, 'nyu40class'].unique()
        if len(cat) > 1:
            raise ValueError('NYU40 id for {} is mapped to {} categories: {}'.format(i, len(cat), cat))
        id_to_cat[i] = cat[0]

    return id_to_cat


def read_rooms(path):
    with open(path, 'r') as f:
        rooms = f.readlines()
        rooms = set([room.strip() for room in rooms])

    return rooms


def main():
    # find all scene names.
    scene_names = set([e[:12] for e in os.listdir(data_dir)])

    # load the mapping from nyu40 ids to categories.
    nyu40Id_to_cat = map_nuy40Id_to_cat()

    # map each room to train, test and val
    train_rooms = read_rooms(train_split_path)
    val_rooms = read_rooms(val_split_path)
    test_rooms = read_rooms(test_split_path)
    room_to_split = {}
    folder_to_rooms = {'train': train_rooms, 'val': val_rooms, 'test': test_rooms}
    for folder, rooms in folder_to_rooms.items():
        for room in rooms:
            room_to_split[room] = folder

    # add the object info for each scene.
    df_data = {'room_name': [], 'objectId': [], 'nyu40Id': [], 'nyu40_category': [], 'aabb': [], 'split': []}
    for scene_name in scene_names:
        split = room_to_split[scene_name]
        bbox = np.load(os.path.join(data_dir, '{}_bbox.npy'.format(scene_name)))
        for i, box in enumerate(bbox):
            # record room name and ids.
            df_data['room_name'].append(scene_name)
            df_data['objectId'].append(i)
            df_data['nyu40Id'].append(int(box[-1]))

            # map nyu40 id to category.
            df_data['nyu40_category'].append(nyu40Id_to_cat[int(box[-1])])

            # record the box and the split the scene belongs to.
            df_data['aabb'].append(list(box[:-1]))
            df_data['split'].append(split)

    # create the frame.
    csv_path = os.path.join(data_root, 'metadata.csv')
    df_metadata = pd.DataFrame(df_data)
    df_metadata = df_metadata.sort_values(by=['room_name', 'objectId'])

    # save the metadata
    df_metadata.to_csv(csv_path, index=False)


if __name__ == '__main__':
    # define paths
    data_root = '../data/scannet'
    data_dir = os.path.join(data_root, 'scannet_train_detection_data')
    train_split_path = os.path.join(data_root, 'meta_data', 'scannetv2_train.txt')
    val_split_path = os.path.join(data_root, 'meta_data', 'scannetv2_val.txt')
    test_split_path = os.path.join(data_root, 'meta_data', 'scannetv2_test.txt')

    main()

