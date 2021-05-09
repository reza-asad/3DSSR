import os
import numpy as np
import pandas as pd
from time import time

from scripts.helper import load_from_json, write_to_json


def map_cats_to_filenames(df):
    cats_to_file_names = {}
    cats = df['mpcat40'].values
    file_names = df['latent_caps'].values
    for i, cat in enumerate(cats):
        if cat not in cats_to_file_names:
            cats_to_file_names[cat] = [file_names[i]]
        else:
            cats_to_file_names[cat].append(file_names[i])
    return cats_to_file_names


def add_context_objects(scene_graph_dir, file_name, mode, topk_context=4):
    # load the scene graph
    scene_name = file_name.split('-')[0]
    obj = file_name.split('-')[1].split('.')[0]
    graph = load_from_json(os.path.join(scene_graph_dir, mode, scene_name + '.json'))

    # read the distances of each object in the scene to the center object
    obj_distances = []
    for metrics in graph[obj]['ring_info'].values():
        obj_distances += metrics['distance']

    # sort the distances
    sorted_obj_distances = sorted(obj_distances, key=lambda x: x[1])

    # take the topk context objects. if you did not find enough context object repeat the center object.
    context_objects = list(list(zip(*sorted_obj_distances))[0][:topk_context])
    num_remianing = topk_context - len(context_objects)
    while num_remianing > 0:
        context_objects = [obj] + context_objects
        num_remianing -= 1

    # add the center obj to the beginning and convert the ids to file names.
    all_objects = [obj] + context_objects
    file_names = []
    for obj in all_objects:
        file_names.append(scene_name + '-' + obj + '.npy')

    return file_names[::-1]


def process_scenes(latent_caps_dir, cats_to_file_names, caps_size=4096):
    # find the mean of the shape embeddings in each cluster (category).
    cats_to_mean_caps = {}
    for cat, file_names in cats_to_file_names.items():
        sum_ = np.zeros((1, caps_size))
        for i, file_name in enumerate(file_names):
            # load the latent caps
            latent_caps = np.load(os.path.join(latent_caps_dir, file_name))
            latent_caps = latent_caps.reshape(1, caps_size)
            sum_ += latent_caps

        # record the mean latent caps for this category
        cats_to_mean_caps[cat] = sum_ / (i + 1)

    # for each category find the distance between its mean latent caps and the latent caps of other categories.
    closest_shape_clusters = {}
    for cat_self, mean_caps_self in cats_to_mean_caps.items():
        closest_shape_clusters[cat_self] = []
        for cat_other, mean_caps_other in cats_to_mean_caps.items():
            if cat_other != cat_self:
                distance = np.linalg.norm(mean_caps_self - mean_caps_other)
                closest_shape_clusters[cat_self].append((cat_other, distance))

    # sort the distances from closest to furthest
    for cat, cats_distances in closest_shape_clusters.items():
        closest_shape_clusters[cat] = sorted(cats_distances, key=lambda x: x[1])

    return closest_shape_clusters


def main():
    # define the paths
    latent_caps_dir = '../../../3D-point-capsule-networks/dataset/matterport3d/latent_caps'
    accepted_cats_path = '../../data/matterport3d/accepted_cats.json'
    metadata_path = '../../data/matterport3d/metadata.csv'

    # load the accepted categories
    accepted_cats = load_from_json(accepted_cats_path)

    # load the metadata containing each the name of each object mesh and their category.
    df_metadata = pd.read_csv(metadata_path)

    # filter the metadata to only include train data and objects with accepted category
    df_metadata = df_metadata[df_metadata['split'] == 'train']
    cat_is_accepted = df_metadata['mpcat40'].apply(lambda x: x in accepted_cats)
    df_metadata = df_metadata.loc[cat_is_accepted]

    # create a column containing the file names for latent caps
    df_metadata['latent_caps'] = df_metadata[['room_name', 'objectId']].\
        apply(lambda x: '-'.join([x['room_name'], str(x['objectId'])]) + '.npy', axis=1)

    # map each category to the latent file names.
    cats_to_file_names = map_cats_to_filenames(df_metadata)

    # find the closest shape clusters for each category (cluster) in accepted cats.
    closest_shape_clusters = process_scenes(latent_caps_dir, cats_to_file_names)

    # save the information about the closest shape clusters for each category.
    write_to_json(closest_shape_clusters, data_output)


if __name__ == '__main__':
    data_output = '../../results/matterport3d/LearningBased/closest_shape_clusters.json'
    main()
