import os
from time import time
import numpy as np
import pandas as pd
from collections import Counter

from helper import load_from_json, write_to_json


def map_cats_to_scene_objects(df):
    cat_to_scene_objects = {}
    cats = df['mpcat40'].values
    scene_objects = df[['room_name', 'objectId']].apply(lambda x: '-'.join([x['room_name'], str(x['objectId'])]), axis=1).values
    for i, cat in enumerate(cats):
        if cat not in cat_to_scene_objects:
            cat_to_scene_objects[cat] = [scene_objects[i]]
        else:
            cat_to_scene_objects[cat].append(scene_objects[i])
    return cat_to_scene_objects


def map_cat_to_objects(scene):
    cat_to_objects = {}
    for obj, node_info in scene.items():
        cat = node_info['category'][0]
        if cat not in cat_to_objects:
            cat_to_objects[cat] = [obj]
        else:
            cat_to_objects[cat].append(obj)
    return cat_to_objects


def Random(query_info, model_names, scene_graph_dir, topk):
    # shuffle the objects in all scenes
    np.random.shuffle(model_names)

    # for each object sample context objects randomly from the object's scene
    num_subscenes = 0
    target_subscenes = []
    num_context_objects = len(query_info['example']['context_objects'])
    train_scenes = set(os.listdir(os.path.join(scene_graph_dir, 'train')))
    for model_name in model_names:
        target = model_name.split('-')[1].split('.')[0]
        scene_name = model_name.split('-')[0] + '.json'
        # only add subscenes using the training data
        if scene_name in train_scenes and num_subscenes < topk:
            num_subscenes += 1
            scene = load_from_json(os.path.join(scene_graph_dir, 'train', scene_name))
            all_except_target = [obj for obj in scene.keys() if obj != target]
            if num_context_objects <= len(all_except_target):
                sample_context = np.random.choice(all_except_target, num_context_objects, replace=False).tolist()
            else:
                sample_context = all_except_target
            subscene = {'scene_name': scene_name, 'target': target, 'context_objects': sample_context}
            target_subscenes.append(subscene)
    return target_subscenes


def CategoryRandom(query_info, query_mode, scene_graph_dir, cat_to_scene_objects, topk):
    # load the query scene and extract the category of the objects
    query_scene_name = query_info['example']['scene_name']
    query = query_info['example']['query']
    context_objects = query_info['example']['context_objects']
    query_scene = load_from_json(os.path.join(scene_graph_dir, query_mode, query_scene_name))
    query_cat = query_scene[query]['category'][0]
    context_cats = [query_scene[obj]['category'][0] for obj in context_objects]
    context_cat_to_freq = Counter(context_cats)

    # for each target object that has the same category as query cat sample context objects
    target_objects = cat_to_scene_objects[query_cat]
    target_subscenes = []
    for target_object in target_objects:
        scene_name = target_object.split('-')[0] + '.json'
        target_object = target_object.split('-')[-1]
        context_objects = []
        target_subscene = {'scene_name': scene_name, 'target': target_object, 'context_objects': context_objects,
                           'context_match': 0}

        # load the scene and map its cats to the objects
        scene = load_from_json(os.path.join(scene_graph_dir, 'train', scene_name))
        cat_to_objects = map_cat_to_objects(scene)

        # for each context object in the query scene sample an object with the same category from the current scene.
        for cat, freq in context_cat_to_freq.items():
            if cat in cat_to_objects:
                if freq <= len(cat_to_objects[cat]):
                    sample = np.random.choice(cat_to_objects[cat], freq, replace=False).tolist()
                else:
                    sample = cat_to_objects[cat]
                context_objects += sample
        target_subscene['context_objects'] = context_objects
        target_subscene['context_match'] = len(context_objects)
        target_subscenes.append(target_subscene)

    # sort the target subscenes based on the number of matching context objects
    target_subscenes = sorted(target_subscenes, reverse=True, key=lambda x: x['context_match'])[:topk]
    return target_subscenes


def main():
    # define initial parameters
    mode = 'val'
    model_name = 'CategoryRandom'
    query_dict_input_path = '../../queries/matterport3d/query_dict_{}.json'.format(mode)
    query_dict_output_path = '../../results/matterport3d/{}/query_dict_{}.json'.format(model_name, mode)
    scene_graph_dir = '../../data/matterport3d/scene_graphs'
    cat_to_scene_objects = {}
    topk = 200

    # read the query dict and the metadata
    query_dict = load_from_json(query_dict_input_path)
    df_metadata = pd.read_csv('../../data/matterport3d/metadata.csv')
    df_metadata = df_metadata[df_metadata['split'] == 'train']

    # map the categories to scene objects if necessary
    if model_name == 'Random':
        model_names = df_metadata[['room_name', 'objectId']].apply(lambda x: '-'.join([x['room_name'],
                                                                                       str(x['objectId'])]), axis=1)
        model_names = model_names.values
        for query, query_info in query_dict.items():
            target_subscenes = Random(query_info, model_names, scene_graph_dir, topk)
            query_info['target_subscenes'] = target_subscenes

    elif model_name == 'CategoryRandom':
        cat_to_scene_objects = map_cats_to_scene_objects(df_metadata)
        for query, query_info in query_dict.items():
            target_subscenes = CategoryRandom(query_info, mode, scene_graph_dir, cat_to_scene_objects, topk)
            query_info['target_subscenes'] = target_subscenes

    else:
        raise Exception('Model not defined')

    # save the changes to query dict
    write_to_json(query_dict, query_dict_output_path)


if __name__ == '__main__':
    t = time()
    main()
    duration = time() - t
    print('Evaluation took {} minutes'.format(round(duration / 60, 2)))
