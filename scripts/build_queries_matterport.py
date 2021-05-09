import os
import sys
from time import time
import pandas as pd
import numpy as np
import json

from scripts.helper import load_from_json, write_to_json, render_single_scene, create_img_table, visualize_scene

bad_cats = {'window', 'board_panel', 'ceiling', 'unlabeled', 'wall', 'misc', 'void', 'stairs', 'floor', 'column',
            'beam', 'door', 'counter', 'railing'}


def build_accepted_cats(cats_csv_path, output_path):
    df = pd.read_csv(cats_csv_path, delimiter='\t')
    cats = set(df['mpcat40'])
    cats = cats.difference(bad_cats)
    write_to_json(list(cats), output_path)


def map_cat_to_frequency(df, accepted_cats, output_path):
    # filter to only include training objects.
    df = df[df['split'] == 'train']

    # filter to only include objects with accepted categories.
    cat_is_accepted = df['mpcat40'].apply(lambda x: x in accepted_cats)
    df = df.loc[cat_is_accepted]

    # map each cat to its frequency.
    groups_by_size = df.groupby(['mpcat40']).size()
    cat_to_frequency = dict(zip(groups_by_size.keys(), groups_by_size.values))
    cat_to_frequency = {cat: int(freq) for cat, freq in cat_to_frequency.items()}

    # save the results
    write_to_json(cat_to_frequency, output_path)


def map_cat_to_objects(graph, accepted_cats):
    cat_to_objects = {}
    for node, node_info in graph.items():
        cat = node_info['category'][0]
        if cat in accepted_cats:
            if cat not in cat_to_objects:
                cat_to_objects[cat] = [node]
            else:
                cat_to_objects[cat].append(node)

    return cat_to_objects


def main(num_chunks, chunk_idx, scene_names):
    # define the path and parameters.
    category_mapping_path = '../data/matterport3d/category_mapping.tsv'
    metadata_path = '../data/matterport3d/metadata.csv'
    accepted_cats_path = '../data/matterport3d/accepted_cats.json'
    cat_to_frequency_path = '../data/matterport3d/accepted_cats_to_frequency.json'
    model_dir = '../data/matterport3d/models'
    find_cat_frequencies = False
    render_all_imgs = False
    img_table_all = False
    vis_scene = False
    render_query_imgs = False
    img_table_queries = True

    if find_cat_frequencies:
        # save accepted cats if they are not already computed
        if accepted_cats_path.split('/')[-1] not in os.listdir('../data/matterport3d'):
            build_accepted_cats(category_mapping_path, accepted_cats_path)

        # load the accepted cats
        accepted_cats = load_from_json(accepted_cats_path)

        # find the frequency of each category in the training data
        df = pd.read_csv(metadata_path)
        map_cat_to_frequency(df, accepted_cats, cat_to_frequency_path)

    if render_all_imgs:
        rendering_path = '../data/matterport3d/{}_scene_imgs/imgs'.format(mode)
        # make the rendering path if it doesn't exist
        if not os.path.exists(rendering_path):
            os.makedirs(rendering_path)

        # render each test scene
        chunk_size = int(np.ceil(len(scene_names) / num_chunks))
        for scene_name in scene_names[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size]:
            if scene_name not in visited:
                # load the scene
                graph = load_from_json(os.path.join(scene_graph_dir, scene_name))
                # render the image
                path = os.path.join(rendering_path, scene_name.split('.')[0]+'.png')
                colormap = load_from_json('../data/matterport3d/color_map.json')
                render_single_scene(graph=graph, objects=graph.keys(), highlighted_object=[], path=path,
                                    model_dir=model_dir, colormap=colormap)
                visited.add(scene_name)

    if img_table_all:
        rendering_path = '../data/matterport3d/{}_scene_imgs/imgs'.format(mode)
        # load the accepted cats
        accepted_cats = set(load_from_json(accepted_cats_path))

        # read img names and build captions.
        imgs = os.listdir(rendering_path)
        captions = []
        for img in imgs:
            # load the scene and maps its categories to the objects.
            scene_name = img.split('.')[0] + '.json'
            graph = load_from_json(os.path.join(scene_graph_dir, scene_name))
            cat_to_objects = map_cat_to_objects(graph, accepted_cats)

            caption = ''
            for cat, objects in cat_to_objects.items():
                caption += '{}: {} <br />\n'.format(cat, objects)
            captions.append(caption)
        create_img_table(rendering_path, 'imgs', imgs, captions=captions, html_file_name='img_table.html',
                         topk=len(imgs), ncols=3)

    if vis_scene:
        scene_name = 'wc2JMjhGNzB_room28.json'
        visualize_scene(scene_graph_dir, model_dir, scene_name, highlighted_objects=['12', '18', '7'], with_backbone=True,
                        as_obbox=False)

    if render_query_imgs:
        # define paths
        rendering_path = '../results/matterport3d/{}_query_imgs/imgs'.format(mode)
        query_dict_path = '../queries/matterport3d/query_dict_test.json'

        # make the rendering path if it doesn't exist
        if not os.path.exists(rendering_path):
            os.makedirs(rendering_path)

        # load the query dict
        query_dict = load_from_json(os.path.join(query_dict_path))

        # render each query scene
        for query_name, query_info in query_dict.items():
            # load the query scene
            query_scene_name = query_info['example']['scene_name']
            graph = load_from_json(os.path.join(scene_graph_dir, query_scene_name))

            # determine objects that are faded
            q = query_info['example']['query']
            query_and_context = set(query_info['example']['context_objects'] + [q])
            faded_nodes = [obj for obj in graph.keys() if obj not in query_and_context]

            # render the image
            path = os.path.join(rendering_path, query_name+'.png')
            colormap = load_from_json('../data/matterport3d/color_map.json')
            render_single_scene(graph=graph, objects=graph.keys(), highlighted_object=[q], faded_nodes=faded_nodes,
                                path=path, model_dir=model_dir, colormap=colormap)

    if img_table_queries:
        # define paths
        rendering_path = '../results/matterport3d/{}_query_imgs/imgs'.format(mode)
        query_dict_path = '../queries/matterport3d/query_dict_test.json'

        # load the query dict
        query_dict = load_from_json(os.path.join(query_dict_path))

        # read img names and build captions.
        imgs = []
        captions = []
        for query_name, query_info in query_dict.items():
            # take the image
            imgs.append(query_name + '.png')

            # load the query scene.
            scene_name = query_info['example']['scene_name']
            q = query_info['example']['query']
            context_objects = query_info['example']['context_objects']
            graph = load_from_json(os.path.join(scene_graph_dir, scene_name))

            # maps the query and context objects to their category.
            obj_to_cat = {obj: graph[obj]['category'][0] for obj in context_objects}

            caption = '<br />\n{}: {} <br />\n'.format(q, graph[q]['category'][0])
            for obj, cat in obj_to_cat.items():
                caption += '{}: {} <br />\n'.format(obj, cat)
            captions.append(caption)
        create_img_table(rendering_path, 'imgs', imgs, captions=captions, html_file_name='img_table.html',
                         topk=len(imgs), ncols=3)


if __name__ == '__main__':
    mode = 'test'
    scene_graph_dir = '../data/matterport3d/scene_graphs/{}'.format(mode)
    scene_names = os.listdir(scene_graph_dir)
    visited = set()
    t = time()
    if len(sys.argv) == 1:
        main(1, 0, scene_names)
    else:
        # To run in parallel you can use the command:
        # export PYTHONPATH="${PYTHONPATH}:/home/reza/Documents/research/3DSSR"
        # parallel -j5 "python3 -u build_queries_matterport.py main {1} {2}" ::: 5 ::: 0 1 2 3 4
        main(int(sys.argv[2]), int(sys.argv[3]), scene_names)
    duration = (time() - t) / 60
    print('Rendering took {} minutes'.format(np.round(duration, 2)))
