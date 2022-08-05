import os
import sys
from time import time
import numpy as np

from scripts.helper import load_from_json, render_single_scene, create_img_table, visualize_scene


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
    if render_all_imgs:
        # make the rendering path if it doesn't exist
        if not os.path.exists(all_imgs_rendering_path):
            os.makedirs(all_imgs_rendering_path)

        # render each test scene
        chunk_size = int(np.ceil(len(scene_names) / num_chunks))
        for scene_name in scene_names[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size]:
            img_name = scene_name.split('.')[0] + '.png'
            if img_name not in visited:
                # load the scene
                graph = load_from_json(os.path.join(scene_dir, scene_name))
                # render the image
                path = os.path.join(all_imgs_rendering_path, scene_name.split('.')[0]+'.png')
                colormap = load_from_json('../data/matterport3d/color_map.json')
                render_single_scene(graph=graph, objects=graph.keys(), highlighted_object=[], path=path,
                                    model_dir=model_dir, colormap=colormap, with_height_offset=False,
                                    rendering_kwargs=rendering_kwargs)
                visited.add(img_name)

    if img_table_all:
        # load the accepted cats
        accepted_cats = set(load_from_json(accepted_cats_path))

        # read img names and build captions.
        imgs = os.listdir(all_imgs_rendering_path)
        captions = []
        for img in imgs:
            # load the scene and maps its categories to the objects.
            scene_name = img.split('.')[0] + '.json'
            graph = load_from_json(os.path.join(scene_dir, scene_name))
            cat_to_objects = map_cat_to_objects(graph, accepted_cats)

            caption = ''
            for cat, objects in cat_to_objects.items():
                caption += '{}: {} <br />\n'.format(cat, objects)
            captions.append(caption)
        create_img_table(all_imgs_rendering_path, 'imgs', imgs, captions=captions, html_file_name='img_table.html',
                         topk=len(imgs), ncols=3)

    if vis_scene:
        scene_name = 'yqstnuAEVhm_room12.json'
        visualize_scene(scene_dir, model_dir, scene_name, highlighted_objects=['22'], with_backbone=True,
                        as_obbox=False)

    if render_query_imgs:
        # make the rendering path if it doesn't exist
        if not os.path.exists(query_rendering_path):
            os.makedirs(query_rendering_path)

        # load the query dict
        query_dict = load_from_json(os.path.join(query_dict_path))

        # render each query scene
        for query_name, query_info in query_dict.items():
            # load the query scene
            query_scene_name = query_info['example']['scene_name']
            graph = load_from_json(os.path.join(scene_dir, query_scene_name))

            # determine objects that are faded
            q = query_info['example']['query']
            query_and_context = set(query_info['example']['context_objects'] + [q])
            faded_nodes = [obj for obj in graph.keys() if obj not in query_and_context]

            # render the image
            path = os.path.join(query_rendering_path, query_name+'.png')
            colormap = load_from_json('../data/matterport3d/color_map.json')
            render_single_scene(graph=graph, objects=graph.keys(), highlighted_object=[q], faded_nodes=faded_nodes,
                                path=path, model_dir=model_dir, colormap=colormap, with_height_offset=True)

    if img_table_queries:
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
            graph = load_from_json(os.path.join(scene_dir, scene_name))

            # maps the query and context objects to their category.
            obj_to_cat = {obj: graph[obj]['category'][0] for obj in context_objects}

            caption = '<br />\n{}: {} <br />\n'.format(q, graph[q]['category'][0])
            for obj, cat in obj_to_cat.items():
                caption += '{}: {} <br />\n'.format(obj, cat)
            captions.append(caption)
        create_img_table(query_rendering_path, 'imgs', imgs, captions=captions, html_file_name='img_table.html',
                         topk=len(imgs), ncols=3)


if __name__ == '__main__':
    rendering_kwargs = {'fov': np.pi / 4, 'light_directional_intensity': 0.01, 'light_point_intensity_center': 0.0,
                        'wall_thickness': 8}
    mode = 'test'
    accepted_cats_path = '../data/matterport3d/accepted_cats_top10.json'
    model_dir = '../data/matterport3d/models'
    scene_dir = '../data/matterport3d/scenes/{}'.format(mode)
    query_dict_path = '../queries/matterport3d/{}/query_dict_objects_top10.json'.format(mode)
    scene_names = os.listdir(scene_dir)
    all_imgs_rendering_path = '../data/matterport3d/{}_scene_imgs/imgs'.format(mode)
    query_rendering_path = '../queries/matterport3d/{}/query_imgs_objects_top10/imgs'.format(mode)

    render_all_imgs = False
    img_table_all = False
    vis_scene = False
    render_query_imgs = False
    img_table_queries = False

    visited = set(os.listdir(all_imgs_rendering_path))
    t = time()
    if len(sys.argv) == 1:
        main(1, 0, scene_names)
    else:
        # parallel -j5 "python3 -u build_queries.py {1} {2}" ::: 5 ::: 0 1 2 3 4
        main(int(sys.argv[1]), int(sys.argv[2]), scene_names)

    duration = (time() - t) / 60
    print('Rendering took {} minutes'.format(np.round(duration, 2)))
