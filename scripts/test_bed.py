import os
import trimesh

from scripts.helper import load_from_json, render_single_scene

# define the paths
scene_graph_dir = '../data/matterport3d/scene_graphs/all'
models_dir = '../data/matterport3d/models'
colormap_path = '../data/matterport3d/color_map.json'
scene_names = ['1pXnuDYAj8r_room2.json', '1pXnuDYAj8r_room17.json', 'TbHJrupSAjP_room11.json']
categorized = False

# define params
colormap = load_from_json(colormap_path)
bad_cats = {'window', 'board_panel', 'ceiling', 'unlabeled', 'wall', 'misc', 'void', 'stairs', 'floor', 'column',
            'beam', 'door', 'counter', 'railing'}
default_color = '#9932CC'
for cat, color in colormap.items():
    if cat not in bad_cats:
        colormap[cat] = default_color

# load the graph and package the faded objects
for scene_name in scene_names:
    graph = load_from_json(os.path.join(scene_graph_dir, scene_name))
    faded_nodes = [n for n in graph.keys() if graph[n]['category'][0] in bad_cats]

    # render the scene with bounding box
    if categorized:
        rendering_path = '{}'.format(scene_name.split('.')[0] + '_categorized' + '.png')
    else:
        rendering_path = '{}'.format(scene_name.split('.')[0] + '.png')
    render_single_scene(graph, graph.keys(), [], rendering_path, models_dir, colormap, resolution=(512, 512),
                        faded_nodes=faded_nodes, with_bounding_box=False)
