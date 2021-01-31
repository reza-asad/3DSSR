import os
import shutil
import struct
import numpy as np
import json
import pandas as pd
from queue import Queue
from matplotlib import pyplot as plt
import trimesh
from PIL import Image
import networkx as nx
from graphviz import Source
import gc
from collections import Counter
from scipy import stats

from obj_3d import Mesh


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


def add_bidirectional_edges(graph_dir):
    graph_names = os.listdir(graph_dir)
    for graph_name in graph_names:
        graph_path = os.path.join(graph_dir, graph_name)
        # read graph
        graph = load_from_json(graph_path)

        # add edges in the other direction except for the parent relation
        old_graph = graph.copy()
        for n1, prop in old_graph.items():
            for n2, relations in prop['neighbours'].items():
                for relation in relations:
                    if relation != 'parent':
                        # if relation != 'my_parent':
                        if n1 not in old_graph[n2]['neighbours']:
                            graph[n2]['neighbours'][n1] = []
                        if relation not in graph[n2]['neighbours'][n1]:
                            # if relation == 'parent':
                            #     graph[n2]['neighbours'][n1].append('my_parent')
                            # else:
                            graph[n2]['neighbours'][n1].append(relation)
        # write the resulting graph
        write_to_json(graph, graph_path)


def find_scene_graph_stats(graph_dir):
    graph_names = os.listdir(graph_dir)
    graph_stats = {}
    num_unique_relations = 0
    # count all the relations excet parent
    for graph_name in graph_names:
        graph = load_from_json(os.path.join(graph_dir, graph_name))
        for _, prop in graph.items():
            for nb, relations in prop['neighbours'].items():
                num_unique_relations += 1
                for relation in relations:
                    if relation == 'parent':
                        continue
                    if relation not in graph_stats:
                        graph_stats[relation] = 1
                    else:
                        graph_stats[relation] += 1
    print('There are {} unique relations'.format(num_unique_relations))
    print(graph_stats)


def extract_direction_metadata(csv_path, data_dir, obj_to_front_filename, obj_to_up_filename):
    def extract_direction(x, default_dir=None):
        if pd.isna(x):
            return default_dir
        else:
            str_dir = x.split('\\')
            float_dir = [float(e.replace(',', '')) for e in str_dir]
            return float_dir

    df = pd.read_csv(csv_path)
    all_obj_files = df['fullId'].apply(lambda x: x.split('.')[-1] + '.obj').tolist()
    all_front_directions = df['front'].apply(lambda x: extract_direction(x, default_dir=[0, -1, 0]))\
        .tolist()
    all_up_directions = df['up'].apply(lambda x: extract_direction(x, default_dir=[0, 0, 1])).tolist()

    obj_to_front = dict(zip(all_obj_files, all_front_directions))
    obj_to_up = dict(zip(all_obj_files, all_up_directions))
    write_to_json(obj_to_front, os.path.join(data_dir, obj_to_front_filename))
    write_to_json(obj_to_up, os.path.join(data_dir, obj_to_up_filename))


def prepare_scene_with_texture(objects, graph, models_dir_with_textures, models_dir, query_objects=[], room_key='0'):
    scene = []
    if room_key not in objects:
        objects.append(room_key)

    for obj in objects:
        if obj in query_objects:
            model_path = os.path.join(models_dir, graph[obj]['file_name'])
        else:
            model_path = os.path.join(models_dir_with_textures, graph[obj]['file_name'])
        mesh_obj = Mesh(model_path)
        transform = np.asarray(graph[obj]['transform']).reshape(4, 4).transpose()
        mesh = mesh_obj.load(transform)
        if obj in query_objects:
            # radii = np.linalg.norm(mesh.vertices - mesh.center_mass, axis=1)
            # mesh.visual.vertex_colors = trimesh.visual.linear_color_map(radii)
            mesh.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#00eb00")
        # elif obj == room_key:
        #     mesh.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#aec7e8")
        scene.append(mesh)
        # Extract camera_pose and room dimensions
        if obj == room_key:
            if type(mesh) == trimesh.base.Trimesh:
                mesh = mesh.scene()
            camera_pose, _ = mesh.graph[mesh.camera.name]
            room_dimension = mesh.extents

    return trimesh.Scene(scene), camera_pose, room_dimension


def prepare_scene_with_color(objects, graph, models_dir, query_objects=[], faded_nodes=[],
                             ceiling_cats=['ceiling', 'roof'], colormap=None):
    # if len(query_objects) > 0:
    #     query_objects = query_objects[0]
    default_color = '#aec7e8'
    if colormap is None:
        colormap = load_from_json('data/example_based/colour_map.json')
    scene = []
    for obj in objects:
        # find the category of the object
        cat = ''
        if len(graph[obj]['category']) > 0:
            cat = graph[obj]['category'][0]

        # if the mesh is a ceiling mesh skip
        if cat in ceiling_cats:
            continue

        # load the mesh
        model_path = os.path.join(models_dir, graph[obj]['file_name'])
        mesh_obj = Mesh(model_path)
        transform = np.asarray(graph[obj]['transform']).reshape(4, 4).transpose()
        mesh = mesh_obj.load(transform)
        mesh.visual.vertex_colors = trimesh.visual.color.hex_to_rgba(default_color)

        # find color based on category
        if cat in colormap:
            mesh.visual.vertex_colors = trimesh.visual.color.hex_to_rgba(colormap[cat])

        # highlight node if its important
        if obj in query_objects:
            mesh.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#0000ff")

        # face color if object is not important
        if obj in faded_nodes:
            mesh.visual.vertex_colors = trimesh.visual.color.hex_to_rgba(default_color)

        scene.append(mesh)

    del mesh
    gc.collect()

    # extract the room dimention and the camera pose
    scene = trimesh.Scene(scene)
    room_dimension = scene.extents
    camera_pose, _ = scene.graph[scene.camera.name]
    return scene, camera_pose, room_dimension


def find_parent_of(graph_dir, category, unique_only=False):
    parent_info = {}
    parent_of = {}
    graph_names = os.listdir(graph_dir)
    for graph_name in graph_names:
        graph = load_from_json(os.path.join(graph_dir, graph_name))
        found_cat = False
        for curr_node, prop in graph.items():
            # Find the mapping from child to parent
            for nb, relations in prop['neighbours'].items():
                for relation in relations:
                    if relation == 'parent' and len(prop['category']) > 0:
                        parent_of[nb] = ['-'.join(prop['category']), curr_node]
            # If the current node is the desired category add its parent to the parent count
            if category.lower() in [cat.lower() for cat in prop['category']]:
                if found_cat and unique_only:
                    print('{} non unique'.format(category))
                    return
                parent_cat, target = parent_of[curr_node]
                if parent_cat in parent_info:
                    parent_info[parent_cat]['count'] += 1
                    parent_info[parent_cat]['scenes'].append(graph_name)
                    parent_info[parent_cat]['source-target'].append((curr_node, target))
                else:
                    parent_info[parent_cat] = {'count': 1, 'scenes': [graph_name], 'source-target': [(curr_node,
                                                                                                      target)]}
                found_cat = True
    parent_count = json.dumps(parent_info, indent=4)
    print(parent_count)


def build_category_dict(csv_path):
    """
    This creates a dictionary that maps each obj file to its category.
    :param csv_path: The path to the csv which contains the labels.
    :return: obj_to_category: The dictionary.
    """
    # build a dictionary that maps each model to its category and sub-category
    def find_clean_categories(x):
        def package_cats(raw_cats):
            result_cats = []
            for e in raw_cats:
                if e[0] != '_':
                    e = e.replace(' ', '').lower()
                    result_cats.append(e)
            return result_cats

        category = x[0]
        wnlemmas = x[1]
        # First try to extract tags from the category column
        hierarchical_cats = [] if pd.isna(category) else category.split(',')
        clean_categories = package_cats(hierarchical_cats)

        # if you didn't find any tag based on the category column try the wnlemmas column.
        if len(clean_categories) == 0:
            hierarchical_cats = [] if pd.isna(wnlemmas) else wnlemmas.split(',')
            clean_categories = package_cats(hierarchical_cats)
        return clean_categories

    df = pd.read_csv(csv_path)
    obj_files = df['fullId'].apply(lambda x: x.split('.')[-1] + '.obj').tolist()
    category_hierarchy = df[['category', 'wnlemmas']].apply(find_clean_categories, axis=1).tolist()
    obj_to_category = dict(zip(obj_files, category_hierarchy))
    return obj_to_category


def one_hot_encoding(graph_dir, data_dir, default_cat="", primary=False):
    cat_to_idx = {default_cat: 0}
    idx = 1
    graph_names = os.listdir(graph_dir)
    # extract all categories and assign an index to each unique category
    for graph_name in graph_names:
        # load the graph
        graph = load_from_json(os.path.join(graph_dir, graph_name))
        # add unique categories to the dictionary
        for _, node_prop in graph.items():
            cats = node_prop['category']
            if primary and len(cats) > 0:
                cats = [cats[0]]
            for cat in cats:
                if cat not in cat_to_idx:
                    cat_to_idx[cat] = idx
                    idx += 1

    # build the one hot encoding based on the cat_to_idx dictionary
    encoding = {}
    enc_length = len(cat_to_idx)
    for graph_name in graph_names:
        graph = load_from_json(os.path.join(graph_dir, graph_name))
        for _, node_prop in graph.items():
            file_name = node_prop['file_name']
            cats = node_prop['category']
            code = np.zeros(enc_length, dtype=int)
            if len(cats) == 0:
                idx = cat_to_idx[default_cat]
                code[idx] = 1
            elif primary:
                cats = [cats[0]]
            for cat in cats:
                idx = cat_to_idx[cat]
                code[idx] = 1
            if file_name not in encoding:
                encoding[file_name] = code.tolist()
            else:
                assert np.all(code == encoding[file_name])

    # save the one hot vectors
    if primary:
        output_path = os.path.join(data_dir, 'cat_to_vec_primary.json')
    else:
        output_path = os.path.join(data_dir, 'cat_to_vec.json')
    write_to_json(encoding, output_path)


def find_co_occurrences(graph_dir):
    def update_observations(g, seen):
        for n1, n1_prop in g.items():
            if len(n1_prop['category']) > 0:
                cat1 = n1_prop['category'][0]
            else:
                cat1 = n1_prop['file_name']
            if cat1 not in seen:
                seen[cat1] = {'num_found': 0, 'objects': {}}
            for n2, n2_prop in g.items():
                if len(n2_prop['category']) > 0:
                    cat2 = n2_prop['category'][0]
                else:
                    cat2 = n2_prop['file_name']
                if cat1 != cat2:
                    if cat2 not in seen[cat1]['objects']:
                        seen[cat1]['objects'][cat2] = 1
                    else:
                        seen[cat1]['objects'][cat2] += 1
                    seen[cat1]['num_found'] += 1
                elif n1 != n2:
                    seen[cat1]['num_found'] += 1

    observation_summary = {}
    graph_names = os.listdir(graph_dir)
    # update the occurrence of objects and pair of objects for each graph
    for graph_name in graph_names:
        graph = load_from_json(os.path.join(graph_dir, graph_name))
        update_observations(graph, observation_summary)

    # compute the co_occurrences by iterating over the computed summary
    co_occurrences = {}
    for cat1, info in observation_summary.items():
        for cat2, num_co_occurrence in info['objects'].items():
            key = '-'.join([cat1, cat2])
            co_occurrences[key] = num_co_occurrence / info['num_found']
    return co_occurrences


def standardize_features(graph_dir, intrinsic_feature_types, extrinsic_feature_types):
    graph_names = os.listdir(graph_dir)
    all_features = {k: [] for k in intrinsic_feature_types + extrinsic_feature_types}
    for graph_name in graph_names:
        graph = load_from_json(os.path.join(graph_dir, graph_name))
        for feature_type in intrinsic_feature_types:
            flatten_features = []
            for e1 in graph['0'][feature_type]:
                for e2 in e1:
                    flatten_features.append(e2)
            all_features[feature_type] += flatten_features
        for n, node_prop in graph.items():
            if n.isdigit():
                for feature_type in extrinsic_feature_types:
                    flatten_features = []
                    for e1 in node_prop[feature_type]:
                        for e2 in e1:
                            flatten_features.append(e2)
                    all_features[feature_type] += flatten_features

    # compute mean and std using the flatten lists
    for feature_type, features in all_features.items():
        # compute mean and std
        mean = np.mean(features)
        std = np.std(features)
        print(feature_type, ', ', 'mean: ', mean, ', ', 'std: ', std)


def extract_subset_scene_graph(in_path, out_path, subset_feature_types, all_feature_types):
    graph_names = os.listdir(in_path)
    for graph_name in graph_names:
        in_graph = load_from_json(os.path.join(in_path, graph_name))
        out_graph = {}
        for node, node_prop in in_graph.items():
            if not node.isdigit():
                out_graph[node] = node_prop
                continue
            subset_node_prop = {}
            for prop, value in node_prop.items():
                if (prop in subset_feature_types) or (prop not in all_feature_types):
                    subset_node_prop[prop] = value
                else:
                    subset_node_prop[prop] = {}
            out_graph[node] = subset_node_prop
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        write_to_json(out_graph, os.path.join(out_path, graph_name))


def create_train_test(data_dir, test_data):
    # make sure the 3 folders exist
    for folder in ['query', 'database', 'all_data']:
        path = os.path.join(data_dir, folder)
        if not os.path.exists(path):
            os.mkdir(path)

    # clean the test data
    test_data = [e.split('.')[0] for e in test_data]

    # move test graph to test folder and train to train
    file_names = os.listdir(data_dir)
    for file_name in file_names:
        file_name_split = file_name.split('.')
        if file_name not in ['query', 'database', 'all_data']:
            d1 = os.path.join(data_dir, file_name)
            if file_name_split[0] in test_data:
                d2 = os.path.join(data_dir, 'all_data', file_name)
            else:
                d2 = os.path.join(data_dir, 'database', file_name)
            shutil.move(d1, d2)

    # copy all the test graphs to query
    curr_test_files = os.listdir(os.path.join(data_dir, 'all_data'))
    for file_name in curr_test_files:
        d1 = os.path.join(data_dir, 'all_data', file_name)
        d2 = os.path.join(data_dir, 'query', file_name)
        shutil.copyfile(d1, d2)

    # copy all the train graphs to test
    db_data = os.listdir(os.path.join(data_dir, 'database'))
    for file_name in db_data:
        d1 = os.path.join(data_dir, 'database', file_name)
        d2 = os.path.join(data_dir, 'all_data', file_name)
        shutil.copyfile(d1, d2)


def create_adjacency_lists(scene_graph_dir, train='train', test='test'):
    def extract_adj_lists(graph_names, graph_dir):
        for graph_name in graph_names:
            # load the graph
            graph = load_from_json(os.path.join(graph_dir, graph_name))

            # build the adj lists
            senders = []
            recievers = []
            num_nodes = np.asarray(graph['adj']).shape[1]
            for n1, node_prop in graph.items():
                if not n1.isdigit():
                    continue
                for n2, _ in node_prop['neighbours'].items():
                    if int(n2) < int(n1):
                        continue
                    sender_item = [0] * num_nodes
                    sender_item[int(n1)] = 1
                    senders.append(sender_item)

                    reciever_item = [0] * num_nodes
                    reciever_item[int(n2)] = 1
                    recievers.append(reciever_item)

            # save the results
            graph['senders'] = senders
            graph['recievers'] = recievers
            write_to_json(graph, os.path.join(graph_dir, graph_name))

    # extract adj list for train graphs
    train_graph_dir = os.path.join(scene_graph_dir, train)
    train_graphs = os.listdir(train_graph_dir)
    extract_adj_lists(train_graphs, train_graph_dir)

    # extract adj lists for test graphs
    test_graph_dir = os.path.join(scene_graph_dir, test)
    test_graphs = os.listdir(test_graph_dir)
    extract_adj_lists(test_graphs, test_graph_dir)


def create_spatial_labels(scene_graph_dir, train='train', test='test'):
    def extract_spatial_labels(graph_names, graph_dir):
        for graph_name in graph_names:
            graph = load_from_json(os.path.join(graph_dir, graph_name))
            adj = np.asarray(graph['adj'])
            num_nodes = adj.shape[1]

            labels = np.zeros((3, num_nodes, num_nodes))
            for n1 in range(num_nodes):
                # everything in the frame of n1
                for n2 in range(num_nodes):
                    # check left/right (1/0)
                    if graph[str(n1)]['centroid'][n2][0] < 0:
                        labels[0, n1, n2] = 1
                    # check behind/front (1/0)
                    if graph[str(n1)]['centroid'][n2][1] < 0:
                        labels[1, n1, n2] = 1
                    # check below/above (1/0)
                    if graph[str(n1)]['centroid'][n2][2] < 0:
                        labels[2, n1, n2] = 1

            graph['spatial_labels'] = labels.tolist()
            write_to_json(graph, os.path.join(graph_dir, graph_name))

    # extract adj list for train graphs
    train_graph_dir = os.path.join(scene_graph_dir, train)
    train_graphs = os.listdir(train_graph_dir)
    extract_spatial_labels(train_graphs, train_graph_dir)

    # extract adj lists for test graphs
    test_graph_dir = os.path.join(scene_graph_dir, test)
    test_graphs = os.listdir(test_graph_dir)
    extract_spatial_labels(test_graphs, test_graph_dir)


def create_img_grid(file_path, ours='combined', others='graph_kernel'):
    def find_img_name(path, q):
        for img_name in os.listdir(path):
            if q == img_name.split('_')[2]:
                return img_name

    def center_crop(img):
        # center crop
        width, height = img.size
        new_width, new_height = 2400, 360
        width_offset = (width - new_width) / 2
        height_offset = (height - new_height) / 2
        return img.crop((width_offset + 80, height_offset, width - 390 - width_offset, height - height_offset))

    img_path_ours = os.path.join(file_path, ours, 'final')
    img_path_others = os.path.join(file_path, others, 'final')

    idx = 0
    for img_name_ours in os.listdir(img_path_ours):
        query = img_name_ours.split('_')[2]
        img_name_others = find_img_name(img_path_others, query)

        # read the images
        img1 = Image.open(os.path.join(img_path_ours, img_name_ours))
        img2 = Image.open(os.path.join(img_path_others, img_name_others))

        # center crop
        img1 = center_crop(img1)
        img2 = center_crop(img2)

        # resize
        width, height = img1.size
        new_width = 1000
        img1 = img1.resize((new_width, new_width * height // width))
        img2 = img2.resize((new_width, new_width * height // width))

        # concat horizontally
        if idx == 0:
            dst = Image.new('RGB', (img1.width, img1.height * 2 * len(os.listdir(img_path_ours))))
        dst.paste(img1, (0, idx * img1.height))
        dst.paste(img2, (0, (idx+1) * img2.height))
        idx += 2
    # dst.show()
    dst.save('results/qualitative2.png')


def create_query_results_img(img_path):
    def center_crop(img):
        # center crop
        width, height = img.size
        new_width, new_height = 2000, 360
        width_offset = (width - new_width) / 2
        height_offset = (height - new_height) / 2
        return img.crop((width_offset + 60, height_offset, width - width_offset, height - height_offset))

    # read the images
    img = Image.open(os.path.join(img_path))

    # center crop
    img = center_crop(img)

    # resize
    # width, height = img.size
    # new_width = 1000
    # img = img.resize((new_width, new_width * height // width))
    # img.show()
    img.save('results/query_results_img.png')


def find_tag_frequency_cutoff_scene(graph_dir):
    primary_tags = []
    graph_names = os.listdir(graph_dir)
    for graph_name in graph_names:
        graph = load_from_json(os.path.join(graph_dir, graph_name))
        for node, node_info in graph.items():
            if not node.isdigit():
                continue
            cat = ''
            if len(node_info['category']) > 0:
                cat = node_info['category'][0]
            primary_tags.append(cat)

    tag_frequency = Counter(primary_tags)
    tag_frequency = sorted(tag_frequency.items(), reverse=True, key=lambda x: x[1])
    tags, frequencies = list(zip(*tag_frequency))
    tags = list(tags)
    return tags, frequencies


def find_BFS(G_q, q):
    BFS_q = []
    queue = Queue()
    queue.put(q)
    visited = set()
    while not queue.empty():
        curr_node = queue.get()
        if curr_node not in visited:
            # curr_node_cat = '-'.join(G_q[curr_node]['category'])
            curr_node_cat = G_q[curr_node]['category']
            BFS_item = {'node': curr_node, 'category': curr_node_cat, 'neighbours': []}
            ring_info = []
            for neighbour, relations in G_q[curr_node]['neighbours'].items():
                # category = '-'.join(G_q[neighbour]['category'])
                category = G_q[neighbour]['category']
                centroid_curr_node = np.asarray(G_q[curr_node]['centroid'])
                dist_to_source = np.linalg.norm(centroid_curr_node[int(neighbour), :])
                ring_info.append((neighbour, category, relations, dist_to_source))

            # sort the ring nodes based on distance to the source node
            ring_info = sorted(ring_info, key=lambda x: x[3])
            for neighbour, category, relations, _ in ring_info:
                BFS_item['neighbours'].append((neighbour, category, relations))
                queue.put(neighbour)

            visited.add(curr_node)
            BFS_q.append(BFS_item)
    return BFS_q


def add_nodes_and_edges(G, nx_graph, n, add_label_id=False, accepted_cats=None, with_fc=True):
    # nx_graph.add_nodes_from(G.keys())
    for n1, node_info in G.items():
        if not n1.isdigit():
            continue
        color = 'black'
        if n1 == n:
            color = 'red'
        label = node_info['category']
        if len(label) > 0:
            label = label[0:1]
        # if it is required to filter the graph make sure to take the required nodes.
        if accepted_cats is not None and label[0] not in accepted_cats:
            continue
        if add_label_id:
            label += [n]
        # if n1 == n:
        #     nx_graph.add_node(n1, label='-'.join(label), color='#5993E5', style='filled')
        # else:
        nx_graph.add_node(n1, label='-'.join(label), color=color)
        if 'neighbours' in node_info:
            for n2, relations in node_info['neighbours'].items():
                label2 = G[n2]['category']
                if len(label2) > 0:
                    label2 = label2[0:1]
                # if it is required to filter the graph make sure to take the required nodes.
                if accepted_cats is not None and label2[0] not in accepted_cats:
                    continue
                # for relation in relations:
                color = 'black'
                if n1 == n:
                    color = 'red'
                # remove parent relation
                if 'parent' in relations:
                    relations.remove('parent')
                if (not with_fc) and ('fc' in relations):
                    continue
                nx_graph.add_edge(n1, n2, label='-'.join(relations), color=color)
    return nx_graph


def add_nodes_and_edges_colormap(G, nx_graph, colormap, add_label_id=False):
    nx_graph.add_nodes_from(G.keys())
    for n1, node_info in G.items():
        color = "#aec7e8"
        label = node_info['category']
        if len(node_info['category']) > 0:
            cat = node_info['category'][0]
            label = node_info['category'][0:1]
            if cat in colormap:
                color = colormap[cat]
        if add_label_id:
            label += [n1]
        nx_graph.add_node(n1, label='-'.join(label), color=color)
        if 'neighbours' in node_info:
            for n2, relations in node_info['neighbours'].items():
                nx_graph.add_edge(n1, n2, label='-'.join(relations), color=color)
    return nx_graph


def visualize_graph(G, n, path, accepted_cats=[], with_color_map=False, colormap=None, add_label_id=False, with_fc=True):
    if len(accepted_cats) == 0:
        accepted_cats = None
    nx_graph = nx.MultiDiGraph()
    if with_color_map:
        nx_graph = add_nodes_and_edges_colormap(G, nx_graph, colormap, add_label_id)
    else:
        nx_graph = add_nodes_and_edges(G, nx_graph, n, add_label_id, accepted_cats, with_fc=with_fc)
    nx.drawing.nx_pydot.write_dot(nx_graph, path)
    s = Source.from_file(path, format='png')
    s.render(path)


def find_node_induced_subgraph(G, constraint_nodes, filter_attributes):
    G_q = {}
    for node, node_info in G.items():
        # only consider nodes
        if not node.isdigit():
            continue
        if node in constraint_nodes:
            G_q[node] = {}
            # restric the neighbours
            G_q[node]['neighbours'] = {}
            for neighbour, relations in node_info['neighbours'].items():
                if neighbour in constraint_nodes:
                    if 'parent' in relations:
                        relations.remove('parent')
                    G_q[node]['neighbours'][neighbour] = relations

            # restrict the features
            for attribute in G['0'].keys():
                if attribute not in filter_attributes:
                    G_q[node][attribute] = G[node][attribute]
    return G_q
# def find_node_induced_subgraph(G, q, constraint_nodes, filter_attributes):
#     G_q = {}
#     if q in constraint_nodes:
#         constraint_nodes.remove(q)
#
#     # initialize the data for the primary node first.
#     G_q[q] = {}
#     G_q[q]['neighbours'] = {}
#     # restrict the features
#     for attribute in G['0'].keys():
#         if attribute not in filter_attributes:
#             G_q[q][attribute] = G[q][attribute]
#
#     for node, node_info in G.items():
#         # only consider nodes
#         if not node.isdigit():
#             continue
#         if node in constraint_nodes:
#             G_q[node] = {}
#             G_q[node]['neighbours'] = {}
#             # add neighbours between primary node and constraint nodes
#             q_to_node_relation = G[q]['neighbours'][node]
#             if 'parent' in q_to_node_relation:
#                 q_to_node_relation.remove('parent')
#             G_q[q]['neighbours'][node] = q_to_node_relation
#             node_to_q_relation = G[node]['neighbours'][q]
#             if 'parent' in node_to_q_relation:
#                 node_to_q_relation.remove('parent')
#             G_q[node]['neighbours'][q] = node_to_q_relation
#
#             # add neighbours within constraint nodes ignoring fc and parent relations
#             for neighbour, relations in node_info['neighbours'].items():
#                 if neighbour in constraint_nodes:
#                     if 'parent' in relations:
#                         relations.remove('parent')
#                     if 'fc' in relations:
#                         relations.remove('fc')
#                     if len(relations) > 0:
#                         G_q[node]['neighbours'][neighbour] = relations
#
#             # restrict the features
#             for attribute in G['0'].keys():
#                 if attribute not in filter_attributes:
#                     G_q[node][attribute] = G[node][attribute]
#     return G_q


def find_edge_induced_subgraph(G, constraint_edges, filter_attributes):
    G_q = {}
    constraint_edges = list(constraint_edges)
    if len(constraint_edges) == 0:
        return G_q

    if len(constraint_edges) == 1 and constraint_edges[0][0] == constraint_edges[0][1]:
        n = constraint_edges[0][0]
        G_q[n] = {}
        G_q[n]['category'] = G[n]['category']
        return G_q

    if len(constraint_edges) > 0:
        heads, tails = list(zip(*constraint_edges))

    for n1, n2 in constraint_edges:
        if n1 not in G_q:
            G_q[n1] = {}
            G_q[n1]['neighbours'] = {}
        if n2 not in G_q and n2 not in heads:
            G_q[n2] = {}
            G_q[n2]['category'] = G[n2]['category']

        if n1 == n2:
            G_q[n1]['category'] = G[n1]['category']
        else:
            G_q[n1]['neighbours'][n2] = G[n1]['neighbours'][n2]
        # add other attributes
        for attribute in G['0'].keys():
            if attribute not in filter_attributes:
                G_q[n1][attribute] = G[n1][attribute]
    return G_q


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


def find_test_scenes(query_dict_path):
    test_scenes = set()
    query_dict_temp = load_from_json(query_dict_path)
    for _, q_name in query_dict_temp.items():
        test_scenes.add(q_name['example']['scene_name'])
    return test_scenes


def find_test_point_clouds(query_dict_path, scene_dir):
    test_scenes = find_test_scenes(query_dict_path)
    test_point_clouds = set()
    for scene_name in test_scenes:
        scene_name = scene_name.split('.')[0] + '.txt'
        with open(os.path.join(scene_dir, scene_name), 'r') as f:
            lines = f.readlines()
            for line in lines:
                words = line.split()
                if words[0] == 'newModel':
                    file_name = words[-1] + '.npy'
                    test_point_clouds.add(file_name)
    return test_point_clouds


def create_train_test_pc(data_dir, label_to_models, test_ratio=0.1, valid_ratio=0.1):
    # create train, test and valid folders
    for folder in ['train', 'valid', 'test']:
        path = os.path.join(data_dir, folder)
        if not os.path.exists(path):
            os.mkdir(path)

    # for each category shuffle the files and split them into train, test and valid
    for label, file_names in label_to_models.items():
        # first shuffle the file names
        np.random.shuffle(file_names)

        # determine the test, valid and train files
        num_files = len(file_names)
        num_test = int(np.ceil(test_ratio * num_files))
        num_valid = int(np.ceil(valid_ratio * num_files))

        file_dict = {'test': file_names[:num_test],
                     'valid': file_names[num_test: num_test + num_valid],
                     'train': file_names[num_test + num_valid:]
                     }

        # create train, test and valid folders and move the files where they belong
        for folder, file_names_subset in file_dict.items():
            for file_name in file_names_subset:
                d1 = os.path.join(data_dir, file_name)
                d2 = os.path.join(data_dir, folder, file_name)
                shutil.move(d1, d2)


def find_pc_mean_std(pc_dir):
    file_names = os.listdir(pc_dir)
    first_point = np.load(os.path.join(pc_dir, file_names[0]))
    num_dims = first_point.shape[1]
    num_points = first_point.shape[0]
    chunk_size = num_points * num_dims
    sum_ = 0
    sum_squared = 0
    num_chunks = 0
    max_ = 0
    min_ = 0
    for file_name in file_names:
        path = os.path.join(pc_dir, file_name)
        pc = np.load(path)
        num_chunks += 1
        sum_ += np.sum(pc)
        sum_squared += np.sum(pc**2)

        # update the max and in
        curr_max = np.max(pc)
        curr_min = np.min(pc)
        if curr_max > max_:
            max_ = curr_max
        if curr_min < min_:
            min_ = curr_min
    mean = sum_ / (num_chunks * chunk_size)
    mean_squred = sum_squared / (num_chunks * chunk_size)
    std = np.sqrt(mean_squred - mean**2)
    print('Mean: ', mean)
    print('STD: ', std)
    print('Max: ', max_)
    print('Min: ', min_)


def visualize_pc(file_names, gt_dir, reconstruction_dir, with_reoncstruction=True):
    for file_name in file_names:
        # visualzie ground truth pc
        gt_path = os.path.join(gt_dir, file_name)
        pc_gt = np.load(gt_path)
        pc_gt = trimesh.points.PointCloud(pc_gt)
        pc_gt.show()

        # visualize the reconstructed pc
        if with_reoncstruction:
            pred_path = os.path.join(reconstruction_dir, file_name)
            pc_pred = np.load(pred_path)
            pc_pred = trimesh.points.PointCloud(pc_pred)
            pc_pred.show()


def find_model_and_labels_dict(models_dir, obj_to_category_dict, query_cats):
    model_to_label = {}
    label_to_models = {}

    # first find the 3d model for each label
    file_names = os.listdir(models_dir)
    for file_name in file_names:
        file_name_obj = file_name.split('.')[0] + '.obj'
        if file_name_obj in obj_to_category_dict:
            cats = obj_to_category_dict[file_name_obj]
            if len(cats) > 0:
                if cats[0] in query_cats:
                    label = cats[0]
                else:
                    label = 'other'
            else:
                label = 'other'
        else:
            label = 'other'
        if label in label_to_models:
            label_to_models[label].append(file_name)
        else:
            label_to_models[label] = [file_name]
        model_to_label[file_name] = label

    return label_to_models, model_to_label


def find_query_cats(graph_dir, query_dict_path):
    query_dict = load_from_json(query_dict_path)
    cats = set()
    for query, query_info in query_dict.items():
        scene_name = query_info['example']['scene_name']
        constraint_nodes = query_info['example']['constraint_nodes']

        # load the graph
        graph = load_from_json(os.path.join(graph_dir, scene_name))
        for n in constraint_nodes:
            cat = graph[n]['category']
            cats.add(cat[0])
    return list(cats)


def find_closest_latent(latent_dir, output_path):
    closest_latent = {}
    file_names = os.listdir(latent_dir)
    i, j = 0, 1
    while i < len(file_names):
        print(i)
        file_name1 = file_names[i]
        latent1 = np.load(os.path.join(latent_dir, file_name1))
        closest_latent[file_name1] = (None, 1000)
        while j < len(file_names):
            file_name2 = file_names[j]
            latent2 = np.load(os.path.join(latent_dir, file_name2))
            dist = np.linalg.norm(latent1 - latent2)
            if dist < closest_latent[file_name1][1]:
                closest_latent[file_name1] = (file_name2, float(dist))
            j += 1

        i += 1
        j = i + 1
    write_to_json(closest_latent, output_path)


# def find_pairwise_latent_dist(latent_dir, output_path, closest_latent, threshold=0.9):
#     latent_dist_dict = {}
#     file_names = os.listdir(latent_dir)
#     i, j = 0, 0
#     while i < len(file_names):
#         print(i)
#         file_name1 = file_names[i]
#         latent1 = np.load(os.path.join(latent_dir, file_name1))
#         min_dist1 = closest_latent[file_name1][1]
#         while j < len(file_names):
#             file_name2 = file_names[j]
#             latent2 = np.load(os.path.join(latent_dir, file_name2))
#             min_dist2 = closest_latent[file_name2][1]
#             curr_dist = np.linalg.norm(latent1 - latent2)
#             if (curr_dist == 0) or ((min_dist1 / curr_dist) >= threshold) or ((min_dist2 / curr_dist) >= threshold):
#                 key = '-'.join(sorted([file_name1, file_name2]))
#                 latent_dist_dict[key] = float(curr_dist)
#             j += 1
#
#         i += 1
#         j = i
#     write_to_json(latent_dist_dict, output_path)
def find_pairwise_latent_dist(latent_dir, output_path):
    latent_dist_dict = {}
    file_names = os.listdir(latent_dir)

    # read all the files once
    file_name_to_latent = {}
    for file_name in file_names:
        latent = np.load(os.path.join(latent_dir, file_name))
        file_name_to_latent[file_name] = latent

    i, j = 0, 1
    while i < len(file_names):
        print('Iteration {}/{}'.format(i, len(file_names)))
        file_name1 = file_names[i]
        latent1 = file_name_to_latent[file_name1]
        while j < len(file_names):
            file_name2 = file_names[j]
            latent2 = file_name_to_latent[file_name2]
            curr_dist = np.linalg.norm(latent1 - latent2)
            key = '-'.join(sorted([file_name1, file_name2]))
            latent_dist_dict[key] = float(curr_dist)
            j += 1

        i += 1
        j = i + 1
    write_to_json(latent_dist_dict, output_path)


def build_category_dict_matterport(csv_path, output_path):
    df = pd.read_csv(csv_path)
    df['key'] = df.apply(lambda x: '-'.join([x['room_name'], str(x['objectId']) + '.ply']), axis=1)
    df['mpcat40'] = df['mpcat40'].apply(lambda x: [x])
    obj_to_category_dict = dict(zip(df['key'], df['mpcat40']))
    write_to_json(obj_to_category_dict, output_path)


def find_tag_frequency_cutoff_models(models_dir, graph_dir, query_dict_path, csv_path, output_path, topk=40):
    query_cats = find_query_cats(graph_dir=graph_dir,
                                 query_dict_path=query_dict_path)
    query_cats.remove('room')
    # find all primary tags
    obj_to_category = build_category_dict(csv_path)
    primary_tags = []
    for model_name in os.listdir(models_dir):
        model_name = model_name.split('.')[0] + '.obj'
        if model_name in obj_to_category:
            cats = obj_to_category[model_name]
            if len(cats) > 0:
                primary_tags.append(cats[0])

    # find the frequency of each primary tag
    tag_frequency = Counter(primary_tags)
    tag_frequency = sorted(tag_frequency.items(), reverse=True, key=lambda x: x[1])
    tags, frequencies = list(zip(*tag_frequency))
    tags = list(tags)

    # make sure to include query categories
    filtered_tags = tags[:topk]
    missing_query_tags = set(query_cats).difference(set(filtered_tags))
    missing_query_tags = list(missing_query_tags)
    i = 0
    while len(missing_query_tags) > 0:
        if filtered_tags[-i-1] not in query_cats:
            filtered_tags[-i-1] = missing_query_tags.pop()
        else:
            i += 1

    print(filtered_tags)
    write_to_json(filtered_tags, output_path)


def filter_node_info(node_info, test_relation):
    filtered_node_info = {}
    for attribute, values in node_info.items():
        if attribute != 'neighbours':
            filtered_node_info[attribute] = values
        else:
            filtered_node_info[attribute] = {}
            for neighbour, relations in node_info[attribute].items():
                for relation in relations:
                    if relation == test_relation:
                        filtered_node_info[attribute][neighbour] = [relation]
                        break
    return filtered_node_info


def find_dims(bbox, local_frame, canonical_frame=None):
    if canonical_frame is None:
        canonical_frame = np.asarray([[-1, 0, 0],
                                      [0, -1, 0],
                                      [0, 0, 1]])

    rot = np.dot(canonical_frame, np.linalg.inv(local_frame))
    rot_bbox = np.dot(rot, bbox.transpose())
    extent = np.abs(rot_bbox[:, 1] - rot_bbox[:, 0])
    return extent


def map_model_to_label_and_size(csv_path, models_dir, model_to_label, obj_to_front, obj_to_up,
                                model_to_scale_example_based, output_path):
    # create a mapping from model name to dims
    df_metadata = pd.read_csv(csv_path)
    df_metadata['model_name'] = df_metadata['fullId'].apply(lambda x: x.split('.')[1] + '.npy')
    df_metadata['dims'] = df_metadata['aligned.dims'].apply(lambda x: x.split('\,'))
    df_metadata['dims'] = df_metadata['dims'].apply(lambda x: np.asarray([float(e) for e in x]))
    model_to_dims = dict(zip(df_metadata['model_name'], df_metadata['dims']))

    num_files = len(model_to_label)
    not_found = 0
    missing_models = set()
    for i, (model_name, label) in enumerate(model_to_label.items()):
        print('Iteration {} / {}'.format(i, num_files))
        if model_name in model_to_dims:
            dims = model_to_dims[model_name]
        else:
            not_found += 1
            missing_models.add(model_name)
            # find the canonical dims yourself
            model_path = os.path.join(models_dir, model_name.split('.')[0] + '.obj')
            mesh_obj = Mesh(model_path=model_path, obj_to_front=obj_to_front, obj_to_up=obj_to_up,
                            default_front=np.asarray([0, -1, 0]), default_up=np.asarray([0, 0, 1]))
            local_frame = mesh_obj.compute_coordinate_frame()

            mesh = mesh_obj.load()
            bbox = mesh.bounding_box.bounds.copy()
            our_dims = find_dims(bbox, local_frame)
            # find the scale
            scale = model_to_scale_example_based[model_name]
            our_dims *= scale
            # correct the orders to match the xzy format in shapenetsem_metadata
            dims = [0, 0, 0]
            dims[0] = our_dims[0]
            dims[1] = our_dims[2]
            dims[2] = our_dims[1]

        model_to_label[model_name] = [label, list(dims)]
    write_to_json(model_to_label, output_path)


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


def size_match(file_name1, file_name2, model_to_dims, threshold=0.67):
    _, dims1 = model_to_dims[file_name1]
    _, dims2 = model_to_dims[file_name2]
    for i in range(3):
        if (abs(dims1[i] - dims2[i]) / (dims2[i] + 0.000001)) > threshold:
            return False
    return True


def find_query_freq():
    query_dict_template = load_from_json('results/example_based/ideal_ranking/query_dict_template.json')
    result = []
    for query, query_info in query_dict_template.items():
        result.append(query_info['example']['scene_name'])
    for k, v in Counter(result).items():
        print(k, v)
    print(len(result))
    print(len(set(result)))


def find_subscene_to_superscene(graph_dir):
    def find_subset_superset(g1, g2, g1_name, g2_name):
        l1, l2 = len(g1), len(g2)
        if l1 > l2:
            g1, g2 = g2, g1
            g1_name, g2_name = g2_name, g1_name
        g2_file_names = [node_info['file_name'] for _, node_info in g2.items()]
        g2_file_names = set(g2_file_names)
        g1_file_names = [node_info['file_name'] for _, node_info in g1.items()]
        g1_file_names = set(g1_file_names)

        # find the amount of intersection between the two scenes
        if len(g1_file_names.intersection(g2_file_names)) / len(g1_file_names) >= 0.9:
            return g1_name, g2_name
        return None, None

    subscene_to_superscene = {}
    graph_names = os.listdir(graph_dir)
    i, j = 0, 1
    while i < len(graph_names):
        graph1 = load_from_json(os.path.join(graph_dir, graph_names[i]))
        while j < len(graph_names):
            graph2 = load_from_json(os.path.join(graph_dir, graph_names[j]))
            subset, superset = find_subset_superset(graph1, graph2, graph_names[i], graph_names[j])
            if subset is not None:
                if subset not in subscene_to_superscene:
                    subscene_to_superscene[subset] = [superset]
                else:
                    subscene_to_superscene[subset].append(superset)
            j += 1
        i += 1
        j = i + 1
    # make sure that this is a one-to-one map
    for subset, supersets in subscene_to_superscene.items():
        if len(supersets) > 0:
            graph_sizes = []
            for superset in supersets:
                graph = load_from_json(os.path.join(graph_dir, superset))
                graph_sizes.append(len(graph))
            argmax = np.argmax(np.asarray(graph_sizes))
            subscene_to_superscene[subset] = supersets[argmax:argmax+1]
    return subscene_to_superscene


def main():
    graph_dir = 'data/example_based/scene_graphs'
    voxel_dir = 'data/example_based/voxels'
    model_dir = 'data/example_based/models'
    model_dir_with_textures = 'data/example_based/models_with_texture'
    data_dir = 'data'
    obj_to_front_filename = 'obj_to_front.json'
    obj_to_up_filename = 'obj_to_up.json'

    if process_voxels:
        command = './ZernikeMoments-master/examples/zernike3d {}/{} 20'
        compute_descriptors(voxel_dir, command)
        nth_closest_dict = nth_closest_descriptor(voxel_dir, n=100)
        # save the nth_closest dict
        write_to_json(nth_closest_dict, os.path.join(voxel_dir, 'nth_closest_obj.josn'))

        data = load_from_json(os.path.join(voxel_dir, 'nth_closest_obj.josn'))
        for k, v in data.items():
            if pd.isna(v[1]):
                print(k, v)

    if process_graphs:
        add_bidirectional_edges(graph_dir)

    if find_stats:
        find_scene_graph_stats(graph_dir)

    if extract_orientation:
        extract_direction_metadata(csv_path='data/example_based/shapenetsem_metadata.csv',
                                   data_dir=data_dir,
                                   obj_to_front_filename=obj_to_front_filename,
                                   obj_to_up_filename=obj_to_up_filename)

    if visualize:
        graph_name = 'scene00122.json'
        graph_path = os.path.join(graph_dir, graph_name)
        graph = load_from_json(graph_path)
        scene, _, _ = prepare_scene_with_texture(graph.keys(), graph, models_dir_with_textures=model_dir_with_textures,
                                                 models_dir=model_dir, query_objects=['22'])
        scene.show()

    if find_parent:
        find_parent_of(graph_dir, 'mirror', unique_only=False)

    if category_to_one_hot:
        one_hot_encoding(graph_dir, data_dir, primary=True)

    if compute_co_occurrence:
        co_occurrences = find_co_occurrences(graph_dir)
        write_to_json(co_occurrences, 'data/example_based/co_occurences.json')

    if feature_standardization:
        standardize_features('data/example_based/scene_graphs_zernike_tag_obbox_cent_undirected/train',
                             intrinsic_feature_types=['zernike', 'dims'],
                             extrinsic_feature_types=['obbox', 'centroid'])

    if extract_subset_graph:
        subgraphs_feature_types = {
            'scene_graphs_tag_dir': ['tag'],
            'scene_graphs_zernike_dir': ['zernike'],
            'scene_graphs_tagzernike_dir': ['tag', 'zernike'],
            'scene_graphs_dims_dir': ['dims'],
            'scene_graphs_dimscent_dir': ['dims', 'centroid'],
            'scene_graphs_obbox_dir': ['obbox', 'dims'],
            'scene_graphs_tagobbox_dir': ['tag', 'obbox', 'dims'],
            'scene_graphs_obboxcent_dir': ['obbox', 'dims', 'centroid'],
            'scene_graphs_obboxcentzernike_dir': ['obbox', 'dims', 'centroid', 'zernike'],
            'scene_graphs_obboxcenttag_dir': ['obbox', 'dims', 'centroid', 'tag'],
        }
        for subgraph, feature_type in subgraphs_feature_types.items():
            print('processing {}'.format(subgraph))
            extract_subset_scene_graph('data/example_based/scene_graphs_zernike_tag_obbox_cent_dir/test',
                                       'data/{}'.format(subgraph),
                                       subset_feature_types=feature_type,
                                       all_feature_types=['tag', 'zernike', 'obbox', 'dims', 'centroid'])

    if add_categories:
        # build the category dict
        category_dict = build_category_dict(csv_path='data/example_based/shapenetsem_metadata.csv')
        # load the scene graphs and add their category
        graph_names = os.listdir('data/example_based/scene_graphs')
        for graph_name in graph_names:
            # load the graph
            graph = load_from_json(os.path.join('data/example_basedscene_graphs', graph_name))
            for _, node_prop in graph.items():
                file_name = node_prop['file_name']
                if file_name in category_dict:
                    node_prop['category'] = category_dict[file_name]
            # save the graph
            write_to_json(graph, os.path.join('data/example_based/scene_graphs', graph_name))

    if split_train_test_scenes:
        # find all test(query) scenes
        test_data = find_test_scenes(query_dict_path='results/example_based/ideal_ranking/query_dict_template.json')
        create_train_test('data/example_based/scene_graphs_views', test_data)

    if create_adj_lists:
        create_adjacency_lists('data/example_based/scene_graphs_zernike_tag_obbox_cent_dir')

    if add_spatial_labels:
        create_spatial_labels('data/example_based/scene_graphs_zernike_tag_obbox_cent_undir')

    if img_grid:
        create_img_grid('results/top_k_results')

    if query_results_img:
        create_query_results_img('results/top4_query_PottedPlant on top of Room_dgi_combined_undir.png')

    if tag_frequency:
        find_tag_frequency_cutoff_scene('data/example_based/scene_graphs_zernike_tag_obbox_cent_dir/test', 'data/example_based/colour_map.json')

    if split_train_test_pc:
        label_to_models = load_from_json('data/example_based/label_to_models.json')
        create_train_test_pc('data/example_based/shape_embedding/latent_caps', label_to_models, test_ratio=0,
                             valid_ratio=0.2)

    if pc_mean_std:
        find_pc_mean_std('data/example_based/point_clouds/train')

    if pc_visualization:
        file_names = ['1bb8e7439a411cd234f718b37d8150d2.npy']
        # file_names = ['test.npy']
        visualize_pc(file_names=file_names,
                     gt_dir='data/example_based/point_clouds/train',
                     reconstruction_dir='data/checkpoints/shape_embedding/reconstruction',
                     with_reoncstruction=False)

    if find_all_mesh_labels:
        query_cats = find_query_cats('data/example_based/scene_graphs',
                                     'results/example_based/ideal_ranking/query_dict_template.json')
        obj_to_category_dict = build_category_dict('data/example_based/shapenetsem_metadata.csv')
        label_to_models, model_to_label = find_model_and_labels_dict(models_dir='data/example_based/point_clouds',
                                                                     obj_to_category_dict=obj_to_category_dict,
                                                                     query_cats=query_cats)
        write_to_json(model_to_label, 'data/example_based/model_to_label_test.json')
        write_to_json(label_to_models, 'data/example_based/label_to_models_test.json')


if __name__ == '__main__':
    process_voxels = False
    process_graphs = False
    find_stats = False
    extract_orientation = False
    visualize = True
    find_parent = False
    category_to_one_hot = False
    compute_co_occurrence = False
    feature_standardization = False
    extract_subset_graph = False
    add_categories = False
    split_train_test_scenes = False
    create_adj_lists = False
    add_spatial_labels = False
    img_grid = False
    query_results_img = False
    tag_frequency = False
    img_table = False
    split_train_test_pc = False
    pc_mean_std = False
    pc_visualization = False
    find_all_mesh_labels = False

    main()
