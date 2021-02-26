import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

from helper import load_from_json, write_to_json, render_single_scene, create_img_table

bad_cats = {'window', 'board_panel', 'ceiling', 'unlabeled', 'wall', 'misc', 'void', 'stairs', 'floor', 'column',
            'beam', 'door', 'counter', 'railing'}


def save_accepted_cats(cats_csv_path, output_path):
    df = pd.read_csv(cats_csv_path, delimiter='\t')
    cats = set(df['mpcat40'])
    cats = cats.difference(bad_cats)
    write_to_json(list(cats), output_path)


def map_cat_to_scenes(scene_graph_dir, mode, output_path):
    cat_to_scenes = {}
    graph_names = os.listdir(os.path.join(scene_graph_dir, mode))
    for graph_name in graph_names:
        # load the graph
        graph = load_from_json(os.path.join(scene_graph_dir, mode, graph_name))

        # map cat to scenes
        for node, node_info in graph.items():
            cat = node_info['category'][0]
            if cat not in bad_cats:
                key = graph_name.split('.')[0] + '-' + node
                if cat not in cat_to_scenes:
                    cat_to_scenes[cat] = [key]
                else:
                    cat_to_scenes[cat].append(key)

    # save the cat to scenes map
    write_to_json(cat_to_scenes, output_path)


def map_cat_to_frequencies(cat_to_scene_objects, output_path):
    # map each category to its frequency
    total_occurences = sum([len(scenes) for _, scenes in cat_to_scene_objects.items()])
    cat_to_frequency = {cat: len(scenes)/total_occurences for cat, scenes in cat_to_scene_objects.items()}

    # save the results
    write_to_json(cat_to_frequency, output_path)


def plot_wss(x, kmax):
    sse = []
    for k in range(1, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(x)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(x)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(x)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (x[i][0] - curr_center[0]) ** 2

        sse.append(curr_sse)
    plt.plot(sse)
    plt.show()


def cluster_frequencies(cat_to_frequency, find_k=True, k=None):
    cats = list(cat_to_frequency.keys())
    frequencies = list(cat_to_frequency.values())
    frequencies = [[f] for f in frequencies]

    if find_k or k is None:
        plot_wss(frequencies, kmax=5)
    else:
        # apply kmeans to the frequencies
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(frequencies)
        clusters = kmeans.predict(frequencies)
        cat_to_cluster = dict(zip(cats, clusters))
        return cat_to_cluster


def sample_query_object(cat_to_scene_objects, cat_to_cluster, num_queries, ratio, query_dict):
    # map each cluster to object instances
    cluster_to_scene_objects = {}
    for cat, cluster in cat_to_cluster.items():
        if cluster not in cluster_to_scene_objects:
            cluster_to_scene_objects[cluster] = cat_to_scene_objects[cat]
        else:
            cluster_to_scene_objects[cluster] += cat_to_scene_objects[cat]

    # sample from each cluster according to the ratio
    probs = np.random.uniform(0, 1, num_queries)
    num_prev_samples = 0
    sampled_query_objects = []
    for cluster, cluster_prob in enumerate(ratio):
        num_samples = sum(probs <= cluster_prob) - num_prev_samples
        sampled_query_objects += np.random.choice(cluster_to_scene_objects[cluster], num_samples, replace=False).tolist()
        num_prev_samples = num_samples

    # map each scene object to its category
    scene_object_to_cat = {}
    for cat, scene_objects in cat_to_scene_objects.items():
        for scene_object in scene_objects:
            scene_object_to_cat[scene_object] = cat

    # build the query dict for sampled query objects
    for i, scene_object in enumerate(sampled_query_objects):
        scene_name, query = scene_object.split('-')
        cat = scene_object_to_cat[scene_object]
        key = cat + '-{}'.format(i)
        query_dict[key] = {'example': {}}
        query_dict[key]['example'] = {}
        query_dict[key]['example']['scene_name'] = scene_name + '.json'
        query_dict[key]['example']['query'] = query
        query_dict[key]['example']['context_objects'] = []


def find_co_occurrences(scene_graph_dir, mode, accepted_cats, output_path):
    def update_co_occurrences(cats_count):
        for c1 in cats_count.keys():
            for c2, count in cats_count.items():
                co_occurrences[c1][c2] += count
            # subtract the self
            co_occurrences[c1][c1] -= 1

    # initialize the co-occurrences
    co_occurrences = {}
    for cat in accepted_cats:
        co_occurrences[cat] = {c: 0 for c in accepted_cats}

    # for each scene add to the co_occurrences
    graph_names = os.listdir(os.path.join(scene_graph_dir, mode))
    for graph_name in graph_names:
        graph = load_from_json(os.path.join(scene_graph_dir, mode, graph_name))
        # filter the objects to accepted categories
        scene_cats = [node_info['category'][0] for _, node_info in graph.items() if node_info['category'][0] in accepted_cats]
        scene_cats_count = Counter(scene_cats)

        # update the co-occurrences
        update_co_occurrences(scene_cats_count)

    # normalize the co-occurrences to probability
    for q, context_freq in co_occurrences.items():
        total_freq = sum(context_freq.values())
        co_occurrences[q] = {context_object: frequency/total_freq for context_object, frequency in context_freq.items()}

    # save the co-occurrences
    write_to_json(co_occurrences, output_path)


def sample_context_objects(scene_graph_dir, mode, query_dict, co_occurrences, cat_to_scene_objects, ratio):
    # for each query object cluster context objects into two groups
    for _, query_info in query_dict.items():
        q = query_info['example']['query']
        scene_name = query_info['example']['scene_name']
        graph = load_from_json(os.path.join(scene_graph_dir, mode, scene_name))
        q_cat = graph[q]['category'][0]

        # cluster frequencies into two groups
        context_cats = list(co_occurrences[q_cat].keys())
        frequencies = co_occurrences[q_cat].values()
        frequencies = [[f] for f in frequencies]
        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(frequencies)
        clusters = kmeans.predict(frequencies)

        # map each cluster to the object instances in the current scene
        cluster_to_objects = {}
        for i, cluster in enumerate(clusters):
            scene_objects = cat_to_scene_objects[context_cats[i]]
            curr_scene_objects = [e for e in scene_objects if e.split('-')[0] == scene_name.split('.')[0]]
            if cluster not in cluster_to_objects:
                cluster_to_objects[cluster] = curr_scene_objects
            else:
                cluster_to_objects[cluster] += curr_scene_objects

        # sample from a uniform distribution to determine the number of context objects
        num_objects = 0
        for objects in cluster_to_objects.values():
            num_objects += len(objects)
        mu = 0.5
        std = 0.3
        num_context_objects = int(np.random.normal(mu, std) * num_objects)
        while num_context_objects <= 0:
            num_context_objects = int(np.random.normal(mu, std))

        # sample according to a split probability from each cluster
        probs = np.random.uniform(0, 1, num_context_objects)
        num_prev_samples = 0
        sampled_context_objects = []
        scene_name_q = '-'.join([scene_name.split('.')[0], q])
        for cluster, cluster_prob in enumerate(ratio):
            num_samples = np.maximum(sum(probs < cluster_prob) - num_prev_samples, 1)
            cluster_objects = cluster_to_objects[cluster]
            # remove the query node from the context objects
            if scene_name_q in cluster_objects:
                cluster_objects.remove(scene_name_q)

            # make sure you don't take more samples than there are objects in the cluster
            if num_samples > len(cluster_objects):
                num_samples = len(cluster_objects)
            sampled_context_objects += np.random.choice(cluster_objects, num_samples, replace=False).tolist()
            num_prev_samples = num_samples

        # print the size of the context objects sampled compared to all of the accepted objects in each scene
        print(len(sampled_context_objects), len(cluster_to_objects[0]) + len(cluster_to_objects[1]))

        # populate the query dict
        query_info['example']['context_objects'] = [e.split('-')[-1] for e in sampled_context_objects]


def main():
    # initialize the query dict and its path
    num_queries = 50
    mode = 'val'
    query_dict = {}
    query_dict_path = '../queries/matterport3d/query_dict_{}.json'.format(mode)
    scene_graph_dir = '../data/matterport3d/scene_graphs'
    rendering_path = '../results/matterport3d/val_query_imgs/imgs'
    sample_queries = False
    sample_contexts = False
    render = True
    with_img_table = True

    if sample_queries:
        # save accepted cats
        save_accepted_cats('../data/matterport3d/category_mapping.tsv', '../data/matterport3d/accepted_cats.json')

        # map each category to the scenes it belongs to
        map_cat_to_scenes(scene_graph_dir, mode, '../data/matterport3d/cat_to_scene_objects_{}.json'.format(mode))

        # map cats to frequencies
        cat_to_scene_objects = load_from_json('../data/matterport3d/cat_to_scene_objects_{}.json'.format(mode))
        map_cat_to_frequencies(cat_to_scene_objects, '../data/matterport3d/cat_to_frequencies_{}.json'.format(mode))

        # cluster the cat frequencies
        cat_to_frequency = load_from_json('../data/matterport3d/cat_to_frequencies_{}.json'.format(mode))
        cat_to_cluster = cluster_frequencies(cat_to_frequency, find_k=False, k=2)

        # sample query nodes.
        ratio = [1/3, 1]
        cat_to_scene_objects = load_from_json('../data/matterport3d/cat_to_scene_objects_{}.json'.format(mode))
        sample_query_object(cat_to_scene_objects, cat_to_cluster, num_queries, ratio, query_dict)
        write_to_json(query_dict, query_dict_path)

    if sample_contexts:
        # find the co-occurrences for each query and other objects using all scenes
        accepted_cats = load_from_json('../data/matterport3d/accepted_cats.json')
        find_co_occurrences(scene_graph_dir, 'all', accepted_cats, '../data/matterport3d/co-occurrences_{}.json'.format(mode))

        # map each query node randomly to a scene and sample context objects
        query_dict = load_from_json(query_dict_path)
        co_occurrences = load_from_json('../data/matterport3d/co-occurrences_{}.json'.format(mode))
        cat_to_scene_objects = load_from_json('../data/matterport3d/cat_to_scene_objects_{}.json'.format(mode))
        ratio = [1/3, 1]
        sample_context_objects(scene_graph_dir, mode, query_dict, co_occurrences, cat_to_scene_objects, ratio)
        write_to_json(query_dict, query_dict_path)

    if render:
        # make the rendering path if it doesn't exist
        if not os.path.exists(rendering_path):
            os.makedirs(rendering_path)

        # for each query render its img
        query_dict = load_from_json(query_dict_path)
        for query, query_info in query_dict.items():
            scene_name = query_info['example']['scene_name']
            graph = load_from_json(os.path.join(scene_graph_dir, mode, scene_name))
            q = query_info['example']['query']
            q_context = set(query_info['example']['context_objects'] + [q])

            # render the image
            faded_nodes = [obj for obj in graph.keys() if obj not in q_context]
            model_dir = '../data/matterport3d/models'
            path = os.path.join(rendering_path, query+'.png')
            colormap = load_from_json('../data/matterport3d/color_map.json')
            render_single_scene(graph=graph, objects=graph.keys(), highlighted_object=[q], faded_nodes=faded_nodes,
                                path=path, model_dir=model_dir, colormap=colormap)

    if with_img_table:
        imgs = os.listdir(rendering_path)
        create_img_table(rendering_path, 'imgs', imgs, html_file_name='img_table.html', topk=len(imgs), ncols=3)


if __name__ == '__main__':
    main()
