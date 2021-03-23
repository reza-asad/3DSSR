import os
import numpy as np
from sklearn.cluster import KMeans

from scripts.helper import load_from_json, visualize_scene


# define the paths
scene_graph_dir = '../data/matterport3d/scene_graphs/val'
models_dir = '../data/matterport3d/models'
embedding_dir = '../../3D-point-capsule-networks/dataset/matterport3d/latent_caps'

# read the accepted categories
accepted_cats = load_from_json('../data/matterport3d/accepted_cats.json')

# load the graphs and count the unique number of object categories
graph_names = ['Z6MFQCViBuw_room0.json', 'EU6Fwq7SyZv_room3.json']
graphs = []
unique_cats = set()
embeddings = []
cats = []
objects = []
scene_names = []
for graph_name in graph_names:
    # load the graph
    graph = load_from_json(os.path.join(scene_graph_dir, graph_name))
    graphs.append(graph)

    for node, node_info in graph.items():
        # only if category is accepted
        cat = node_info['category'][0]
        if cat in accepted_cats:
            # collect the embedding for each object in the graph
            embedding = np.load(os.path.join(embedding_dir, node_info['file_name'].split('.')[0] + '.npy'))
            embeddings.append(embedding.reshape(1, -1))

            # collect the category
            unique_cats.add(cat)
            cats.append(cat)

            # collect the graph and scene name
            objects.append(node)
            scene_names.append(graph_name)

# combine the embeddings into a matrix
embeddings = np.concatenate(embeddings, axis=0)

# apply Kmeans to the set of embeddings.
k = len(unique_cats)
kmeans = KMeans(n_clusters=k).fit(embeddings)
pred_clusters = kmeans.predict(embeddings)

# print the categories in each cluster
for i in range(k):
    # if i != 1:
    #     continue
    print('Results for cluster {}: '.format(i))
    cluster_cats = []
    for j in range(len(cats)):
        if pred_clusters[j] == i:
            cluster_cats.append(cats[j])
            if cats[j] == 'chest_of_drawers':
                visualize_scene(scene_graph_dir, models_dir, scene_names[j], accepted_cats,
                                highlighted_objects=[objects[j]], with_backbone=False)

    print(cluster_cats)
    print('*' * 50)



