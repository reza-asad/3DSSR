import os
import numpy as np
import torch
from sklearn.cluster import KMeans
from collections import Counter
from matplotlib import pyplot as plt

from scripts.helper import write_to_json
from extract_rank_subscenes import map_file_names_to_idx


def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
    return sse


def kmeans_embeddings(features, file_names_to_idx, k_max):
    # find the within cluster sum of squared errors to choose k
    sse = calculate_WSS(features, k_max)
    plt.plot(range(1, len(sse) + 1), sse)
    plt.show()

    # determine the number of clusters using the elbow method.
    diffs = []
    # compute first derivative
    for i in range(1, len(sse)):
        diffs.append(abs(sse[i] - sse[i-1]))
    # compute second derivative
    k = 2
    for i in range(1, len(diffs)):
        if abs(diffs[i] - diffs[i-1]) / diffs[i-1] > 0.8:
            k = i + 3
            break

    # cluster using kmeans
    kmeans = KMeans(n_clusters=k).fit(features)
    pred_clusters = kmeans.predict(features)

    # record the cluster_id for each file name
    file_name_to_cluster = {}
    for file_name, idx in file_names_to_idx.items():
        file_name_to_cluster[file_name] = int(pred_clusters[idx])

    return file_name_to_cluster


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def prepare_embedding_features(args):
    # load the features and file names.
    features_dir = os.path.join(args.cp_dir, args.results_folder_name, args.features_dir_name_thresholding)
    features = torch.load(os.path.join(features_dir, "trainfeat.pth"))
    print('Loaded features of shape {}'.format(features.shape))

    # map the file names to feature indices.
    file_indices = torch.load(os.path.join(features_dir, "test_file_names.pth"))
    file_name_to_idx = map_file_names_to_idx(args, file_indices, args.metadata_path_queries)

    return features, file_name_to_idx


def cluster(args):
    # load features and file_name mappings.
    features, file_name_to_idx = prepare_embedding_features(args)

    # find the thresholds by clustering the similarity scores.
    file_name_to_cluster = kmeans_embeddings(features, file_name_to_idx, args.k_max)

    # save the clustering.
    write_to_json(file_name_to_cluster, os.path.join(args.cp_dir, args.results_folder_name,
                                                     args.predicted_clusters_file_name))


def find_sim_threshold(args):
    # load features and file_name mappings.
    features, file_name_to_idx = prepare_embedding_features(args)

    # sample from the features; Otherwise, it Kmeans will crash.
    sample_size = len(features)
    if sample_size > 13000:
        sample_size = 13000
    random_indices = np.random.choice(range(len(features)), sample_size, replace=False)
    features = features[random_indices, ...]

    # compute similarities between all train features
    sims = torch.matmul(features, features.T)
    sims = sims.view(-1, 1)

    # cluster the similarities into two clusters
    kmeans = KMeans(n_clusters=2).fit(sims)
    centroids = kmeans.cluster_centers_

    # find the thresholds and sort them to send the highest one
    thresholds = [centroids[0, 0], centroids[1, 0]]

    return sorted(thresholds)[1]

