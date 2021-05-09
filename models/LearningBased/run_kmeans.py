import os
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from time import time
from optparse import OptionParser

from scripts.helper import load_from_json, write_to_json
from train_classifier import compute_accuracy


def train_kmeans(latent_caps_dir, latent_caps_filenames, cp_dir, k):
    # collect the latent caps
    latent_caps = [np.load(os.path.join(latent_caps_dir, filename)).reshape(1, -1) for filename in latent_caps_filenames]

    # concat the latent caps to form a matrix
    latent_caps = np.concatenate(latent_caps, axis=0)

    # train kmeans
    kmeans = KMeans(n_clusters=k).fit(latent_caps)

    # save the model
    pickle.dump(kmeans, open(os.path.join(cp_dir, 'kmeans.pkl'), 'wb'))

    return kmeans


def run_kmeans(kmeans, scene_graph_dir, latent_caps_dir, output_dir, mode, accepted_cats):
    # create the output directory if it doesn't already exist.
    scene_graph_dir_out = os.path.join(output_dir, mode)
    if not os.path.exists(scene_graph_dir_out):
        os.makedirs(scene_graph_dir_out)

    # load the scene graphs in val or test dataset
    graph_names = os.listdir(os.path.join(scene_graph_dir, mode))
    for graph_name in graph_names:
        graph = load_from_json(os.path.join(scene_graph_dir, mode, graph_name))
        # assign a cluster id to each object in the scene.
        for obj in graph.keys():
            if graph[obj]['category'][0] in accepted_cats:
                # load the latent caps for the object
                obj_file_name = graph[obj]['file_name'].split('.')[0] + '.npy'
                latent_caps = np.load(os.path.join(latent_caps_dir, obj_file_name)).reshape(1, -1)

                # assign the cluster id.
                graph[obj]['cluster_id'] = int(kmeans.predict(latent_caps)[0])

        # save the graph with predicted categories.
        write_to_json(graph, os.path.join(scene_graph_dir_out, graph_name))


def get_args():
    parser = OptionParser()
    parser.add_option('--train_kmeans', dest='train_kmeans', default=False)
    parser.add_option('--mode', dest='mode', default='test', help='val or test')
    parser.add_option('--accepted_cats_path', dest='accepted_cats_path',
                      default='../../data/matterport3d/accepted_cats.json')
    parser.add_option('--metadata_path', dest='metadata_path',
                      default='../../data/matterport3d/metadata.csv')
    parser.add_option('--scene_graph_dir', dest='scene_graph_dir',
                      default='../../results/matterport3d/LearningBased/scene_graphs_cl')
    parser.add_option('--latent_caps_dir', dest='latent_caps_dir',
                      default='../../../3D-point-capsule-networks/dataset/matterport3d/latent_caps')
    parser.add_option('--cp_dir', dest='cp_dir',
                      default='../../results/matterport3d/LearningBased/kmeans_cat_prediction')
    parser.add_option('--output_dir', dest='output_dir',
                      default='../../results/matterport3d/LearningBased/scene_graphs_cl_with_predictions_kmeans')

    (options, args) = parser.parse_args()
    return options


def main():
    t0 = time()
    # get the arguments
    args = get_args()

    # prepare the accepted categories for training.
    accepted_cats = load_from_json(args.accepted_cats_path)
    accepted_cats = sorted(accepted_cats)

    # extract the file name for object meshes in training data.
    df = pd.read_csv(args.metadata_path)
    df = df[df['split'] == 'train']
    accepted_record = df['mpcat40'].apply(lambda x: x in accepted_cats)
    df = df[accepted_record]
    df['latent_caps_filename'] = df[['room_name', 'objectId']].\
        apply(lambda x: '-'.join([x['room_name'], str(x['objectId'])]) + '.npy', axis=1)
    latent_caps_filenames = df['latent_caps_filename'].tolist()

    # create a directory for checkpoints
    if not os.path.exists(args.cp_dir):
        os.makedirs(args.cp_dir)

    # train kmeans on the training objects
    if args.train_kmeans:
        train_kmeans(args.latent_caps_dir, latent_caps_filenames, args.cp_dir, k=len(accepted_cats))

    # load the model
    kmeans = pickle.load(open(os.path.join(args.cp_dir, 'kmeans.pkl'), 'rb'))

    # apply kmeans to test or validation data
    run_kmeans(kmeans=kmeans, scene_graph_dir=args.scene_graph_dir, latent_caps_dir=args.latent_caps_dir,
               output_dir=args.output_dir, mode=args.mode, accepted_cats=accepted_cats)
    duration_all = (time() - t0) / 60
    print('Processing all scenes took {} minutes'.format(round(duration_all, 2)))


if __name__ == '__main__':
    main()

