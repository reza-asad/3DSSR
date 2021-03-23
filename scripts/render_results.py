import os
import sys
import numpy as np
from time import time
from matplotlib import pyplot as plt
from PIL import Image

from scripts.helper import load_from_json, render_single_scene, create_img_table


def make_rendering_folders(query_results, rendering_path):
    if not os.path.exists(rendering_path):
        os.makedirs(rendering_path)
    for query, resutls in query_results.items():
        query_path = os.path.join(rendering_path, query, 'imgs')
        if not os.path.exists(query_path):
            os.makedirs(query_path)


def render_model_results(query_results, scene_graph_dir, models_dir, rendering_path, colormap, num_chunks, chunk_size,
                         chunk_idx, mode, topk=50):
    # visualize the result of each query
    for i, (query, results) in enumerate(query_results.items()):
        if query not in visited:
            # render the query scene once only
            if chunk_idx == i % num_chunks:
                scene_name = results['example']['scene_name']
                query_graph = load_from_json(os.path.join(scene_graph_dir, mode, scene_name))
                q = results['example']['query']
                q_context = set(results['example']['context_objects'] + [q])

                # render the image
                faded_nodes = [obj for obj in query_graph.keys() if obj not in q_context]
                path = os.path.join(rendering_path, query, 'imgs', 'query_{}_{}.png'.format(scene_name.split('.')[0],
                                                                                            q))
                render_single_scene(graph=query_graph, objects=query_graph.keys(), highlighted_object=[q],
                                    faded_nodes=faded_nodes, path=path, model_dir=models_dir, colormap=colormap)

            # render the topk results from the model
            top_results_chunk = results['target_subscenes'][: topk][chunk_idx * chunk_size:
                                                                    (chunk_idx + 1) * chunk_size]

            for j, target_subscene in enumerate(top_results_chunk):
                target_scene_name = target_subscene['scene_name']
                target_graph_path = os.path.join(scene_graph_dir, 'all', target_scene_name)
                target_graph = load_from_json(target_graph_path)
                t = target_subscene['target']
                highlighted_objects = [t]
                t_context = set(list(target_subscene['correspondence'].keys()) + [t])

                # render the image
                faded_nodes = [obj for obj in target_graph.keys() if obj not in t_context]
                path = os.path.join(rendering_path, query, 'imgs', 'top_{}_{}_{}.png'
                                    .format(chunk_idx * chunk_size + j + 1, target_scene_name.split('.')[0], t))
                render_single_scene(graph=target_graph, objects=target_graph.keys(),
                                    highlighted_object=highlighted_objects, faded_nodes=faded_nodes, path=path,
                                    model_dir=models_dir, colormap=colormap)

            visited.add(query)


def plot_evaluations(x, y, fig, ax, label, output_path):
    ax.plot(x, y, label=label)
    plt.title("Distance and Overlap mAPs")
    plt.xlabel("Thresholds")
    plt.ylabel('mAP')
    leg = ax.legend()


def main(num_chunks, chunk_idx):
    # define initial parameters
    make_folders = False
    render = False
    with_img_table = True
    filter_queries = ['sink-34', 'chair-26', 'chest_of_drawers-25', 'table-17', 'sofa-39']
    mode = 'val'
    model_name = 'LearningBased'
    experiment_name = 'lstm'
    topk = 50
    query_results_path = '../results/matterport3d/{}/query_dict_{}_{}_evaluated.json'.format(model_name, mode,
                                                                                             experiment_name)
    scene_graph_dir = '../data/matterport3d/scene_graphs'
    rendering_path = '../results/matterport3d/{}/rendered_results/{}'.format(model_name, experiment_name)
    models_dir = '../data/matterport3d/models'
    colormap = load_from_json('../data/matterport3d/color_map.json')
    caption_keys = {'distance_mAP', 'distance_precision', 'overlap_mAP', 'overlap_precision', 'overlap_rotation', 'theta'}

    # load the query results and filter it if necessary
    query_results = load_from_json(query_results_path)
    filtered_query_results = {}
    if filter_queries[0] != 'all':
        for query, results in query_results.items():
            if query in filter_queries:
                filtered_query_results[query] = results
    else:
        filtered_query_results = query_results

    if make_folders:
        # make rending folders
        make_rendering_folders(filtered_query_results, rendering_path)

    if render:
        # render results from the model
        chunk_size = int(np.ceil(topk / num_chunks))
        render_model_results(filtered_query_results, scene_graph_dir, models_dir, rendering_path, colormap, num_chunks,
                             chunk_size, chunk_idx, mode, topk)

    if with_img_table:
        for query, results in filtered_query_results.items():
            # plot the distance and overlap mAP for the query
            evaluation_plot_name = 'evaluation.png'
            evaluation_plot_path = os.path.join(rendering_path, query, 'imgs', evaluation_plot_name)
            fig, ax = plt.subplots()
            for metric in ['distance_mAP', 'overlap_mAP']:
                x, y = list(zip(*results[metric]['mAP']))
                plot_evaluations(x, y, fig, ax, metric, evaluation_plot_path)
            plt.savefig(evaluation_plot_path)

            # read images in the img directory and find the query img
            imgs_path = os.path.join(rendering_path, query, 'imgs')
            imgs = os.listdir(imgs_path)
            query_img = [e for e in imgs if 'query' in e]

            # sort the topk images
            imgs.remove(query_img[0])
            imgs.remove(evaluation_plot_name)
            ranks = [int(img_name.split('_')[1]) for img_name in imgs]
            ranks_imgs = zip(ranks, imgs)
            ranks_imgs = sorted(ranks_imgs, key=lambda x: x[0])
            sorted_imgs = list(list(zip(*ranks_imgs))[1])

            # add captions for the top10 results
            captions = []
            target_subscenes = results['target_subscenes']
            topk = 10
            num_objects = len(results['example']['context_objects']) + 1
            for i in range(len(sorted_imgs)):
                caption = '<br />\n'
                # add number of context matches
                if i < topk:
                    caption += 'num_objects: {} <br />\n'.format(target_subscenes[i]['context_match'] + 1)
                for key, value in target_subscenes[i].items():
                    if key in caption_keys:
                        if 'precision' in key:
                            caption_value = '{}/{}'.format(int(value * num_objects),
                                                           num_objects)
                        elif 'theta' in key:
                            caption_value = np.round(value * 180 / np.pi, 0)
                        elif 'overlap_rotation' in key:
                            caption_value = np.round(value * 180 / np.pi, 0)
                        else:
                            caption_value = value
                        caption += ' {}: {} <br />\n'.format(key, caption_value)
                captions.append(caption)

            create_img_table(imgs_path, 'imgs', sorted_imgs, with_query_scene=True, query_img=query_img[0],
                             evaluation_plot=evaluation_plot_name, html_file_name='img_table.html', topk=len(imgs),
                             ncols=3, captions=captions)


if __name__ == '__main__':
    visited = set()
    t = time()
    if len(sys.argv) == 1:
        main(1, 0)
    else:
        # To run in parallel you can use the command:
        # export PYTHONPATH="${PYTHONPATH}:/home/reza/Documents/research/3DSSR"
        # parallel -j5 "python3 -u render_results.py main {1} {2}" ::: 5 ::: 0 1 2 3 4
        main(int(sys.argv[2]), int(sys.argv[3]))
    duration = (time() - t)/60
    print('Rendering took {} minutes'.format(np.round(duration, 2)))
