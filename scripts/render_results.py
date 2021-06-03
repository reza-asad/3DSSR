import os
import sys
import numpy as np
from matplotlib import pyplot as plt

from scripts.helper import load_from_json, render_single_scene, create_img_table


def make_rendering_folders(query_results, rendering_path, img_folder='imgs'):
    if not os.path.exists(rendering_path):
        os.makedirs(rendering_path)
    for query, resutls in query_results.items():
        query_path = os.path.join(rendering_path, query, img_folder)
        if not os.path.exists(query_path):
            os.makedirs(query_path)


def render_model_results(query_results, scene_dir, models_dir, rendering_path, colormap, num_chunks, chunk_size,
                         chunk_idx, mode, topk=50, img_folder='imgs'):
    # visualize the result of each query
    for i, (query, results) in enumerate(query_results.items()):
        seen = os.listdir(os.path.join(rendering_path, query, img_folder))
        if len(seen) == (topk + 1):
            continue
        # render the query scene once only
        if chunk_idx == i % num_chunks:
            scene_name = results['example']['scene_name']
            query_graph = load_from_json(os.path.join(scene_dir, mode, scene_name))
            q = results['example']['query']
            q_context = set(results['example']['context_objects'] + [q])

            # render the image
            faded_nodes = [obj for obj in query_graph.keys() if obj not in q_context]
            path = os.path.join(rendering_path, query, img_folder, 'query_{}_{}.png'.format(scene_name.split('.')[0], q))
            render_single_scene(graph=query_graph, objects=query_graph.keys(), highlighted_object=[q],
                                faded_nodes=faded_nodes, path=path, model_dir=models_dir, colormap=colormap)

        # render the topk results from the model
        top_results_chunk = results['target_subscenes'][: topk][chunk_idx * chunk_size:
                                                                (chunk_idx + 1) * chunk_size]
        for j, target_subscene in enumerate(top_results_chunk):
            target_scene_name = target_subscene['scene_name']
            target_graph_path = os.path.join(scene_dir, 'all', target_scene_name)
            target_graph = load_from_json(target_graph_path)
            t = target_subscene['target']
            # rotate the target scene if the model outputs a rotation angle or matrix
            alpha, beta, gamma = 0, 0, 0
            if 'theta' in target_subscene:
                alpha = target_subscene['theta']
            if 'alpha' in target_subscene:
                alpha = target_subscene['alpha']
            if 'beta' in target_subscene:
                beta = target_subscene['beta']
            if 'gamma' in target_subscene:
                gamma = target_subscene['gamma']

            highlighted_object = [t]
            t_context = set(list(target_subscene['correspondence'].keys()) + [t])

            # render the image
            faded_nodes = [obj for obj in target_graph.keys() if obj not in t_context]
            path = os.path.join(rendering_path, query, img_folder, 'top_{}_{}_{}.png'
                                .format(chunk_idx * chunk_size + j + 1, target_scene_name.split('.')[0], t))
            render_single_scene(graph=target_graph, objects=target_graph.keys(),
                                highlighted_object=highlighted_object, faded_nodes=faded_nodes, path=path,
                                model_dir=models_dir, colormap=colormap, alpha=alpha, beta=beta, gamma=gamma)


def plot_evaluations(x, y, fig, ax, label,):
    ax.plot(x, y, label=label)
    plt.title("Distance and Overlap mAPs")
    plt.xlabel("Thresholds")
    plt.ylabel('mAP')
    leg = ax.legend()


def main(num_chunks, chunk_idx, mode, model_name='LearningBased', experiment_name='AlignRank', render=False, topk=5,
         make_folders=False, with_img_table=False, include_queries=['all']):
    # define paths
    query_results_path = '../results/matterport3d/{}/query_dict_{}_{}_evaluated.json'.format(model_name, mode,
                                                                                             experiment_name)
    scene_dir = '../data/matterport3d/scenes'
    rendering_path = '../results/matterport3d/rendered_results/{}/{}'.format(mode, experiment_name)
    models_dir = '../data/matterport3d/models'
    colormap = load_from_json('../data/matterport3d/color_map.json')
    img_folder = 'imgs'
    caption_keys = {'overlap_mAP', 'theta'}

    # load the query results and filter them if necessary.
    query_results = load_from_json(query_results_path)
    filtered_query_results = {}
    if include_queries[0] != 'all':
        for query, results in query_results.items():
            if query in include_queries:
                filtered_query_results[query] = results
    else:
        filtered_query_results = query_results

    if make_folders:
        # make rending folders
        make_rendering_folders(filtered_query_results, rendering_path, img_folder)

    if render:
        # render results from the model
        chunk_size = int(np.ceil(topk / num_chunks))
        render_model_results(filtered_query_results, scene_dir, models_dir, rendering_path, colormap, num_chunks,
                             chunk_size, chunk_idx, mode, topk, img_folder)

    if with_img_table:
        for query, results in filtered_query_results.items():
            # plot the distance and overlap mAP for the query
            evaluation_plot_name = 'evaluation.png'
            evaluation_plot_path = os.path.join(rendering_path, query, img_folder, evaluation_plot_name)
            fig, ax = plt.subplots()
            for metric in ['overlap_mAP']:
                x, y = list(zip(*results[metric]['mAP']))
                plot_evaluations(x, y, fig, ax, metric)
            plt.savefig(evaluation_plot_path)
            plt.close()

            # read images in the img directory and find the query img
            imgs_path = os.path.join(rendering_path, query, img_folder)
            imgs = os.listdir(imgs_path)
            query_img = [e for e in imgs if 'query' in e]

            # sort the topk images
            imgs.remove(query_img[0])
            imgs.remove(evaluation_plot_name)
            ranks = [int(img_name.split('_')[1]) for img_name in imgs]
            ranks_imgs = zip(ranks, imgs)
            ranks_imgs = sorted(ranks_imgs, key=lambda x: x[0])
            sorted_imgs = []
            if len(ranks_imgs) > 0:
                sorted_imgs = list(list(zip(*ranks_imgs))[1])

            # add captions for the top10 results
            captions = []
            target_subscenes = results['target_subscenes']
            topk = 10
            for i in range(len(sorted_imgs)):
                caption = '<br />\n'
                # add number of context matches
                if i < topk:
                    caption += 'num_objects: {} <br />\n'.format(target_subscenes[i]['context_match'] + 1)
                for key, value in target_subscenes[i].items():
                    if key in caption_keys:
                        if 'theta' in key:
                            caption_value = np.round(value * 180 / np.pi, 0)
                        else:
                            caption_value = value
                        caption += ' {}: {} <br />\n'.format(key, caption_value)
                captions.append(caption)

            create_img_table(imgs_path, img_folder, sorted_imgs, with_query_scene=True, query_img=query_img[0],
                             evaluation_plot=evaluation_plot_name, html_file_name='img_table.html', topk=len(imgs),
                             ncols=3, captions=captions)


if __name__ == '__main__':
    include_queries = sys.argv[10]
    qs = include_queries.split(',')
    new_qs = []
    for i, e in enumerate(qs):
        if i == 0:
            new_qs.append(e[1:])
        elif i == (len(qs) - 1):
            new_qs.append(e[:-1])
        else:
            new_qs.append(e)

    bool_args = [sys.argv[6], sys.argv[8], sys.argv[9]]
    for i in range(len(bool_args)):
        bool_args[i] = bool_args[i].lower() == 'true'

    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], bool_args[0],
         int(sys.argv[7]), bool_args[1], bool_args[2], new_qs)

