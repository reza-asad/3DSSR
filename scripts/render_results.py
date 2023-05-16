import argparse
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

from scripts.helper import load_from_json, render_single_scene, render_scene_subscene, create_img_table_scrollable
from scripts.box import Box


def translate_obbox(box, translation):
    # build the transformation matrix
    transformation = np.eye(4)
    transformation[:3, 3] = translation

    # apply tranlsation to the obbox
    return box.apply_transformation(transformation)


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
            render_scene_subscene(graph=query_graph, objects=query_graph.keys(), highlighted_object=[q],
                                  faded_nodes=faded_nodes, path=path, model_dir=models_dir, colormap=colormap,
                                  with_height_offset=True)

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
            render_scene_subscene(graph=target_graph, objects=target_graph.keys(),
                                  highlighted_object=highlighted_object, faded_nodes=faded_nodes, path=path,
                                  model_dir=models_dir, colormap=colormap, alpha=alpha, beta=beta, gamma=gamma,
                                  with_height_offset=True)


def plot_evaluations(x, y, fig, ax, label,):
    ax.plot(x, y, label=label)
    plt.title("Distance and Overlap mAPs")
    plt.xlabel("Thresholds")
    plt.ylabel('mAP')
    leg = ax.legend()


def find_anchor_translation(scene, o):
    vertices = np.asarray(scene[o]['obbox'])
    o_box = Box(vertices)

    return -o_box.translation


def find_obj_center(target_scene, t_c, t_translation):
    vertices = np.asarray(target_scene[t_c]['obbox'])
    t_c_box = Box(vertices)
    t_c_box = translate_obbox(t_c_box, t_translation)
    center = t_c_box.translation[:2]
    center = [np.round(c, 2) for c in center]

    return center


def main():
    # path to render the experiment
    rendering_path = os.path.join(args.rendering_path, args.mode, args.experiment_name)

    # load the query results.
    if args.results_folder_name == 'None':
        args.results_folder_name = ''
    query_results_path = os.path.join(args.cp_dir, args.model_name, args.results_folder_name,
                                      'query_dict_top10_{}_{}.json'.format(args.mode, args.experiment_name))
    # print(query_results_path)
    # t=y
    query_results = load_from_json(query_results_path)

    # filter query results if necessary.
    filtered_query_results = {}
    if args.query_list[0] != 'all':
        for query, results in query_results.items():
            if query in args.query_list:
                filtered_query_results[query] = results
    else:
        filtered_query_results = query_results

    if args.action == 'make_folders':
        make_rendering_folders(filtered_query_results, rendering_path, args.img_folder_name)

    elif args.action == 'render':
        # load the color map for rendering
        colormap = load_from_json(args.colormap_path)

        # render results from the model
        chunk_size = int(np.ceil(args.topk / args.num_chunks))
        render_model_results(filtered_query_results, args.scene_dir, args.models_dir, rendering_path, colormap,
                             args.num_chunks, chunk_size, args.chunk_idx, args.mode, args.topk, args.img_folder_name)

    elif args.action == 'create_img_table':
        for query, results in filtered_query_results.items():
            # read images in the img directory and find the query img
            imgs_path = os.path.join(rendering_path, query, args.img_folder_name)
            imgs = os.listdir(imgs_path)
            query_img = [e for e in imgs if 'query' in e]

            # sort the topk images
            imgs.remove(query_img[0])
            # imgs.remove(evaluation_plot_name)
            ranks = [int(img_name.split('_')[1]) for img_name in imgs]
            ranks_imgs = zip(ranks, imgs)
            ranks_imgs = sorted(ranks_imgs, key=lambda x: x[0])
            sorted_imgs = []
            if len(ranks_imgs) > 0:
                sorted_imgs = list(list(zip(*ranks_imgs))[1])

            # load the query scene.
            query_scene = load_from_json(os.path.join(args.scene_dir, args.mode, results['example']['scene_name']))
            q = results['example']['query']
            context_objects = results['example']['context_objects']

            # add caption for the query.
            query_caption = '<br />\n'
            q_translation = find_anchor_translation(query_scene, q)
            for q_c in context_objects:
                center = find_obj_center(query_scene, q_c, q_translation)
                query_caption += '{} {}: {} <br />\n'.format(query_scene[q_c]['category'][0], q_c, center)

            # add captions for the top10 results
            captions = []
            target_subscenes = results['target_subscenes']
            for i in range(len(sorted_imgs)):
                caption = '<br />\n'
                # load the target scene
                target_scene_name = target_subscenes[i]['scene_name']
                t = target_subscenes[i]['target']
                target_scene = load_from_json(os.path.join(args.scene_dir, args.mode, target_scene_name))
                correspondence = target_subscenes[i]['correspondence']
                # find the translation for the query and target boxes to the origin
                t_translation = find_anchor_translation(target_scene, t)
                for t_c, _ in correspondence.items():
                    center = find_obj_center(target_scene, t_c, t_translation)
                    caption += '{} {}: {} <br />\n'.format(target_scene[t_c]['category'][0], t_c, center)
                captions.append(caption)

            create_img_table_scrollable(imgs_path, args.img_folder_name, sorted_imgs, query_img=query_img[0],
                                        html_file_name='img_table.html', topk=len(imgs), ncols=2, captions=captions,
                                        query_caption=query_caption)


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def get_args():
    parser = argparse.ArgumentParser('Render Results', add_help=False)
    parser.add_argument('--action', default='make_folders', help='make_folders | render | create_img_table')
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--mode', default='val', help='train | val | test')
    parser.add_argument('--scene_dir', default='../data/{}/scenes')
    parser.add_argument('--models_dir', default='../data/{}/models')
    parser.add_argument('--cp_dir', default='../results/{}')
    parser.add_argument('--img_folder_name', default='imgs')
    parser.add_argument('--rendering_path', default='../results/{}/rendered_results')
    parser.add_argument('--results_folder_name', default='')
    parser.add_argument('--colormap_path', default='../data/{}/color_map.json')
    parser.add_argument('--query_list', dest='query_list', default=["all"], type=str, nargs='+',
                        help='Name of the queries to render. If ["all"] is chosen all queries will be rendered')

    parser.add_argument('--topk', dest='topk', default=10, type=int, help='Number of images rendered for each query.')
    parser.add_argument('--model_name', default='')
    parser.add_argument('--experiment_name', default='')

    parser.add_argument('--num_chunks', default=1, type=int, help='number of chunks for parallel run')
    parser.add_argument('--chunk_idx', default=0, type=int, help='chunk id for parallel run')

    return parser


if __name__ == '__main__':
    # read the args
    parser_ = argparse.ArgumentParser('Extract Regions', parents=[get_args()])
    args = parser_.parse_args()
    adjust_paths(args, exceptions=[])

    main()

