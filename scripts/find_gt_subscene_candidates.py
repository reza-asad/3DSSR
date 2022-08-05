import os
from time import time
import argparse
import numpy as np

from scripts.helper import load_from_json, write_to_json, render_scene_subscene, create_img_table, \
    create_img_table_scrollable
from scripts.box import Box
from scripts.iou import IoU
from render_results import find_anchor_translation, find_obj_center


def compute_iou(b1, b2):
    try:
        iou = IoU(b1, b2).iou()
    except Exception:
        # this could happen if the obbox is too thin.
        iou = 0

    return iou


def translate_obbox(box, translation):
    # build the transformation matrix
    transformation = np.eye(4)
    transformation[:3, 3] = translation

    # apply tranlsation to the obbox
    return box.apply_transformation(transformation)


def load_boxes(scene, objects, box_type):
    boxes = {}
    for obj in objects:
        vertices = np.asarray(scene[obj][box_type])
        boxes[obj] = Box(vertices)

    return boxes


def find_topk_target_nodes(args, query_number, query_scene_name, target_scene_names, query_node):
    # load the query scene and find the category of the query object.
    query_scene = load_from_json(os.path.join(args.scene_dir, query_scene_name))

    # for each object in the target scene take it if it has the same category as the query object.
    target_scene_to_nodes = {}
    for target_scene_name in target_scene_names:
        # skip if the target and query scene are the same
        if target_scene_name == query_scene_name:
            continue

        # load the target scene.
        target_scene = load_from_json(os.path.join(args.scene_dir, target_scene_name))

        # take all the objects with the same category as the query object and assign them to the target scene.
        for obj in target_scene.keys():
            if target_scene_name not in target_scene_to_nodes:
                target_scene_to_nodes[target_scene_name] = [obj]
            else:
                target_scene_to_nodes[target_scene_name].append(obj)

    return target_scene_to_nodes


def find_best_correspondence(query_number, query_scene, target_scene_name, target_scene, query_node, target_node,
                             context_objects, q_boxes, t_boxes):
    # read the box for the query object and find the translation that brings it to the origin.
    q_box = q_boxes[query_node]
    q_translation = -q_box.translation
    q_box_translated = translate_obbox(q_box, q_translation)

    # read the box for the target object and find the translation that brings it to the origin.
    t_box = t_boxes[target_node]
    t_translation = -t_box.translation
    t_box_translated = translate_obbox(t_box, t_translation)

    # for each context object find the best candidate in the target object.
    correspondence = {}
    total_iou = 0
    num_matches = 0
    if query_scene[query_node]['category'][0] == target_scene[target_node]['category'][0]:
        total_iou = compute_iou(q_box_translated, t_box_translated)
        num_matches += 1
    for context_object in context_objects:
        best_candidate = None
        best_iou = 0

        # translate the context object using the translation that brings the query object to the origin.
        q_c_box = q_boxes[context_object]
        q_c_box_translated = translate_obbox(q_c_box, q_translation)

        # the best candidate has highest IoU
        all_candidates = [candidate for candidate in target_scene.keys() if candidate != target_node]
        for candidate in all_candidates:
            # skip if the candidate is already assigned or is the target node.
            if candidate in correspondence:
                continue

            # skip if the candidate and context have different categories.
            if query_scene[context_object]['category'][0] != target_scene[candidate]['category'][0]:
                continue

            # translate the candidate object using the translation that brings the target object to the origin.
            t_c_box = t_boxes[candidate]
            t_c_box_translated = translate_obbox(t_c_box, t_translation)

            # compute the IoU between the context and candidate objects.
            iou = compute_iou(t_c_box_translated, q_c_box_translated)
            if iou > best_iou:
                best_candidate = candidate
                best_iou = iou

        if best_candidate is not None:
            correspondence[best_candidate] = context_object
            total_iou += best_iou
            num_matches += 1

    target_subscene = None
    if num_matches >= (len(context_objects) - 1):
        target_subscene = {'scene_name': target_scene_name, 'target': target_node, 'correspondence': correspondence,
                           'num_matches': num_matches, 'total_iou': total_iou}

    return target_subscene


def find_best_target_subscenes(args, query_number, query_info, target_scene_names):
    # load the query info
    query_scene_name = query_info['example']['scene_name']
    query_node = query_info['example']['query']
    context_objects = query_info['example']['context_objects']
    query_scene = load_from_json(os.path.join(args.scene_dir, query_scene_name))
    q_boxes = load_boxes(query_scene, [query_node] + context_objects, box_type='aabb')

    # find the topk target nodes and their corresponding scenes.
    target_scene_to_nodes = find_topk_target_nodes(args, query_number, query_scene_name, target_scene_names, query_node)

    # find the best matching target subscenes.
    target_subscenes = []
    for target_scene_name in target_scene_to_nodes.keys():
        # skip if the target scene is the same as the query scene.
        if target_scene_name == query_scene_name:
            continue

        # for each query object find the best corresponding object in the query scene.
        target_scene = load_from_json(os.path.join(args.scene_dir, target_scene_name))
        t_boxes = load_boxes(target_scene, target_scene.keys(), box_type='aabb')
        for target_node in target_scene_to_nodes[target_scene_name]:
            target_subscene = find_best_correspondence(query_number, query_scene, target_scene_name, target_scene,
                                                       query_node, target_node, context_objects, q_boxes, t_boxes)
            if target_subscene is not None:
                target_subscenes.append(target_subscene)

    # rank the target object based on the highest number of correspondences and overall IoU.
    target_subscenes = sorted(target_subscenes, reverse=True, key=lambda x: (x['num_matches'], x['total_iou']))

    return target_subscenes


def process_queries(args, query_dict, output_dir):
    query_output_file_name = args.query_input_file_name.split('.')[0] + '_{}_{}.json'.format(args.mode,
                                                                                             args.experiment_name)
    query_dict_output_path = os.path.join(output_dir, query_output_file_name)

    # find all the potential target scenes.
    target_scene_names = os.listdir(os.path.join(args.scene_dir))

    # apply scene alignment for each query
    t0 = time()
    query_result = {}
    for i, (query, query_info) in enumerate(query_dict.items()):
        t = time()
        if query not in args.query_list:
            continue
        print('Processing query: {}'.format(query))
        query_result[query] = query_info
        target_subscenes = find_best_target_subscenes(args, i, query_info, target_scene_names)
        query_result[query]['target_subscenes'] = target_subscenes
        duration = (time() - t) / 60
        print('Processing the query took {} minutes'.format(round(duration, 2)))
        print('*' * 50)

    duration_all = (time() - t0) / 60
    print('Processing all queries took {} minutes'.format(round(duration_all, 2)))

    # save the changes to query dict
    write_to_json(query_result, query_dict_output_path)


def render_gt_results(args, query_dict, gt_subscene_candidates, rendering_path):
    for query, query_info in query_dict.items():
        # skip if query is not in the list.
        if query not in args.query_list:
            continue

        query_path = os.path.join(rendering_path, query, 'imgs')
        if not os.path.exists(query_path):
            try:
                os.makedirs(query_path)
            except FileExistsError:
                pass

        # render the query
        scene_name = query_info['example']['scene_name']
        query_graph = load_from_json(os.path.join(args.scene_dir_raw, scene_name))
        q = query_info['example']['query']
        q_context = set(query_info['example']['context_objects'] + [q])

        faded_nodes = [obj for obj in query_graph.keys() if obj not in q_context]
        path = os.path.join(query_path, 'query_{}_{}.png'.format(scene_name.split('.')[0], q))
        render_scene_subscene(graph=query_graph, objects=query_graph.keys(), highlighted_object=[q],
                              faded_nodes=faded_nodes, path=path, model_dir=args.models_dir, colormap=args.colormap,
                              with_height_offset=False)

        # load the gt candidates for the query.
        gt_q_candidates = gt_subscene_candidates[query]

        # render the gt candidates.
        for gt_q_candidate in gt_q_candidates:
            target_scene_name = gt_q_candidate['scene_name']
            target_graph = load_from_json(os.path.join(args.scene_dir_raw, target_scene_name + '.json'))
            t = gt_q_candidate['target']
            t_context = set(list(gt_q_candidate['match_info'].keys()) + [t])

            faded_nodes = [obj for obj in target_graph.keys() if obj not in t_context]
            path = os.path.join(query_path, '{}_{}.png'.format(target_scene_name, t))
            render_scene_subscene(graph=target_graph, objects=target_graph.keys(),
                                  highlighted_object=[t], faded_nodes=faded_nodes, path=path, model_dir=args.models_dir,
                                  colormap=args.colormap, with_height_offset=False)


def create_gt_img_table(args, query_dict, gt_subscene_candidates, rendering_path):
    # add relevance score as the caption for the gt results
    for query, target_subscenes in gt_subscene_candidates.items():
        if query not in args.query_list:
            continue

        # find the query img name
        query_img = 'query_{}_{}.png'.format(query_dict[query]['example']['scene_name'].split('.')[0],
                                             query_dict[query]['example']['query'])
        imgs_path = os.path.join(rendering_path, query, 'imgs')

        scores = []
        for target_subscene in target_subscenes:
            # find the img name
            img_name = '{}_{}.png'.format(target_subscene['scene_name'], target_subscene['target'])

            # compute the relevance score for each target subscene
            rel_score = np.sum(list(target_subscene['match_info'].values()))
            caption = '<br />\n'
            caption += '{} <br />\n'.format(rel_score)
            scores.append((rel_score, img_name, caption))

        # sort the subscene imgs by their relevance score
        scores = sorted(scores, reverse=True, key=lambda x: x[0])
        sorted_imgs = list(list(zip(*scores)))[1]
        captions = list(list(zip(*scores)))[2]

        create_img_table(imgs_path, 'imgs', sorted_imgs, with_query_scene=True, query_img=query_img,
                         evaluation_plot=None, html_file_name='img_table.html', topk=len(sorted_imgs),
                         ncols=3, captions=captions, query_caption=None)


def map_img_names_to_letters(args, gt_subscene_candidates):
    letters = [l for l in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
    img_name_to_letter = {}
    for query, target_subscenes in gt_subscene_candidates.items():
        if query not in args.query_list:
            continue

        img_names = []
        for target_subscene in target_subscenes:
            target_scenen_name = target_subscene['scene_name']
            t = target_subscene['target']

            # find the img name.
            img_name = '{}_{}.png'.format(target_scenen_name, t)
            img_names.append(img_name)

        # create the map
        letters_sub = letters[: len(img_names)]
        img_name_to_letter[query] = dict(zip(img_names, letters_sub))

    return img_name_to_letter


def create_user_img_table(args, query_dict, gt_subscene_candidates, rendering_path):
    # find a mapping from the image names to letters.
    img_name_to_letter = map_img_names_to_letters(args, gt_subscene_candidates)

    # for each query add captions for the retrieved subscenes indicating their id, category and centroid.
    for query, target_subscenes in gt_subscene_candidates.items():
        if query not in args.query_list:
            continue

        # find the query img name
        query_scene_name = query_dict[query]['example']['scene_name']
        q = query_dict[query]['example']['query']
        context_objects_and_q = [q] + query_dict[query]['example']['context_objects']

        query_img = 'query_{}_{}.png'.format(query_scene_name.split('.')[0], q)
        imgs_path = os.path.join(rendering_path, query, 'imgs')

        # load the query scene and find the translation that brings the query object to origin.
        query_scene = load_from_json(os.path.join(args.scene_dir_raw, query_scene_name))
        q_translation = find_anchor_translation(query_scene, q)

        # add caption for the query img.
        query_caption = '<br />\n'
        for q_c in context_objects_and_q:
            center = find_obj_center(query_scene, q_c, q_translation)
            query_caption += '{} {}: {} <br />\n'.format(query_scene[q_c]['category'][0], q_c, center)

        imgs = []
        captions = []
        for target_subscene in target_subscenes:
            target_subscene_name = target_subscene['scene_name']
            t = target_subscene['target']

            # find the img name
            img_name = '{}_{}.png'.format(target_subscene_name, t)
            imgs.append(img_name)

            # load the target scene and find the translation that brings the target object to origin.
            target_scene = load_from_json(os.path.join(args.scene_dir_raw, '{}.json'.format(target_subscene_name)))
            t_translation = find_anchor_translation(target_scene, t)

            # add caption for the image name
            caption = '<br />\n'
            caption += '{} <br />\n'.format(img_name_to_letter[query][img_name])
            for t_c in target_subscene['match_info'].keys():
                center = find_obj_center(target_scene, t_c, t_translation)
                caption += '{} {}: {} <br />\n'.format(target_scene[t_c]['category'][0], t_c, center)
            captions.append(caption)

        # create the img table for the query
        create_img_table_scrollable(imgs_path, 'imgs', imgs, 'ground_truth.html', query_img, topk=len(imgs), ncols=2,
                                    captions=captions, query_caption=query_caption)


def get_args():
    parser = argparse.ArgumentParser('Extracting and Ranking 3D Subscenes', add_help=False)

    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--action', dest='action', default='img_table_users',
                        help='extract | render | img_table_gt | img_table_users')
    parser.add_argument('--mode', dest='mode', default='test', help='val or test')
    parser.add_argument('--scene_dir_raw', default='../data/{}/scenes')
    parser.add_argument('--scene_dir', default='../results/{}/scenes_top10')
    parser.add_argument('--models_dir', default='../data/{}/models')
    parser.add_argument('--colormap_path', default='../data/{}/color_map.json')
    parser.add_argument('--query_dir', default='../queries/{}/')
    parser.add_argument('--query_input_file_name', default='query_dict_top10.json')
    # parser.add_argument('--query_list', default=["cabinet-18"], type=str, nargs='+')
    parser.add_argument('--query_list', default=["table-43", "curtain-7", "mirror-10", "cushion-20", "cabinet-18",
                                                 "chair-45", "cushion-33", "lighting-11", "picture-39", "lighting-46"],
                        type=str, nargs='+')
    parser.add_argument('--gt_subscenes_file_name', default='gt_subscene_candidates.json')
    parser.add_argument('--rendering_path', default='../results/{}/rendered_results')
    parser.add_argument('--cp_dir', default='../results/{}')
    parser.add_argument('--results_folder_name',  default='OracleRankV2')
    parser.add_argument('--experiment_name', default='OracleRankV2')

    return parser


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def main():
    # get the arguments
    parser = argparse.ArgumentParser('Extracting and Ranking 3D Subscenes', parents=[get_args()])
    args = parser.parse_args()
    adjust_paths(args, exceptions=[])

    # make sure to retrieve data from the requested mode
    args.scene_dir = os.path.join(args.scene_dir, args.mode)
    args.scene_dir_raw = os.path.join(args.scene_dir_raw, args.mode)
    args.query_dir = os.path.join(args.query_dir, args.mode)
    args.rendering_path = os.path.join(args.rendering_path, args.mode)
    args.colormap = load_from_json(args.colormap_path)
    args.gt_subscenes_path = os.path.join(args.cp_dir, args.results_folder_name, args.gt_subscenes_file_name)

    # create the output dir for gt candidates if it doesn't exist.
    output_dir_all_candidates = os.path.join(args.cp_dir, args.results_folder_name)
    if not os.path.exists(output_dir_all_candidates):
        try:
            os.makedirs(output_dir_all_candidates)
        except FileExistsError:
            pass

    # output dir for the gt rendered results
    output_dir_gt = os.path.join(args.rendering_path, 'GroundTruthCandidates')

    # load the query
    query_dict_input_path = os.path.join(args.query_dir, args.query_input_file_name)
    query_dict = load_from_json(query_dict_input_path)

    if args.action == 'extract':
        process_queries(args, query_dict, output_dir_all_candidates)
    elif args.action == 'render':
        # load the gt subscene candidates.
        gt_subscene_candidates = load_from_json(args.gt_subscenes_path)
        render_gt_results(args, query_dict, gt_subscene_candidates, output_dir_gt)
    elif args.action == 'img_table_gt':
        # load the gt subscene candidates.
        gt_subscene_candidates = load_from_json(args.gt_subscenes_path)
        create_gt_img_table(args, query_dict, gt_subscene_candidates, output_dir_gt)
    elif args.action == 'img_table_users':
        # load the gt subscene candidates.
        gt_subscene_candidates = load_from_json(args.gt_subscenes_path)
        create_user_img_table(args, query_dict, gt_subscene_candidates, output_dir_gt)


if __name__ == '__main__':
    main()
