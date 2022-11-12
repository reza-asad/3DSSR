import argparse
import heapq
import os
from time import time
import numpy as np
import torch

from scripts.helper import load_from_json, write_to_json
from scripts.box import Box
from scripts.iou import IoU


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


def load_boxes_target(boxes_list):
    boxes = {}
    for i, vertices in enumerate(boxes_list):
        vertices = np.asarray(vertices)
        boxes[i] = Box(vertices)

    return boxes


def find_topk_target_nodes(query_scene_name, query_node, target_subscenes, scene_to_category_dict):
    # load the query scene and find the category of the query object.
    query_scene = load_from_json(os.path.join(args.scene_dir, query_scene_name))
    query_cat = query_scene[query_node]['category'][0]

    # for each object in the target scene take it if it has the same category as the query object.
    target_node_indices = []
    heapq.heapify(target_node_indices)
    for i, target_subscene in enumerate(target_subscenes):
        target_scene_name = target_subscene['scene_name']

        # skip if the target and query scene are the same
        if target_scene_name == query_scene_name:
            continue

        # for the current target subscene take all the objects with the same category as the query object.
        for j, _ in enumerate(target_subscene['cats']):
            if query_cat in scene_to_category_dict[target_scene_name][j]:
                # if len(target_node_indices) < args.topk:
                heapq.heappush(target_node_indices, (target_subscene['scores'][j], (i, j)))
                # else:
                #     heapq.heappushpop(target_node_indices, (target_subscene['scores'][j], (i, j)))

    return target_node_indices


def find_best_correspondence(query_scene_name, target_scene_name, query_node, context_objects, q_boxes, t_obj_idx,
                             t_boxes, target_subscene_raw, id_to_predicted_cat):
    # load the query scene.
    query_scene = load_from_json(os.path.join(args.scene_dir, query_scene_name))

    # read the box for the query object and find the translation that brings it to the origin.
    q_box = q_boxes[query_node]
    q_translation = -q_box.translation

    # read the box for the target object and find the translation that brings it to the origin.
    t_box = t_boxes[t_obj_idx]
    t_translation = -t_box.translation

    # for each context object find the best candidate in the target object.
    correspondence = {}
    total_score = target_subscene_raw['scores'][t_obj_idx]
    for context_object in context_objects:
        highest_score = 0
        best_candidate = None

        # translate the context object using the translation that brings the query object to the origin.
        q_c_box = q_boxes[context_object]
        q_c_box_translated = translate_obbox(q_c_box, q_translation)

        # find the category of the context object.
        context_object_cat = query_scene[context_object]['category'][0]

        # the best candidate has IoU > 0 and highest embedding similarity.
        for j, score in enumerate(target_subscene_raw['scores']):
            # skip if the candidate is already assigned or is the target node.
            if (j in correspondence) or (j == t_obj_idx):
                continue

            # skip if the predicted category of the candidate is different from the context object.
            if context_object_cat not in id_to_predicted_cat[j]:
                continue

            # translate the candidate object using the translation that brings the target object to the origin.
            t_c_box = t_boxes[j]
            t_c_box_translated = translate_obbox(t_c_box, t_translation)

            # compute the IoU between the context and candidate objects.
            try:
                iou = IoU(t_c_box_translated, q_c_box_translated).iou()
            except Exception:
                # this could happen if the obbox is too thin.
                iou = 0

            # only continue if the candidate has and iou with the context greater than 0.
            if iou > 0:
                # consider scores for using the current candidate over previous assignments.
                current_score = target_subscene_raw['scores'][j]

                if current_score > highest_score:
                    highest_score = current_score
                    best_candidate = j

        if best_candidate is not None:
            correspondence[str(best_candidate)] = context_object
            total_score += highest_score

    obj_to_cat = {}
    obj_to_box = {}
    for obj, cats in id_to_predicted_cat.items():
        if (obj == t_obj_idx) or (str(obj) in correspondence):
            obj_to_cat[str(obj)] = cats
            obj_to_box[str(obj)] = target_subscene_raw['boxes'][obj]
    target_subscene = {'scene_name': target_scene_name + '.json', 'target': str(t_obj_idx), 'correspondence': correspondence,
                       'context_match': len(correspondence), 'total_score': float(total_score), 'obj_to_cat': obj_to_cat,
                       'obj_to_box': obj_to_box}

    return target_subscene


def find_cat_at_threshold(target_scene_name, target_subscene_raw, t_boxes):
    # load the target scene.
    target_scene = load_from_json(os.path.join(args.scene_dir, target_scene_name+'.json'))

    # for each predicted box, only take its predicted category if the IoU between the box and at least one gt box of the
    # same category is above a threshold.
    id_to_predicted_cat = {}
    for obj, obj_info in target_scene.items():
        gt_box_vertices = np.asarray(obj_info['aabb'], dtype=np.float32)
        gt_box = Box(gt_box_vertices)
        gt_cat = obj_info['category'][0]
        # iterate through the predicted boxes.
        for i, pred_cat in enumerate(target_subscene_raw['cats']):
            # predicted and gt categories must match.
            if pred_cat == gt_cat:
                # if IoU greater than a threshold, accept the predicted cat.
                try:
                    iou = IoU(t_boxes[i], gt_box).iou()
                except Exception:
                    iou = 0
                if iou > args.cat_threshold:
                    if i in id_to_predicted_cat:
                        id_to_predicted_cat[i].append(pred_cat)
                    else:
                        id_to_predicted_cat[i] = [pred_cat]

    # assign -1 to all boxes that have no match.
    for i in range(len(target_subscene_raw['cats'])):
        if i not in id_to_predicted_cat:
            id_to_predicted_cat[i] = ['-1']

    return id_to_predicted_cat


def map_scene_to_accepted_predicted_cats(target_subscenes):
    scene_to_category_dict = {}
    for target_subscene_raw in target_subscenes:
        target_scene_name = target_subscene_raw['scene_name']
        t_boxes = load_boxes_target(target_subscene_raw['boxes'])
        id_to_predicted_cat = find_cat_at_threshold(target_scene_name, target_subscene_raw, t_boxes)
        scene_to_category_dict[target_scene_name] = id_to_predicted_cat

    return scene_to_category_dict


def find_best_target_subscenes(query_info):
    # load the query info
    query_scene_name = query_info['example']['scene_name']
    query_node = query_info['example']['query']
    context_objects = query_info['example']['context_objects']
    query_scene = load_from_json(os.path.join(args.scene_dir, query_scene_name))
    q_boxes = load_boxes(query_scene, [query_node] + context_objects, box_type='aabb')

    # find a mapping from each scene to the predicted categories that match gt (above a threshold).
    scene_to_category_dict = map_scene_to_accepted_predicted_cats(query_info['target_subscenes'])

    # find the index (scene_idx, obj_idx) of topk target nodes among the retrieved subscenes.
    target_node_indices = find_topk_target_nodes(query_scene_name, query_node, query_info['target_subscenes'],
                                                 scene_to_category_dict)

    # find the best matching target subscenes given each anchor object.
    target_subscenes = []
    for score, (scene_idx, t_obj_idx) in target_node_indices:
        # find the target scene name.
        target_scene_name = query_info['target_subscenes'][scene_idx]['scene_name']

        # load the boxes for the current target subscene.
        target_subscene_raw = query_info['target_subscenes'][scene_idx]
        t_boxes = load_boxes_target(target_subscene_raw['boxes'])
        id_to_predicted_cat = scene_to_category_dict[target_scene_name]
        target_subscene = find_best_correspondence(query_scene_name, target_scene_name, query_node, context_objects,
                                                   q_boxes, t_obj_idx, t_boxes, target_subscene_raw,
                                                   id_to_predicted_cat)
        target_subscenes.append(target_subscene)

    # rank the target object based on the highest number of correspondences and overall IoU.
    target_subscenes = sorted(target_subscenes, reverse=True, key=lambda x: (x['context_match'], x['total_score']))[:args.topk]

    return target_subscenes


def make_args_parser():
    parser = argparse.ArgumentParser("Saving Detected Boxes", add_help=False)

    parser.add_argument('--dataset_name', default='matterport3d')
    parser.add_argument('--mode', default='test')
    parser.add_argument('--scene_dir', default='../../results/{}/scenes')
    parser.add_argument("--output_path", default='../../results/{}/LayoutMatching/', type=str)
    parser.add_argument('--query_input_file_name', default='query_dict_top10.json')
    parser.add_argument('--results_folder_name', default='full_3dssr_real_query')
    parser.add_argument("--experiment_name", default='3detr_pre_rank', type=str)
    parser.add_argument('--cat_threshold', default=0.25, help='Threshold for categorizing the predicted boxes.')
    parser.add_argument('--topk', dest='topk', default=20, help='Number of most similar subscenes to be returned.')

    return parser


def adjust_paths(exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset_name.split('_')[0])
            vars(args)[k] = os.path.join(base_dir, v)


def main():
    # make sure to retrieve data from the requested mode
    args.scene_dir = os.path.join(args.scene_dir, args.mode)

    # set the input output paths for the query results.
    query_input_file_name = args.query_input_file_name.split('.')[0] + '_{}_{}.json'.format(args.mode,
                                                                                            args.experiment_name)
    experiment_name_out = '-'.join(args.experiment_name.split('_')[:-2])
    query_output_file_name = args.query_input_file_name.split('.')[0] + '_{}_{}.json'.format(args.mode,
                                                                                             experiment_name_out)

    output_path = os.path.join(args.output_path, args.results_folder_name)
    query_dict_input_path = os.path.join(output_path, query_input_file_name)
    query_dict_output_path = os.path.join(output_path, query_output_file_name)

    # load the query results before assignment and ranking.
    query_results = load_from_json(query_dict_input_path)

    # apply scene alignment for each query
    t0 = time()
    query_results_ranked = {}
    for i, (query, query_info) in enumerate(query_results.items()):
        t = time()
        print('Processing query: {}'.format(query))
        # find best target subscenes for the query.
        target_subscenes = find_best_target_subscenes(query_info)

        query_info['target_subscenes'] = target_subscenes
        query_results_ranked[query] = query_info

        duration = (time() - t) / 60
        print('Processing the query took {} minutes'.format(round(duration, 2)))
        print('*' * 50)

    duration_all = (time() - t0) / 60
    print('Processing all queries took {} minutes'.format(round(duration_all, 2)))

    # save the changes to query dict
    write_to_json(query_results_ranked, query_dict_output_path)


if __name__ == "__main__":
    parser = make_args_parser()
    args = parser.parse_args()
    adjust_paths([])

    main()
