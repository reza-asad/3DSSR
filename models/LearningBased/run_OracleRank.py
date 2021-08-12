import os
import torch
import numpy as np
from time import time
from optparse import OptionParser

from scripts.helper import load_from_json, write_to_json
from scripts.box import Box
from scripts.iou import IoU


def find_boxes(scene, objects, box_type):
    boxes = {}
    for obj in objects:
        vertices = np.asarray(scene[obj][box_type])
        boxes[obj] = Box(vertices)

    return boxes


def translate_obbox(obbox, translation):
    # build the transformation matrix
    transformation = np.eye(4)
    transformation[:3, 3] = translation

    # apply tranlsation to the obbox
    obbox = obbox.apply_transformation(transformation)

    return obbox


def find_best_correspondence(target_scene_name, target_scene, query_node, target_node, context_objects, q_boxes,
                             t_boxes):
    # read the box for the query object and find the translation that brings it to the origin.
    q_box = q_boxes[query_node]
    q_translation = -q_box.translation
    q_box_origin = translate_obbox(q_box, q_translation)

    # read the box for the target object and find the translation that brings it to the origin.
    t_box = t_boxes[target_node]
    t_translation = -t_box.translation
    t_box_origin = translate_obbox(t_box, t_translation)

    # for each context object find the best candidate in the target object.
    correspondence = {}
    overall_iou = IoU(t_box_origin, q_box_origin).iou()

    # for each context object find the candidate of the same category with highest IoU.
    for context_object in context_objects:
        highest_iou = 0
        best_candidate = None

        # translate the context object using the translation that brings the query object to the origin.
        q_c_box = q_boxes[context_object]
        q_c_box_translated = translate_obbox(q_c_box, q_translation)

        # the best candidate has highest IoU with the context object.
        for candidate in target_scene.keys():
            # skip if the candidate is already assigned or is the target node.
            if (candidate in correspondence) or (candidate == target_node):
                continue

            # translate the candidate object using the translation that brings the target object to the origin.
            t_c_box = t_boxes[candidate]
            t_c_box_translated = translate_obbox(t_c_box, t_translation)

            # record the IoU between the context object and each candidate
            try:
                iou = IoU(q_c_box_translated, t_c_box_translated).iou()
            except Exception:
                # this could happen if the obbox is too thin.
                iou = 0

            if iou > highest_iou:
                highest_iou = iou
                best_candidate = candidate

        if best_candidate is not None:
            correspondence[best_candidate] = context_object
            overall_iou += highest_iou

    # assemble the target subscene
    target_subscene = {'scene_name': target_scene_name, 'target': target_node, 'correspondence': correspondence,
                       'context_match': len(correspondence), 'IoU': overall_iou}

    return target_subscene


def find_best_target_subscenes(query_info, scene_dir, target_scene_names, mode):
    # load the query info
    query_scene_name = query_info['example']['scene_name']
    query_node = query_info['example']['query']
    context_objects = query_info['example']['context_objects']
    query_scene = load_from_json(os.path.join(scene_dir, mode, query_scene_name))
    q_boxes = find_boxes(query_scene, [query_node] + context_objects, 'obbox')

    # apply the models to valid/test data
    target_subscenes = []
    for target_scene_name in target_scene_names:
        # skip if the target scene is the same as the query scene.
        if target_scene_name == query_scene_name:
            continue

        # TODO: rotate the target scene to best align with the query subscene

        # for each query object find the best corresponding object in the query scene.
        target_scene = load_from_json(os.path.join(scene_dir, mode, target_scene_name))
        t_boxes = find_boxes(target_scene, target_scene.keys(), 'obbox')
        target_nodes = [n for n in target_scene.keys() if target_scene[n]['category'][0] ==
                        query_scene[query_node]['category'][0]]
        for target_node in target_nodes:
            target_subscene = find_best_correspondence(target_scene_name, target_scene, query_node, target_node,
                                                       context_objects, q_boxes, t_boxes)
            target_subscenes.append(target_subscene)

    # rank the target object based on the highest number of correspondences and overall IoU.
    target_subscenes = sorted(target_subscenes, reverse=True, key=lambda x: (x['context_match'], x['IoU']))

    return target_subscenes


def get_args():
    parser = OptionParser()
    parser.add_option('--mode', dest='mode', default='val', help='val or test')
    parser.add_option('--scene_dir', dest='scene_dir',
                      default='../../results/matterport3d/LearningBased/scenes_with_perfect_regions')
    parser.add_option('--experiment_name', dest='experiment_name', default='OracleRank')

    (options, args) = parser.parse_args()
    return options


def main():
    # get the arguments
    args = get_args()

    # set the input and output paths for the query dict.
    query_dict_input_path = '../../queries/matterport3d/query_dict_{}.json'.format(args.mode)
    query_dict_output_path = '../../results/matterport3d/LearningBased/query_dict_{}_{}.json'.format(args.mode,
                                                                                                     args.experiment_name)

    # find all the potential target scenes.
    target_scene_names = os.listdir(os.path.join(args.scene_dir, args.mode))

    # read the query dict
    query_dict = load_from_json(query_dict_input_path)

    # apply scene alignment for each query
    t0 = time()
    for i, (query, query_info) in enumerate(query_dict.items()):
        t = time()
        print('Processing query: {}'.format(query))
        target_subscenes = find_best_target_subscenes(query_info, args.scene_dir, target_scene_names, args.mode)
        query_info['target_subscenes'] = target_subscenes
        duration = (time() - t) / 60
        print('Processing the query took {} minutes'.format(round(duration, 2)))
        print('*' * 50)
    duration_all = (time() - t0) / 60
    print('Processing all queries too {} minutes'.format(round(duration_all, 2)))

    # save the changes to query dict
    write_to_json(query_dict, query_dict_output_path)


if __name__ == '__main__':
    main()
