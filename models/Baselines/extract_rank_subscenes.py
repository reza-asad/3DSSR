import os
from time import time
import argparse
import numpy as np

from scripts.helper import load_from_json, write_to_json
from scripts.box import Box
from scripts.iou import IoU


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
    query_scene = load_from_json(os.path.join(args.scene_dir_queries, query_scene_name))
    query_cat = query_scene[query_node]['category'][0]

    # for each object in the target scene take it if it has the same category as the query object.
    target_scene_to_nodes = {}
    for target_scene_name in target_scene_names:
        # skip if the target and query scene are the same
        if target_scene_name == query_scene_name:
            continue

        # load the target scene.
        target_scene = load_from_json(os.path.join(args.scene_dir, target_scene_name))

        if args.include_cat:
            # take all the objects with the same category as the query object and assign them to the target scene.
            for obj in target_scene.keys():
                cat_key = 'category'
                if args.with_cat_predictions:
                    cat_key = 'predicted_category'
                target_cat = target_scene[obj][cat_key][0]
                if query_cat == target_cat:
                    if target_scene_name not in target_scene_to_nodes:
                        target_scene_to_nodes[target_scene_name] = [obj]
                    else:
                        target_scene_to_nodes[target_scene_name].append(obj)
        else:
            # take a random object as categories are not used
            all_objects = list(target_scene.keys())
            if len(all_objects) > 0:
                np.random.seed(query_number)
                np.random.shuffle(all_objects)
                target_scene_to_nodes[target_scene_name] = [all_objects[0]]

    return target_scene_to_nodes


def find_best_correspondence(args, query_number, query_scene, target_scene_name, target_scene, query_node, target_node,
                             context_objects, q_boxes, t_boxes, theta):
    # read the box for the query object and find the translation that brings it to the origin.
    q_box = q_boxes[query_node]
    q_translation = -q_box.translation

    # read the box for the target object and find the translation that brings it to the origin.
    t_box = t_boxes[target_node]
    t_translation = -t_box.translation

    # compute the rotation matrix.
    transformation = np.eye(4)
    rotation = np.asarray([[np.cos(theta), -np.sin(theta), 0],
                           [np.sin(theta), np.cos(theta), 0],
                           [0, 0, 1]])
    transformation[:3, :3] = rotation

    # for each context object find the best candidate in the target object.
    correspondence = {}
    for context_object in context_objects:
        best_candidate = None

        # translate the context object using the translation that brings the query object to the origin.
        q_c_box = q_boxes[context_object]
        q_c_box_translated = translate_obbox(q_c_box, q_translation)

        # the best candidate has IoU > 0 and highest embedding similarity.
        all_candidates = [candidate for candidate in target_scene.keys() if candidate != target_node]
        np.random.seed(query_number)
        np.random.shuffle(all_candidates)
        for candidate in all_candidates:
            # skip if the candidate is already assigned or is the target node.
            if candidate in correspondence:
                continue

            if args.include_cat:
                # skip if the candidate and context have different categories.
                cat_key = 'category'
                if args.with_cat_predictions:
                    cat_key = 'predicted_category'
                if query_scene[context_object]['category'][0] != target_scene[candidate][cat_key][0]:
                    continue

            if not args.include_iou:
                best_candidate = candidate
                break

            # translate the candidate object using the translation that brings the target object to the origin.
            t_c_box = t_boxes[candidate]
            t_c_box_translated = translate_obbox(t_c_box, t_translation)

            # rotate the candidate object.
            t_c_box_translated_rotated = t_c_box_translated.apply_transformation(transformation)

            # compute the IoU between the context and candidate objects.
            iou = compute_iou(t_c_box_translated_rotated, q_c_box_translated)
            if iou > args.iou_threshold:
                best_candidate = candidate
                break

        if best_candidate is not None:
            correspondence[best_candidate] = context_object

    target_subscene = {'scene_name': target_scene_name, 'target': target_node, 'correspondence': correspondence,
                       'context_match': len(correspondence), 'theta': theta}

    return target_subscene


def find_best_target_subscenes(args, query_number, query_info, target_scene_names, theta):
    # load the query info
    query_scene_name = query_info['example']['scene_name']
    query_node = query_info['example']['query']
    context_objects = query_info['example']['context_objects']
    query_scene = load_from_json(os.path.join(args.scene_dir_queries, query_scene_name))
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
            target_subscene = find_best_correspondence(args, query_number, query_scene, target_scene_name, target_scene,
                                                       query_node, target_node, context_objects, q_boxes, t_boxes, theta)
            target_subscenes.append(target_subscene)

    # rank the target object based on the highest number of correspondences and overall IoU.
    target_subscenes = sorted(target_subscenes, reverse=True, key=lambda x: x['context_match'])

    return target_subscenes


def get_args():
    parser = argparse.ArgumentParser('Extracting and Ranking 3D Subscenes', add_help=False)

    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--mode', dest='mode', default='test', help='val or test')
    parser.add_argument('--with_cat_predictions', action='store_true', default=False,
                        help='If true predicted categories are used')
    parser.add_argument('--scene_dir_queries', default='../../results/{}/scenes')
    parser.add_argument('--scene_dir', default='../../results/{}/scenes')
    parser.add_argument('--query_dir', default='../../queries/{}/')
    parser.add_argument('--query_input_file_name', default='query_dict_top10.json')
    parser.add_argument('--cp_dir', default='../../results/{}')
    parser.add_argument('--iou_threshold', type=float, default=0)
    parser.add_argument('--grid_res', type=float, default=45)
    parser.add_argument('--include_cat', action='store_true', default=True)
    parser.add_argument('--include_iou', action='store_true', default=True)
    parser.add_argument('--with_rotations', action='store_true', default=True)
    parser.add_argument('--results_folder_name',  default='OracleRank')
    parser.add_argument('--experiment_name', default='OracleRankRot')

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
    args.scene_dir_queries = os.path.join(args.scene_dir_queries, args.mode)
    args.scene_dir = os.path.join(args.scene_dir, args.mode)
    args.query_dir = os.path.join(args.query_dir, args.mode)

    # set the input and output paths for the query dict.
    output_dir = os.path.join(args.cp_dir, args.results_folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    query_dict_input_path = os.path.join(args.query_dir, args.query_input_file_name)
    query_output_file_name = args.query_input_file_name.split('.')[0] + '_{}_{}.json'.format(args.mode,
                                                                                             args.experiment_name)
    query_dict_output_path = os.path.join(output_dir, query_output_file_name)

    # read the query dict
    query_dict = load_from_json(query_dict_input_path)

    # find all the potential target scenes.
    target_scene_names = os.listdir(os.path.join(args.scene_dir))

    # find rotation increments if necessary.
    num_thetas = 1
    grid_res = 0
    if args.with_rotations:
        # prepare the grid search parameters for the best rotation.
        grid_res = args.grid_res * (np.pi / 180)
        num_thetas = int(np.round((2 * np.pi) / grid_res))

    # apply scene alignment for each query
    t0 = time()
    for i, (query, query_info) in enumerate(query_dict.items()):
        t = time()
        print('Processing query: {}'.format(query))
        np.random.seed(i)
        np.random.shuffle(target_scene_names)

        # apply rotation module
        rotated_target_subscenes = []
        for j in range(num_thetas):
            curr_theta = j * grid_res
            target_subscenes = find_best_target_subscenes(args, i, query_info, target_scene_names, curr_theta)
            rotated_target_subscenes += target_subscenes

        # choose the best among all possible rotations of the scenes.
        rotated_target_subscenes = sorted(rotated_target_subscenes, reverse=True, key=lambda x: (x['context_match']))
        query_info['target_subscenes'] = rotated_target_subscenes
        duration = (time() - t) / 60
        print('Processing the query took {} minutes'.format(round(duration, 2)))
        print('*' * 50)

    duration_all = (time() - t0) / 60
    print('Processing all queries took {} minutes'.format(round(duration_all, 2)))

    # save the changes to query dict
    write_to_json(query_dict, query_dict_output_path)


if __name__ == '__main__':
    main()

