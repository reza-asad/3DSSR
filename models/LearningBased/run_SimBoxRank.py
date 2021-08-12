import heapq
import os
import numpy as np
from time import time
from optparse import OptionParser

from scripts.helper import load_from_json, write_to_json
from scripts.box import Box
from scripts.iou import IoU


def cos_sim(a, b, norm_a, norm_b):
    return np.dot(a, b) / (norm_a * norm_b)


def translate_obbox(box, translation):
    # build the transformation matrix
    transformation = np.eye(4)
    transformation[:3, 3] = translation

    # apply tranlsation to the obbox
    return box.apply_transformation(transformation)


def find_boxes(scene, objects, box_type):
    boxes = {}
    for obj in objects:
        vertices = np.asarray(scene[obj][box_type])
        boxes[obj] = Box(vertices)

    return boxes


def load_embedding(embedding_dir, scene_name, obj_id):
    embedding = np.load(os.path.join(embedding_dir, '{}-{}.npy'.format(scene_name.split('.')[0], obj_id)))

    return embedding.reshape(-1)


def find_topk_target_nodes(query_scene_name, target_scene_names, query_node, embedding_dir, scene_dir, mode, topk):
    # load the embedding of the query object
    embedding_query = load_embedding(embedding_dir, query_scene_name, query_node)
    embedding_query_norm = np.linalg.norm(embedding_query)

    # for each object in the target scene compute the similarity of its embedding to the query object and inert the
    # result in a fixed size heap
    sim_results = []
    heapq.heapify(sim_results)
    for target_scene_name in target_scene_names:
        # skip if the target and query scene are the same
        if target_scene_name == query_scene_name:
            continue

        # load the target scene.
        target_scene = load_from_json(os.path.join(scene_dir, mode, target_scene_name))

        # for each object in the target scene compute its similarity with the query object.
        for obj in target_scene.keys():
            # load the obj embedding
            embedding_obj = load_embedding(embedding_dir, target_scene_name, obj)
            embedding_obj_norm = np.linalg.norm(embedding_obj)

            # compute the similarity
            current_sim = cos_sim(embedding_query, embedding_obj, embedding_query_norm, embedding_obj_norm)

            # insert the similarity into the heap if there are less than topk elements there. otherwise compare the
            # similarity with the smallest and replace if necessary.
            if len(sim_results) < topk:
                heapq.heappush(sim_results, (current_sim, (target_scene_name, obj)))
            elif heapq.nsmallest(1, sim_results)[0][0] < current_sim:
                heapq.heappushpop(sim_results, (current_sim, (target_scene_name, obj)))

    # for each target scene gather its target node candidates in a list
    target_scene_to_nodes = {}
    for _, (target_scene_name, obj) in sim_results:
        if target_scene_name not in target_scene_to_nodes:
            target_scene_to_nodes[target_scene_name] = [obj]
        else:
            target_scene_to_nodes[target_scene_name].append(obj)

    return target_scene_to_nodes


def find_best_correspondence(query_scene_name, target_scene_name, target_scene, query_node, target_node,
                             context_objects, embedding_dir, q_boxes, t_boxes):
    # read the box for the query object and find the translation that brings it to the origin.
    q_box = q_boxes[query_node]
    q_translation = -q_box.translation

    # read the embedding of the query object.
    embedding_query = load_embedding(embedding_dir, query_scene_name, query_node)
    embedding_query_norm = np.linalg.norm(embedding_query)

    # read the box for the target object and find the translation that brings it to the origin.
    t_box = t_boxes[target_node]
    t_translation = -t_box.translation

    # read the embedding of the target object.
    embedding_target = load_embedding(embedding_dir, target_scene_name, target_node)
    embedding_target_norm = np.linalg.norm(embedding_target)

    # for each context object find the best candidate in the target object.
    correspondence = {}
    total_sim = cos_sim(embedding_query, embedding_target, embedding_query_norm, embedding_target_norm)

    for context_object in context_objects:
        highest_sim = 0
        best_candidate = None

        # translate the context object using the translation that brings the query object to the origin.
        q_c_box = q_boxes[context_object]
        q_c_box_translated = translate_obbox(q_c_box, q_translation)

        # read the embedding of the context object.
        embedding_context = load_embedding(embedding_dir, query_scene_name, context_object)
        embedding_context_norm = np.linalg.norm(embedding_context)

        # the best candidate has IoU > 0 and highest embedding similarity.
        for candidate in target_scene.keys():
            # skip if the candidate is already assigned or is the target node.
            if (candidate in correspondence) or (candidate == target_node):
                continue

            # translate the candidate object using the translation that brings the target object to the origin.
            t_c_box = t_boxes[candidate]
            t_c_box_translated = translate_obbox(t_c_box, t_translation)

            # compute the IoU between the context and candidate objects.
            try:
                iou = IoU(t_c_box_translated, q_c_box_translated).iou()
            except Exception:
                # this could happen if the obbox is too thin.
                iou = 0

            # for positive IoUs find the candidate with highest embedding similarity.
            if iou > 0:
                # read the embedding for the candidate object.
                embedding_candidate = load_embedding(embedding_dir, target_scene_name, candidate)
                embedding_candidate_norm = np.linalg.norm(embedding_candidate)

                # compute the embedding similarity between candidate and context objects.
                current_sim = cos_sim(embedding_context, embedding_candidate, embedding_context_norm,
                                      embedding_candidate_norm)
                if current_sim > highest_sim:
                    highest_sim = current_sim
                    best_candidate = candidate

        if best_candidate is not None:
            correspondence[best_candidate] = context_object
            total_sim += highest_sim

    target_subscene = {'scene_name': target_scene_name, 'target': target_node, 'correspondence': correspondence,
                       'total_sim': float(total_sim)}

    return target_subscene


def find_best_target_subscenes(query_info, scene_dir, embedding_dir, mode, target_scene_names, box_type, topk=100):
    # load the query info
    query_scene_name = query_info['example']['scene_name']
    query_node = query_info['example']['query']
    context_objects = query_info['example']['context_objects']
    query_scene = load_from_json(os.path.join(scene_dir, mode, query_scene_name))
    q_boxes = find_boxes(query_scene, [query_node] + context_objects, box_type)

    # find the topk target nodes and their corresponding scenes.
    target_scene_to_nodes = find_topk_target_nodes(query_scene_name, target_scene_names, query_node, embedding_dir,
                                                   scene_dir, mode, topk)

    # find the best matching target subscenes.
    target_subscenes = []
    for target_scene_name in target_scene_to_nodes.keys():
        # skip if the target scene is the same as the query scene.
        if target_scene_name == query_scene_name:
            continue

        # TODO: rotate the target scene to best align with the query subscene

        # for each query object find the best corresponding object in the query scene.
        target_scene = load_from_json(os.path.join(scene_dir, mode, target_scene_name))
        t_boxes = find_boxes(target_scene, target_scene.keys(), box_type)
        for target_node in target_scene_to_nodes[target_scene_name]:
            target_subscene = find_best_correspondence(query_scene_name, target_scene_name, target_scene, query_node,
                                                       target_node, context_objects, embedding_dir, q_boxes, t_boxes)
            target_subscenes.append(target_subscene)

    # rank the target object based on the highest number of correspondences and overall IoU.
    target_subscenes = sorted(target_subscenes, reverse=True, key=lambda x: x['total_sim'])

    return target_subscenes


def get_args():
    parser = OptionParser()
    parser.add_option('--mode', dest='mode', default='val', help='val or test')
    parser.add_option('--scene_dir', dest='scene_dir',
                      default='../../results/matterport3d/LearningBased/scenes_with_regions')
    parser.add_option('--embedding_dir', dest='embedding_dir',
                      default='../../results/matterport3d/LearningBased/latent_caps_obbox_60DINO')
    parser.add_option('--box_type', dest='box_type', default='obbox', help='obbox|bbox_region|bbox_region_predicted')
    parser.add_option('--topk', dest='topk', default=100, help='Number of most similar subscenes to be returned.')
    parser.add_option('--experiment_name', dest='experiment_name', default='obbox_DINO_sim')

    (options, args) = parser.parse_args()
    return options


def main():
    # get the arguments
    args = get_args()

    # set the input and output paths for the query dict.
    query_dict_input_path = '../../queries/matterport3d/query_dict_{}.json'.format(args.mode)
    query_dict_output_path = '../../results/matterport3d/LearningBased/query_dict_{}_{}.json'.format(args.mode,
                                                                                                     args.experiment_name)

    # read the query dict
    query_dict = load_from_json(query_dict_input_path)

    # find all the potential target scenes.
    target_scene_names = os.listdir(os.path.join(args.scene_dir, args.mode))

    # apply scene alignment for each query
    t0 = time()
    for i, (query, query_info) in enumerate(query_dict.items()):
        t = time()
        # keep track of processed queries.
        if query in visited:
            continue
        visited.add(query)

        print('Processing query: {}'.format(query))
        target_subscenes = find_best_target_subscenes(query_info, args.scene_dir, args.embedding_dir,
                                                      args.mode, target_scene_names, args.box_type, args.topk)
        query_info['target_subscenes'] = target_subscenes
        duration = (time() - t) / 60
        print('Processing the query took {} minutes'.format(round(duration, 2)))
        print('*' * 50)

    duration_all = (time() - t0) / 60
    print('Processing all queries took {} minutes'.format(round(duration_all, 2)))

    # save the changes to query dict
    write_to_json(query_dict, query_dict_output_path)


if __name__ == '__main__':
    visited = set()
    main()

