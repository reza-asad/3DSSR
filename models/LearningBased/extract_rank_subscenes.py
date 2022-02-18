import heapq
import os
from time import time
import numpy as np
import torch
import pandas as pd

from scripts.helper import load_from_json, write_to_json
from scripts.box import Box
from scripts.iou import IoU


def map_file_names_to_idx(args, file_indices):
    # load accepted categories.
    accepted_cats = load_from_json(args.accepted_cats_path)

    # find a mapping from all the region files (train+val) to their indices.
    df = pd.read_csv(args.metadata_path)
    is_accepted = df['mpcat40'].apply(lambda x: x in accepted_cats)
    df = df.loc[is_accepted]
    df = df.loc[(df['split'] == 'train') | (df['split'] == args.mode)]
    file_names = df[['room_name', 'objectId']].apply(lambda x: '-'.join([str(x[0]), str(x[1]) + '.npy']), axis=1).tolist()
    file_names = sorted(file_names)
    file_idx_to_file_name = {i: file_name for i, file_name in enumerate(file_names)}

    # find a mapping from the file names in the validation data to their actual indices in the test_file_names.pth.
    file_name_to_idx = {}
    for i, file_idx in enumerate(file_indices):
        file_name = file_idx_to_file_name[file_idx.item()]
        file_name_to_idx[file_name] = i

    return file_name_to_idx


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
        # transformation = np.eye(4)
        # transformation[:3, 3] = boxes[obj].vertices[0, ...]
        # box_vis = trimesh.creation.box(boxes[obj].scale, transform=transformation)
        # box_vis_list.append(box_vis)
        # mesh = trimesh.load(os.path.join('../data/matterport3d/models', scene[obj]['file_name']))
        # mesh = mesh.apply_transform(transformation)
        # meshes.append(mesh)

    # trimesh.Scene(meshes).show()
    # trimesh.Scene(box_vis_list).show()
    # t=y

    return boxes


def find_topk_target_nodes(args, query_scene_name, target_scene_names, query_node, features, labels, predicted_labels,
                           predicted_clusters, file_name_to_idx):
    # load the embedding of the query object
    query_file_name = '{}-{}.npy'.format(query_scene_name.split('.')[0], query_node)
    query_idx = file_name_to_idx[query_file_name]
    query_feature = features[query_idx, :]

    # for each object in the target scene compute the similarity of its embedding to the query object and inert the
    # result in a fixed size heap.
    sim_results = []
    heapq.heapify(sim_results)
    for target_scene_name in target_scene_names:
        # skip if the target and query scene are the same
        if target_scene_name == query_scene_name:
            continue

        # load the target scene.
        target_scene = load_from_json(os.path.join(args.scene_dir, target_scene_name))

        # for each object in the target scene compute its similarity with the query object.
        for obj in target_scene.keys():
            # load the obj embedding
            obj_file_name = '{}-{}.npy'.format(target_scene_name.split('.')[0], obj)
            obj_idx = file_name_to_idx[obj_file_name]
            obj_feature = features[obj_idx, :]

            # compute the similarity
            current_sim = torch.dot(obj_feature, query_feature)

            # choose a match either based on predicted labels if available
            if predicted_labels is not None:
                if labels[query_idx] != predicted_labels[obj_file_name]:
                    continue
            elif predicted_clusters is not None:
                if predicted_clusters[query_file_name] != predicted_clusters[obj_file_name]:
                    continue

            # insert the similarity into the heap if there are less than topk elements there. otherwise, compare the
            # similarity with the smallest and replace if necessary.
            if len(sim_results) < args.topk:
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


def find_best_correspondence(args, query_scene_name, target_scene_name, target_scene, query_node, target_node,
                             context_objects, features, labels, predicted_labels, predicted_clusters, file_name_to_idx,
                             q_boxes, t_boxes):
    # read the box for the query object and find the translation that brings it to the origin.
    q_box = q_boxes[query_node]
    q_translation = -q_box.translation
    q_box_origin = translate_obbox(q_box, q_translation)

    # read the embedding of the query object.
    query_file_name = '{}-{}.npy'.format(query_scene_name.split('.')[0], query_node)
    query_idx = file_name_to_idx[query_file_name]
    query_feature = features[query_idx, :]

    # read the embedding of the target object.
    target_file_name = '{}-{}.npy'.format(target_scene_name.split('.')[0], target_node)
    target_idx = file_name_to_idx[target_file_name]
    target_feature = features[target_idx, :]

    # read the box for the target object and find the translation that brings it to the origin.
    t_box = t_boxes[target_node]
    t_translation = -t_box.translation
    t_box_origin = translate_obbox(t_box, t_translation)

    # for each context object find the best candidate in the target object.
    correspondence = {}
    total_sim = IoU(t_box_origin, q_box_origin).iou()
    if args.include_embedding_sim:
        total_sim += torch.dot(query_feature, target_feature)
    for context_object in context_objects:
        highest_sim = 0
        best_candidate = None

        # translate the context object using the translation that brings the query object to the origin.
        q_c_box = q_boxes[context_object]
        q_c_box_translated = translate_obbox(q_c_box, q_translation)

        # read the embedding of the context object.
        context_file_name = '{}-{}.npy'.format(query_scene_name.split('.')[0], context_object)
        context_idx = file_name_to_idx[context_file_name]
        context_feature = features[context_idx, :]

        # the best candidate has IoU > 0 and highest embedding similarity.
        for candidate in target_scene.keys():
            # skip if the candidate is already assigned or is the target node.
            if (candidate in correspondence) or (candidate == target_node):
                continue

            # choose a match either based on predicted labels if available.
            candidate_file_name = '{}-{}.npy'.format(target_scene_name.split('.')[0], candidate)
            candidate_idx = file_name_to_idx[candidate_file_name]
            if predicted_labels is not None:
                if labels[context_idx] != predicted_labels[candidate_file_name]:
                    continue
            elif predicted_clusters is not None:
                if predicted_clusters[context_file_name] != predicted_clusters[candidate_file_name]:
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

            # include embedding similarity if necessary.
            if args.include_embedding_sim:
                # compute the embedding similarity between the context and candidate object
                candidate_feature = features[candidate_idx, :]
                current_sim = torch.dot(candidate_feature, context_feature)
                iou += current_sim

            if iou > highest_sim:
                highest_sim = iou
                best_candidate = candidate

        if best_candidate is not None:
            correspondence[best_candidate] = context_object
            total_sim += highest_sim

    target_subscene = {'scene_name': target_scene_name, 'target': target_node, 'correspondence': correspondence,
                       'context_match': len(correspondence), 'total_sim': float(total_sim)}

    return target_subscene


def find_best_target_subscenes(args, query_info, target_scene_names, features, labels, predicted_labels,
                               predicted_clusters, file_name_to_idx):
    # load the query info
    query_scene_name = query_info['example']['scene_name']
    query_node = query_info['example']['query']
    context_objects = query_info['example']['context_objects']
    query_scene = load_from_json(os.path.join(args.scene_dir, query_scene_name))
    q_boxes = load_boxes(query_scene, [query_node] + context_objects, box_type='obbox')

    # find the topk target nodes and their corresponding scenes.
    target_scene_to_nodes = find_topk_target_nodes(args, query_scene_name, target_scene_names, query_node, features,
                                                   labels, predicted_labels, predicted_clusters, file_name_to_idx)

    # find the best matching target subscenes.
    target_subscenes = []
    for target_scene_name in target_scene_to_nodes.keys():
        # skip if the target scene is the same as the query scene.
        if target_scene_name == query_scene_name:
            continue

        # TODO: rotate the target scene to best align with the query subscene

        # for each query object find the best corresponding object in the query scene.
        target_scene = load_from_json(os.path.join(args.scene_dir, target_scene_name))
        t_boxes = load_boxes(target_scene, target_scene.keys(), box_type='aabb')
        for target_node in target_scene_to_nodes[target_scene_name]:
            target_subscene = find_best_correspondence(args, query_scene_name, target_scene_name, target_scene,
                                                       query_node, target_node, context_objects, features, labels,
                                                       predicted_labels, predicted_clusters, file_name_to_idx, q_boxes,
                                                       t_boxes)
            target_subscenes.append(target_subscene)

    # rank the target object based on the highest number of correspondences and overall IoU.
    target_subscenes = sorted(target_subscenes, reverse=True, key=lambda x: (x['context_match'], x['total_sim']))

    return target_subscenes


def apply_3dssr(args):
    # make sure to retrieve data from the requested mode
    args.pc_dir = os.path.join(args.pc_dir, args.mode)
    args.scene_dir = os.path.join(args.scene_dir, args.mode)

    # set the input and output paths for the query dict.
    query_dict_input_path = os.path.join(args.query_dir, args.query_input_file_name)
    query_output_file_name = args.query_input_file_name.split('.')[0] + '_{}_{}.json'.format(args.mode,
                                                                                             args.experiment_name)
    query_dict_output_path = os.path.join(args.cp_dir, args.results_folder_name, query_output_file_name)

    # read the query dict.
    query_dict = load_from_json(query_dict_input_path)

    # find all the potential target scenes.
    target_scene_names = os.listdir(os.path.join(args.scene_dir))

    # load the features and file names.
    features_dir = os.path.join(args.cp_dir, args.results_folder_name, args.features_dir_name)
    features = torch.load(os.path.join(features_dir, "{}feat.pth".format(args.mode)))
    print('Loaded features of shape {}'.format(features.shape))

    # load labels and predicted labels ( if necessary).
    labels = torch.load(os.path.join(features_dir, "{}labels.pth".format(args.mode)))
    predicted_labels = None
    predicted_clusters = None
    if args.with_cat_predictions:
        predicted_labels = load_from_json(os.path.join(args.cp_dir, args.results_folder_name,
                                                       args.predicted_labels_file_name))
        print('Loaded predicted labels for {} files'.format(len(predicted_labels)))
    elif args.with_cluster_predictions:
        predicted_clusters = load_from_json(os.path.join(args.cp_dir, args.results_folder_name,
                                                         args.predicted_clusters_file_name))
        print('Loaded predicted clusters for {} files'.format(len(predicted_clusters)))

    # map the file names to feature indices.
    file_indices = torch.load(os.path.join(features_dir, "{}_file_names.pth".format(args.mode)))
    file_name_to_idx = map_file_names_to_idx(args, file_indices)

    # apply scene alignment for each query
    t0 = time()
    for i, (query, query_info) in enumerate(query_dict.items()):
        t = time()
        print('Processing query: {}'.format(query))
        target_subscenes = find_best_target_subscenes(args, query_info, target_scene_names, features, labels,
                                                      predicted_labels, predicted_clusters, file_name_to_idx)
        query_info['target_subscenes'] = target_subscenes
        duration = (time() - t) / 60
        print('Processing the query took {} minutes'.format(round(duration, 2)))
        print('*' * 50)

    duration_all = (time() - t0) / 60
    print('Processing all queries took {} minutes'.format(round(duration_all, 2)))

    # save the changes to query dict
    write_to_json(query_dict, query_dict_output_path)


