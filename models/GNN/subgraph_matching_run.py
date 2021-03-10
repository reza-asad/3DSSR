import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter
from time import time

from scripts.helper import load_from_json, write_to_json
from scene_dataset import Scene


def apply_ring_gnn(query_info, data_dir, mode):
    # load the query info
    query_scene_name = query_info['example']['scene_name']
    query_node = query_info['example']['query']
    context_objects = query_info['example']['context_objects']

    # create data loader
    dataset = Scene(data_dir, mode=mode, query_scene_name=query_scene_name, q_context_objects=context_objects,
                    query_node=query_node)
    #TODO: change num workesrs
    loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

    # apply the models to valid/test data
    target_subscenes = []
    for i, data in enumerate(loader):
        # read the data
        bp_graphs = data['bp_graphs']
        file_name = data['file_name'][0]
        target_nodes = [e[0] for e in data['target_nodes']]

        # skip if the file name is the same as the query scene name
        if file_name == query_scene_name:
            continue

        # For each target object find the best pair (t, c) interms of IoU
        target_node_candidates = {}
        for j, target_node in enumerate(target_nodes):
            # record the context objects and IoU pairs given each target object
            t_context_objects = []
            best_iou = 0
            best_iou_obj = None
            for q_context, q_context_info in bp_graphs[j].items():
                t_context, iou = q_context_info['match']
                if t_context != -1:
                    t_context_objects.append(t_context[0])
                    if iou > best_iou:
                        best_iou = iou.item()
                        best_iou_obj = t_context[0]
            target_node_candidates[target_node] = {'context_objects': t_context_objects, 'best_iou': best_iou,
                                                   'best_iou_obj': best_iou_obj}

        # pick the target object with highest IoU for the pair (t, c)
        best_target_node = None
        best_context_objects = []
        best_iou = -1
        best_iou_obj = None
        for target_node, target_node_info in target_node_candidates.items():
            if target_node_info['best_iou'] > best_iou:
                best_iou = target_node_info['best_iou']
                best_iou_obj = target_node_info['best_iou_obj']
                best_target_node = target_node
                best_context_objects = target_node_info['context_objects']

        # test if there are duplicates in the context objects
        context_to_count = Counter(best_context_objects)
        for _, count in context_to_count.items():
            if count > 1:
                print(best_context_objects)
                raise Exception('Duplicate context objects detected')

        target_subscene = {'scene_name': file_name, 'target': best_target_node, 'context_objects': best_context_objects,
                           'IoU': best_iou, 'best_iou_obj': best_iou_obj, 'context_match': len(best_context_objects)}
        target_subscenes.append(target_subscene)

    # sort the target subscenes based on their best IoU pair (t, c)
    target_subscenes = sorted(target_subscenes, reverse=True, key=lambda x: (x['context_match'], x['IoU']))

    return target_subscenes


def main():
    mode = 'val'
    experiment_name = 'dev'

    query_dict_input_path = '../../queries/matterport3d/query_dict_{}.json'.format(mode)
    query_dict_output_path = '../../results/matterport3d/GNN/query_dict_{}_{}.json'.format(mode, experiment_name)
    ring_data_dir = '../../results/matterport3d/GNN/scene_graphs_cl'

    # read the query dict
    query_dict = load_from_json(query_dict_input_path)

    # apply subgraph matching for each query
    t0 = time()
    for i, (query, query_info) in enumerate(query_dict.items()):
        t = time()
        print('Iteration {}/{}'.format(i+1, len(query_dict)))
        print('Processing query: {}'.format(query))
        target_subscenes = apply_ring_gnn(query_info, ring_data_dir, mode)
        query_info['target_subscenes'] = target_subscenes
        duration = (time() - t) / 60
        print('Processing the query took {} minutes'.format(round(duration, 2)))
    duration_all = (time() - t0) / 60
    print('Processing all queries too {} minutes'.format(round(duration_all, 2)))

    # save the changes to query dict
    write_to_json(query_dict, query_dict_output_path)


if __name__ == '__main__':
    main()

