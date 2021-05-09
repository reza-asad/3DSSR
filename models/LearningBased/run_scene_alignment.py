import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from time import time
from optparse import OptionParser

from scripts.helper import load_from_json, write_to_json
from scene_dataset import Scene
from models import Lstm, CosSinRegressor
from scripts.box import Box
from scripts.iou import IoU


def compute_angle(cos_sin):
    cos, sin = cos_sin
    angle = np.arccos(cos)
    if sin < 0:
        angle = 2 * np.pi - angle

    return angle


def translate_obbox(obbox, translation):
    # build the transformation matrix
    transformation = np.eye(4)
    transformation[:3, 3] = translation

    # apply tranlsation to the obbox
    obbox = obbox.apply_transformation(transformation)

    return obbox


def examine_target_query_overlap(data_dir, mode, query_graph, query_node, target_scene_name, target_node,
                                 context_candidates, cos_sin_hat):
    # load the query and target graphs
    # query_graph = load_from_json(os.path.join(data_dir, mode, query_scene_name))
    target_graph = load_from_json(os.path.join(data_dir, mode, target_scene_name))

    # build obbox for the query object and translate it to the origin
    query_obbox_vertices = np.asarray(query_graph[query_node]['obbox'])
    q_obbox = Box(query_obbox_vertices)
    q_translation = -q_obbox.translation
    q_obbox = translate_obbox(q_obbox, q_translation)

    # build obbox for the target object and find its translation to the origin
    target_obbox_vertices = np.asarray(target_graph[target_node]['obbox'])
    t_obbox = Box(target_obbox_vertices)
    t_translation = -t_obbox.translation
    t_obbox = translate_obbox(t_obbox, t_translation)

    # find the rotation angle from its sine and cosine.
    delta_angle = compute_angle(cos_sin_hat)
    # compute the rotation matrix
    transformation = np.eye(4)
    rotation = np.asarray([[np.cos(delta_angle), -np.sin(delta_angle), 0],
                           [np.sin(delta_angle), np.cos(delta_angle), 0],
                           [0, 0, 1]])
    transformation[:3, :3] = rotation

    # rotate the target obbox and compute its iou with the query obbox
    t_obbox = t_obbox.apply_transformation(transformation)
    t_q_iou = IoU(t_obbox, q_obbox).iou()

    # record the candidates for each context object, rotate them using the predicted angle and record the IoU.
    candidate_to_obbox = {}
    bp_graph = {'context_candidates': {}, 't_q_iou': t_q_iou, 'theta': delta_angle}
    for context_obj, candidates in context_candidates.items():
        # record the candidates
        bp_graph['context_candidates'][context_obj] = {'candidates': candidates, 'obboxes': [], 'IoUs': []}

        # create the obbox of the context object in the query scene.
        q_context_obbox_vertices = np.asarray(query_graph[context_obj]['obbox'])
        q_c_obbox = Box(q_context_obbox_vertices)
        q_c_obbox = translate_obbox(q_c_obbox, q_translation)

        # record the obboxes and IoUs for each candidate
        for candidate in candidates:
            # create the obbox of the candidate object and translate according to the translation of the target node.
            if candidate not in candidate_to_obbox:
                t_candidate_obbox_vertices = np.asarray(target_graph[candidate]['obbox'])
                t_c_obbox = Box(t_candidate_obbox_vertices)
                t_c_obbox = translate_obbox(t_c_obbox, t_translation)

                # rotate the candidate objects based on the predicted rotation angle
                candidate_to_obbox[candidate] = t_c_obbox.apply_transformation(transformation)

            # record the rotated obboxes for each candidate
            bp_graph['context_candidates'][context_obj]['obboxes'] += [candidate_to_obbox[candidate]]

            # record the IoU between the context object and each candidate
            try:
                iou = IoU(candidate_to_obbox[candidate], q_c_obbox).iou()
            except Exception:
                iou = 0
            bp_graph['context_candidates'][context_obj]['IoUs'] += [iou]

    return bp_graph


def find_correspondence(target_scene_name, target_node, bp_graph):

    # read the iou between the query and target obboxes
    overal_iou = bp_graph['t_q_iou']

    # traverse the bipartite graph and pick the best candidates based on iou
    correspondence = {}
    for context_object, candidate_info in bp_graph['context_candidates'].items():
        best_iou = 0
        best_candidate = None
        for i, candidate in enumerate(candidate_info['candidates']):
            # pick the candidate with the highest IoU that is not already visited
            if (candidate_info['IoUs'][i] > best_iou) and (candidate not in correspondence):
                best_iou = candidate_info['IoUs'][i]
                best_candidate = candidate

        # if a candidate with IoU greater than 0 was found, record the correspondence
        if best_candidate is not None:
            correspondence[best_candidate] = context_object
            overal_iou += best_iou

    target_subscene = {'scene_name': target_scene_name, 'target': target_node, 'correspondence': correspondence,
                       'context_match': len(correspondence), 'IoU': overal_iou, 'theta': float(bp_graph['theta'])}

    return target_subscene


def find_best_target_subscenes(query_info, data_dir, mode, model_dic, device, with_cat_predictions=False,
                               with_clustering=False, with_alignment=True, q_theta=0):
    # load the query info
    query_scene_name = query_info['example']['scene_name']
    query_node = query_info['example']['query']
    context_objects = query_info['example']['context_objects']

    # create data data loader
    dataset = Scene(data_dir, mode=mode, query_scene_name=query_scene_name, q_context_objects=context_objects,
                    query_node=query_node, with_cat_predictions=with_cat_predictions, with_clustering=with_clustering,
                    q_theta=q_theta)
    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    # apply the models to valid/test data
    target_subscenes = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            # if i not in [0]:
            #     continue
            # load data
            all_features = data['features']
            target_scene_name = data['file_name'][0]
            target_nodes = [e[0] for e in data['target_nodes']]

            # if no target nodes found or the target scene is the same as query, skip.
            if len(target_nodes) == 0 or target_scene_name == query_scene_name:
                continue

            # read the context candidates and clean them
            raw_bp_graphs = data['bp_graphs']
            bp_graphs = {target_node: {} for target_node in raw_bp_graphs.keys()}
            for target_node, context_to_candidates in raw_bp_graphs.items():
                for context_object, candidates in context_to_candidates.items():
                    clean_candidates = []
                    for e in candidates:
                        clean_candidates.append(e[0])
                    bp_graphs[target_node][context_object] = clean_candidates

            # sort the features by their IoU
            # for e in all_features:
            #     print(e.shape)
            for j in range(len(target_nodes)):
                # TODO: remove this after you remove the match column
                features = all_features[j][:, :, :-1]
                features = features.to(device=device, dtype=torch.float32)

                # sort the features by their IoU
                indices = torch.sort(features[:, :, -1], descending=False)[1]
                sorted_features = features[:, indices[0], :]

                # apply the lstm model if there were at least one pair
                cos_sin_hat = [1.0, 0.0]
                if features.shape[1] > 0:
                    # if alignment is disabled set theta to 0
                    if with_alignment:
                        h = model_dic['lstm'].init_hidden(batch_size=1)
                        output, h = model_dic['lstm'](sorted_features, h)

                        # apply a fc layer to regress the output of the lstm
                        mean_output = torch.mean(output, dim=1).unsqueeze_(dim=1)
                        cos_sin_hat = model_dic['lin_layer'](mean_output)

                        # bring the cos and sin back to cpu and covert to numpy array
                        cos_sin_hat = cos_sin_hat.cpu().detach().numpy()[0, 0, :]

                # rotate the target scene using predicted rotation
                bp_graph = examine_target_query_overlap(data_dir, mode, dataset.query_graph, query_node,
                                                        target_scene_name, target_nodes[j], bp_graphs[target_nodes[j]],
                                                        cos_sin_hat)

                # find the corresponding objects between the rotated target scene and the query scene. record the
                # overall IoU and the correspondence.
                target_subscene = find_correspondence(target_scene_name, target_nodes[j], bp_graph)
                target_subscenes.append(target_subscene)

    # rank the target object based on the highest number of correspondences and overall IoU.
    target_subscenes = sorted(target_subscenes, reverse=True, key=lambda x: (x['context_match'], x['IoU']))

    return target_subscenes


def get_args():
    parser = OptionParser()
    parser.add_option('--mode', dest='mode', default='test', help='val or test')
    parser.add_option('--data-dir', dest='data_dir',
                      default='../../results/matterport3d/LearningBased/scene_graphs_cl_with_predictions_kmeans')
    parser.add_option('--cp_dir', dest='cp_dir',
                      default='../../results/matterport3d/LearningBased/lstm_alignment_no_cats_kmeans')
    parser.add_option('--experiment_name', dest='experiment_name', default='lstm_top1_predictions_kmeans')
    parser.add_option('--hidden_dim', dest='hidden_dim', default=512, type='int')
    parser.add_option('--input_dim', dest='input_dim', default=5)
    parser.add_option('--with_cat_predictions', dest='with_cat_predictions', default=False)
    parser.add_option('--with_clustering', dest='with_clustering', default=True)
    parser.add_option('--with_alignment', dest='with_alignment', default=True)
    parser.add_option('--q_theta', dest='q_theta', default=0*np.pi/4)
    parser.add_option('--gpu', action='store_true', dest='gpu', default=True, help='use cuda')

    (options, args) = parser.parse_args()
    return options


def main():
    # get the arguments
    args = get_args()

    # Set the right device for all the models
    device = torch.device('cpu')
    if args.gpu:
        device = torch.device('cuda')

    # initialize the models and set them on the right device
    lstm = Lstm(args.input_dim, args.hidden_dim, device)
    lin_layer = CosSinRegressor(args.hidden_dim)
    lstm = lstm.to(device=device)
    lin_layer = lin_layer.to(device=device)

    # load the models and set the models on evaluation mode
    model_dic = {'lstm': lstm, 'lin_layer': lin_layer}
    for model_name, model in model_dic.items():
        model.load_state_dict(torch.load(os.path.join(args.cp_dir, 'CP_{}_best.pth'.format(model_name))))
        model.eval()

    # set the input and output paths for the query dict.
    query_dict_input_path = '../../queries/matterport3d/query_dict_{}.json'.format(args.mode)
    query_dict_output_path = '../../results/matterport3d/LearningBased/query_dict_{}_{}.json'.format(args.mode,
                                                                                                     args.experiment_name)

    # read the query dict
    query_dict = load_from_json(query_dict_input_path)

    # apply scene alignment for each query
    t0 = time()
    for i, (query, query_info) in enumerate(query_dict.items()):
        t = time()
        print('Iteration {}/{}'.format(i+1, len(query_dict)))
        print('Processing query: {}'.format(query))
        target_subscenes = find_best_target_subscenes(query_info, args.data_dir, args.mode, model_dic, device,
                                                      args.with_cat_predictions, args.with_clustering,
                                                      args.with_alignment, args.q_theta)
        query_info['target_subscenes'] = target_subscenes
        duration = (time() - t) / 60
        print('Processing the query took {} minutes'.format(round(duration, 2)))
    duration_all = (time() - t0) / 60
    print('Processing all queries too {} minutes'.format(round(duration_all, 2)))

    # save the changes to query dict
    write_to_json(query_dict, query_dict_output_path)


if __name__ == '__main__':
    main()

