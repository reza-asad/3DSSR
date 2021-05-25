import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from time import time
from optparse import OptionParser

from scripts.helper import load_from_json, write_to_json
from alignment_dataset import Scene
from models import Lstm, CosSinRegressor
from scripts.box import Box


def translate_obbox(obbox, translation):
    # build the transformation matrix
    transformation = np.eye(4)
    transformation[:3, 3] = translation

    # apply tranlsation to the obbox
    obbox = obbox.apply_transformation(transformation)

    return obbox


def rotate_obbox(obbox, alpha, beta, gamma):
    # build the transformation matrix
    transformation = np.eye(4)
    rotation_z = np.asarray([[np.cos(alpha), -np.sin(alpha), 0],
                             [np.sin(alpha), np.cos(alpha), 0],
                             [0, 0, 1]], dtype=np.float128)
    rotation_y = np.asarray([[np.cos(beta), 0, np.sin(beta)],
                             [0, 1, 0],
                             [-np.sin(beta), 0, np.cos(beta)]], dtype=np.float128)
    rotation_x = np.asarray([[1, 0, 0],
                             [0, np.cos(gamma), -np.sin(gamma)],
                             [0, np.sin(gamma), np.cos(gamma)]], dtype=np.float128)
    rotation = np.matmul(np.matmul(rotation_z, rotation_y), rotation_x)
    transformation[:3, :3] = rotation

    # apply tranlsation to the obbox
    obbox = obbox.apply_transformation(transformation)

    return obbox


def compute_alignment_error(vertices1, vertices2):
    err = np.linalg.norm(vertices2 - vertices1)

    return err


def svd_rotation(query_obboxes, target_obboxes):
    # combine the vertices in the query subscene
    q_vertices = [query_obbox.vertices for query_obbox in query_obboxes]
    q_vertices = np.concatenate(q_vertices, axis=0)

    # combine the vertices in the target scene
    t_vertices = [target_obbox.vertices for target_obbox in target_obboxes]
    t_vertices = np.concatenate(t_vertices, axis=0)

    # use svd to find the rotation that best aligns the target scene with the query subscene
    COR = np.dot(t_vertices.transpose(), q_vertices)
    U, S, Vt = np.linalg.svd(COR)
    R = np.dot(Vt.transpose(), U.transpose())
    best_R = R.copy()
    rot_t_vertices = np.dot(best_R, t_vertices.transpose()).transpose()
    best_err = compute_alignment_error(rot_t_vertices, q_vertices)

    # If R is a reflection matrix look for best rotation.
    if np.linalg.det(R) < 0:
        best_err = np.float('inf')
        for i in range(3):
            U, S, Vt = np.linalg.svd(R)
            U[i, :] *= -1
            new_R = np.dot(Vt.transpose(), U.transpose())
            rot_t_vertices = np.dot(new_R, t_vertices.transpose()).transpose()
            curr_err = compute_alignment_error(rot_t_vertices, q_vertices)
            if curr_err < best_err:
                best_err = curr_err
                best_R = new_R.copy()

    return best_R, best_err


def find_best_rotation_svd(query_graph, target_graph, query_node, target_node, bp_graph, with_projection):
    # build obbox for the query object and translate it to the origin
    q_obj_to_obbox = {}
    query_obbox_vertices = np.asarray(query_graph[query_node]['obbox'][0])
    if with_projection:
        query_obbox_vertices[:, -1] = 0.0
    q_obbox = Box(query_obbox_vertices)
    q_translation = -q_obbox.translation
    q_obbox = translate_obbox(q_obbox, q_translation)
    q_obj_to_obbox[query_node] = q_obbox

    # build obbox for the target object and find its translation to the origin
    t_obj_to_obbox = {}
    target_obbox_vertices = np.asarray(target_graph[target_node]['obbox'])
    if with_projection:
        target_obbox_vertices[:, -1] = 0.0
    t_obbox = Box(target_obbox_vertices)
    t_translation = -t_obbox.translation
    t_obbox = translate_obbox(t_obbox, t_translation)
    t_obj_to_obbox[target_node] = t_obbox

    # for each context object pick the candidate that leads to the least error.
    for context_obj in bp_graph.keys():
        # translate the obbox of the context object using the vector that translates the query object to the origin.
        q_context_obbox_vertices = np.asarray(query_graph[context_obj]['obbox'][0])
        if with_projection:
            q_context_obbox_vertices[:, -1] = 0.0
        q_c_obbox = Box(q_context_obbox_vertices)
        q_c_obbox = translate_obbox(q_c_obbox, q_translation)
        q_obj_to_obbox[context_obj] = q_c_obbox

        # find the rotation and error for each candidate using svd
        candidate_to_err = {}
        for candidate in bp_graph[context_obj]['candidates']:
            # translate the obbox of the candidate object using a vector that translates the target object to the origin
            t_candidate_obbox_vertices = np.asarray(target_graph[candidate]['obbox'])
            if with_projection:
                t_candidate_obbox_vertices[:, -1] = 0.0
            t_c_obbox = Box(t_candidate_obbox_vertices)
            t_c_obbox = translate_obbox(t_c_obbox, t_translation)
            t_obj_to_obbox[candidate] = t_c_obbox

            # apply svd to compute the rotation and error for the pair of context and candidate
            _, err = svd_rotation(query_obboxes=[q_c_obbox], target_obboxes=[t_c_obbox])
            candidate_to_err[candidate] = err

        # pick the best candidate
        if len(candidate_to_err) > 0:
            match, lowest_err = sorted(candidate_to_err.items(), key=lambda x: x[1], reverse=False)[0]
            bp_graph[context_obj]['match'] = match

    # find the rotation that best aligns all the matching candidates with the query subscene.
    query_obboxes = [q_obj_to_obbox[query_node]]
    target_obboxes = [t_obj_to_obbox[target_node]]
    for context_obj in bp_graph.keys():
        if 'match' in bp_graph[context_obj]:
            query_obboxes.append(q_obj_to_obbox[context_obj])
            target_obboxes.append(t_obj_to_obbox[bp_graph[context_obj]['match']])

    rotation, _ = svd_rotation(query_obboxes=query_obboxes, target_obboxes=target_obboxes)

    return rotation


def apply_svd(data_dir, loader, mode, alpha, with_projection):
    errors = []
    for i, data in enumerate(loader):
        # load the query data.
        query_node = data['query_node'][0]
        query_graph = data['query_graph']

        # load the target data.
        target_node = query_node
        query_scene_name = data['file_name'][0]
        target_graph = load_from_json(os.path.join(data_dir, mode, query_scene_name))

        # load the bipartite graph and clean it.
        bp_graph = data['bp_graph']
        for context_object, candidates_info in bp_graph.items():
            clean_candidates = []
            for e in candidates_info['candidates']:
                clean_candidates.append(e[0])
            bp_graph[context_object]['candidates'] = clean_candidates

        # find the rotation that best aligns the target scene and with the query subscene using SVD.
        rotation_hat = find_best_rotation_svd(query_graph, target_graph, query_node, target_node, bp_graph,
                                              with_projection)
        alpha_hat = np.arctan2(rotation_hat[1, 0], rotation_hat[0, 0])

        # compute the error for each alpah
        alpha_hat = np.mod(alpha_hat + 2 * np.pi, 2 * np.pi)
        error = np.minimum(np.abs(alpha - alpha_hat), np.abs(alpha + 2 * np.pi - alpha_hat))
        errors.append(error * 180 / np.pi)

    return errors


def apply_alignment_module(loader, model_dic, device, alpha):
    errors = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            # load data
            features = data['features'].squeeze(dim=0)

            # TODO: remove this after you remove the match column
            features = features[:, :, :-1]
            features = features.to(device=device, dtype=torch.float32)

            # sort the features by their IoU
            indices = torch.sort(features[:, :, -1], descending=False)[1]
            sorted_features = features[:, indices[0], :]

            # apply the lstm model if there were at least one pair
            h = model_dic['lstm'].init_hidden(batch_size=1)
            output, h = model_dic['lstm'](sorted_features, h)

            # apply a fc layer to regress the output of the lstm
            mean_output = torch.mean(output, dim=1).unsqueeze_(dim=1)
            cos_sin_hat = model_dic['lin_layer'](mean_output)

            # bring the cos and sin back to cpu and covert to numpy array
            cos_sin_hat = cos_sin_hat.cpu().detach().numpy()[0, 0, :]

            # compute the predicted angle
            alpha_hat = np.arccos(cos_sin_hat[0])
            if cos_sin_hat[1] < 0:
                alpha_hat = 2 * np.pi - alpha_hat

            # compute the errro between the predicted and original angle
            error = np.minimum(np.abs(alpha - alpha_hat), np.abs(alpha + 2 * np.pi - alpha_hat))
            errors.append(error * 180/np.pi)

    return errors


def compute_alignment_errors(data_dir, num_queries, mode, model_dic, device):
    # partition the unit circle equally into 7 sectors.
    alphas = [(n * np.pi)/4.0 for n in range(8)]

    # rotate the query by alpha and predict the rotation using the alignment module and svd.
    alignment_errors = {'SVD1D': np.zeros((len(alphas), num_queries), dtype=np.float),
                        'SVD3D': np.zeros((len(alphas), num_queries), dtype=np.float),
                        'AlignmentModule': np.zeros((len(alphas), num_queries), dtype=np.float)}
    for i, alpha in enumerate(alphas):
        # create data data loader
        dataset = Scene(data_dir, num_queries, mode=mode, alpha=alpha)
        loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

        # apply the various alignment models on test data and compute the error.
        alignment_errors['AlignmentModule'][i, :] = apply_alignment_module(loader, model_dic, device, alpha)
        alignment_errors['SVD1D'][i, :] = apply_svd(data_dir, loader, mode, alpha, with_projection=True)
        alignment_errors['SVD3D'][i, :] = apply_svd(data_dir, loader, mode, alpha, with_projection=False)

    # compute the mean of the absolute alignment errors for each alignment model
    for alignment_model, errors in alignment_errors.items():
        alignment_errors[alignment_model] = np.mean(errors, axis=0).tolist()
    # print(alignment_errors['SVD3D'])
    return alignment_errors


def get_args():
    parser = OptionParser()
    parser.add_option('--mode', dest='mode', default='test', help='val or test')
    parser.add_option('--data-dir', dest='data_dir',
                      default='../../results/matterport3d/LearningBased/scene_graphs_cl_with_predictions_gnn')
    parser.add_option('--cp_dir', dest='cp_dir',
                      default='../../results/matterport3d/LearningBased/lstm_alignment')
    parser.add_option('--num_queries', dest='num_queries', default=50, type='int')
    parser.add_option('--experiment_name', dest='experiment_name', default='align_vs_svd')
    parser.add_option('--hidden_dim', dest='hidden_dim', default=512, type='int')
    parser.add_option('--input_dim', dest='input_dim', default=5)
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

    # set output path for the query dict.
    alignment_errors_path = '../../results/matterport3d/evaluations/ablation/alignment_error_{}.json'.\
        format(args.experiment_name)

    # apply scene alignment for each query
    t0 = time()
    alignment_errors = compute_alignment_errors(args.data_dir, args.num_queries, args.mode, model_dic, device)
    duration_all = (time() - t0) / 60
    print('Processing all queries took {} minutes'.format(round(duration_all, 2)))

    # save the changes to query dict
    write_to_json(alignment_errors, alignment_errors_path)


if __name__ == '__main__':
    main()

