import os
import numpy as np
from time import time
from optparse import OptionParser

from scripts.helper import load_from_json, write_to_json
from scripts.box import Box
from scripts.iou import IoU


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


def map_cat_to_objects(query_cats, target_graph, target_node, with_cat_predictions=False):
    cat_to_objects = {cat: [] for cat in query_cats}
    for node, node_info in target_graph.items():
        if node != target_node:
            if with_cat_predictions:
                target_cat = node_info['category_predicted'][0]
            else:
                target_cat = node_info['category'][0]
            if target_cat in cat_to_objects:
                cat_to_objects[target_cat].append(node)

    return cat_to_objects


def build_bipartite_graph(query_graph, context_objects, cat_to_objects):
    bp_graph = {}
    # for each context object find its candidates from the target scene with the same category.
    for context_obj in context_objects:
        context_obj_cat = query_graph[context_obj]['category'][0]
        bp_graph[context_obj] = {'candidates': cat_to_objects[context_obj_cat]}

    return bp_graph


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
    query_obbox_vertices = np.asarray(query_graph[query_node]['obbox'])
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
        q_context_obbox_vertices = np.asarray(query_graph[context_obj]['obbox'])
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


def examine_target_query_overlap(data_dir, mode, query_graph, query_node, target_scene_name, target_node,
                                 bp_graph, rotation, with_projection=False):

    # extract the euler angles from the 3D rotation matrix
    alpha = np.arctan2(rotation[1, 0], rotation[0, 0])
    beta = np.arctan2(-rotation[2, 0], np.sqrt((rotation[2, 1]**2 + rotation[2, 2]**2)))
    gamma = np.arctan2(rotation[2, 1], rotation[2, 2])

    # sanity check that the euler angles build the original rotation matrix
    rotation_z = np.asarray([[np.cos(alpha), -np.sin(alpha), 0],
                             [np.sin(alpha), np.cos(alpha), 0],
                             [0, 0, 1]], dtype=np.float64)
    rotation_y = np.asarray([[np.cos(beta), 0, np.sin(beta)],
                             [0, 1, 0],
                             [-np.sin(beta), 0, np.cos(beta)]], dtype=np.float64)
    rotation_x = np.asarray([[1, 0, 0],
                             [0, np.cos(gamma), -np.sin(gamma)],
                             [0, np.sin(gamma), np.cos(gamma)]], dtype=np.float64)
    if with_projection:
        rotation_x = np.eye(3)
        rotation_y = np.eye(3)
        beta = 0.0
        gamma = 0.0
        rotation = np.matmul(np.matmul(rotation_z, rotation_y), rotation_x)
    else:
        rotation_hat = np.matmul(np.matmul(rotation_z, rotation_y), rotation_x)
        if np.linalg.norm(rotation_hat - rotation) > 0.01:
            raise Exception

    # load the target graph.
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

    # apply the computed rotation to the target scene.
    transformation = np.eye(4)
    transformation[:3, :3] = rotation

    # rotate the target obbox and compute its iou with the query obbox
    t_obbox = t_obbox.apply_transformation(transformation)
    t_q_iou = IoU(t_obbox, q_obbox).iou()

    # record the candidates for each context object, rotate them using the predicted angle and record the IoU.
    candidate_to_obbox = {}
    updated_bp_graph = {'candidates_info': {}, 't_q_iou': t_q_iou, 'alpha': alpha, 'beta': beta, 'gamma': gamma}
    for context_obj, candidates_info in bp_graph.items():
        # record the candidates
        updated_bp_graph['candidates_info'][context_obj] = {'candidates': candidates_info['candidates'], 'obboxes': [],
                                                            'IoUs': []}

        # create the obbox of the context object in the query scene.
        q_context_obbox_vertices = np.asarray(query_graph[context_obj]['obbox'])
        q_c_obbox = Box(q_context_obbox_vertices)
        q_c_obbox = translate_obbox(q_c_obbox, q_translation)

        # record the obboxes and IoUs for each candidate
        for candidate in candidates_info['candidates']:
            # create the obbox of the candidate object and translate according to the translation of the target node.
            if candidate not in candidate_to_obbox:
                t_candidate_obbox_vertices = np.asarray(target_graph[candidate]['obbox'])
                t_c_obbox = Box(t_candidate_obbox_vertices)
                t_c_obbox = translate_obbox(t_c_obbox, t_translation)

                # rotate the candidate objects based on the predicted rotation angle
                candidate_to_obbox[candidate] = t_c_obbox.apply_transformation(transformation)

            # record the rotated obboxes for each candidate
            updated_bp_graph['candidates_info'][context_obj]['obboxes'] += [candidate_to_obbox[candidate]]

            # record the IoU between the context object and each candidate
            try:
                iou = IoU(candidate_to_obbox[candidate], q_c_obbox).iou()
            except Exception:
                iou = 0
            updated_bp_graph['candidates_info'][context_obj]['IoUs'] += [iou]

    return updated_bp_graph


def find_correspondence(target_scene_name, target_node, bp_graph):

    # read the iou between the query and target obboxes
    overal_iou = bp_graph['t_q_iou']

    # traverse the bipartite graph and pick the best candidates based on iou
    correspondence = {}
    for context_object, candidate_info in bp_graph['candidates_info'].items():
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
                       'context_match': len(correspondence), 'IoU': overal_iou, 'alpha': float(bp_graph['alpha']),
                       'beta': float(bp_graph['beta']), 'gamma': float(bp_graph['gamma'])}

    return target_subscene


def find_best_target_subscenes(query_info, data_dir, mode, with_cat_predictions=False, with_projection=False):
    # load the query info
    query_scene_name = query_info['example']['scene_name']
    query_node = query_info['example']['query']
    context_objects = query_info['example']['context_objects']

    # load the query graph and find the category of the query objects
    query_graph = load_from_json(os.path.join(data_dir, mode, query_scene_name))
    query_cat = query_graph[query_node]['category'][0]
    query_cats = [query_graph[c]['category'][0] for c in context_objects]

    # align each target scene with the query subscene and find the corresponding objects.
    target_subscenes = []
    target_scene_names = os.listdir(os.path.join(data_dir, mode))
    for i, target_scene_name in enumerate(target_scene_names):
        # if target scene is the same as query, skip.
        if target_scene_name == query_scene_name:
            continue

        # load the target scene and find the target objects matching the query object.
        target_graph = load_from_json(os.path.join(data_dir, mode, target_scene_name))
        if with_cat_predictions:
            target_nodes = [n for n in target_graph.keys() if query_cat == target_graph[n]['category_predicted'][0]]
        else:
            target_nodes = [n for n in target_graph.keys() if query_cat == target_graph[n]['category'][0]]

        # for each matching target object build a bipartite graph mapping query objects to target objects (based on cat)
        for target_node in target_nodes:
            # map the category of each context object to the object ids in the target graph.
            cat_to_objects = map_cat_to_objects(query_cats, target_graph, target_node,
                                                with_cat_predictions=with_cat_predictions)
            bp_graph = build_bipartite_graph(query_graph, context_objects, cat_to_objects)

            # find the rotation that best aligns the target scene and with the query subscene using SVD.
            rotation = find_best_rotation_svd(query_graph, target_graph, query_node, target_node, bp_graph,
                                              with_projection)

            # rotate the target scene using predicted rotation
            bp_graph = examine_target_query_overlap(data_dir, mode, query_graph, query_node, target_scene_name,
                                                    target_node, bp_graph, rotation, with_projection)

            # find the corresponding objects between the rotated target scene and the query scene. record the
            # overall IoU and the correspondence.
            target_subscene = find_correspondence(target_scene_name, target_node, bp_graph)
            target_subscenes.append(target_subscene)

    # rank the target object based on the highest number of correspondences and overall IoU.
    target_subscenes = sorted(target_subscenes, reverse=True, key=lambda x: (x['context_match'], x['IoU']))

    return target_subscenes


def rotate_scene(graph, source_node, alpha, beta, gamma):
    source_obbox_vertices = np.asarray(graph[source_node]['obbox'])
    source_obbox = Box(source_obbox_vertices)
    source_translation = -source_obbox.translation
    source_obbox = translate_obbox(source_obbox, source_translation)
    source_obbox = rotate_obbox(source_obbox, alpha, beta, gamma)
    graph[source_node]['obbox'] = source_obbox.vertices.tolist()

    for node, node_info in graph.items():
        if node != source_node:
            nb_obbox_vertices = np.asarray(graph[node]['obbox'])
            nb_obbox = Box(nb_obbox_vertices)
            nb_obbox = translate_obbox(nb_obbox, source_translation)
            nb_obbox = rotate_obbox(nb_obbox, alpha, beta, gamma)
            graph[node]['obbox'] = nb_obbox.vertices.tolist()

    return graph


def get_args():
    parser = OptionParser()
    parser.add_option('--mode', dest='mode', default='test', help='val or test')
    parser.add_option('--data-dir', dest='data_dir',
                      default='../../results/matterport3d/LearningBased/scene_graphs_with_predictions_gnn')
    parser.add_option('--experiment_name', dest='experiment_name', default='SVDRank1D')
    parser.add_option('--with_projection', dest='with_projection', default=True)
    parser.add_option('--with_cat_predictions', dest='with_cat_predictions', default=True)

    (options, args) = parser.parse_args()
    return options


def main():
    # get the arguments
    args = get_args()

    # set the input and output paths for the query dict.
    query_dict_input_path = '../../queries/matterport3d/query_dict_{}.json'.format(args.mode)
    query_dict_output_path = '../../results/matterport3d/SVDRank/query_dict_{}_{}.json'.format(args.mode,
                                                                                               args.experiment_name)

    # read the query dict
    query_dict = load_from_json(query_dict_input_path)

    # apply scene alignment for each query
    t0 = time()
    for i, (query, query_info) in enumerate(query_dict.items()):
        t = time()
        print('Iteration {}/{}'.format(i+1, len(query_dict)))
        print('Processing query: {}'.format(query))
        target_subscenes = find_best_target_subscenes(query_info, args.data_dir, args.mode,
                                                      args.with_cat_predictions, args.with_projection)
        query_info['target_subscenes'] = target_subscenes
        duration = (time() - t) / 60
        print('Processing the query took {} minutes'.format(round(duration, 2)))
    duration_all = (time() - t0) / 60
    print('Processing all queries too {} minutes'.format(round(duration_all, 2)))

    # save the changes to query dict
    write_to_json(query_dict, query_dict_output_path)


if __name__ == '__main__':
    main()

