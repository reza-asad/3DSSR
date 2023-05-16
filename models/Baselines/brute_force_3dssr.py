import os
from time import time
import argparse
import numpy as np
import trimesh

from scripts.helper import load_from_json, write_to_json
from scripts.box import Box
import torch
from chamferdist import ChamferDistance


def transform_pc(pc, translation, rotation):
    transformation = np.eye(4)
    transformation[:3, 3] = translation
    transformation[:3, :3] = rotation

    new_pc = np.ones((len(pc), 4), dtype=np.float32)
    new_pc[:, 0] = pc[:, 0]
    new_pc[:, 1] = pc[:, 1]
    new_pc[:, 2] = pc[:, 2]

    new_pc = np.dot(transformation, new_pc.T).T
    new_pc = new_pc[:, :3]

    return new_pc


def rotate_box(box, rotation):
    # build the transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = rotation

    # apply tranlsation to the obbox
    return box.apply_transformation(transformation)


def translate_box(box, translation):
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


def find_query_candidates(anchor, query_objects, query_scene, target_scene):
    # map the categories in the target scene to the object ids.
    cat_to_target_objects = {}
    for obj, obj_info in target_scene.items():
        cat = obj_info['category'][0]
        if cat not in cat_to_target_objects:
            cat_to_target_objects[cat] = [obj]
        else:
            cat_to_target_objects[cat].append(obj)

    # map each query object to the target objects with same category
    q_to_candidates = {}
    for q_obj in query_objects:
        q_cat = query_scene[q_obj]['category'][0]
        if q_cat in cat_to_target_objects:
            q_to_candidates[q_obj] = cat_to_target_objects[q_cat]

    # return empty dict if the anchor has not match.
    if anchor not in q_to_candidates:
        return {}

    return q_to_candidates


def find_all_permutations(q_objects):

    def permutations(sofar, remaining, results):
        if len(remaining) == 0:
            results.append(sofar.copy())
            return results

        for i in range(len(remaining)):
            sofar.append(remaining[i])
            results = permutations(sofar, remaining[:i] + remaining[i+1:], results)
            sofar.remove(remaining[i])

        return results

    q_objects_permutations = permutations([], q_objects, [])

    return q_objects_permutations


def find_corresponding_objects_groups(q_objects, q_to_candidates):

    def find_combinations(curr_correspondence, q_index, visited, corr_groups):
        # take the next q if there is any left.
        if q_index < len(q_objects):
            # load the current query object.
            q = q_objects[q_index]

            took_candidate = False
            for candidate in q_to_candidates[q]:
                # take the candidate if it's not already taken
                if candidate not in visited:
                    took_candidate = True
                    curr_correspondence[q] = candidate
                    visited.add(candidate)

                    # recurse
                    corr_groups = find_combinations(curr_correspondence, q_index+1, visited, corr_groups)

                    # pop out the candidate to try another one.
                    curr_correspondence.pop(q)
                    visited.remove(candidate)
            if not took_candidate:
                corr_groups = find_combinations(curr_correspondence, q_index + 1, visited, corr_groups)
        else:
            corr_groups.append(curr_correspondence.copy())
            return corr_groups

        return corr_groups

    corresponding_object_groups = []
    corresponding_object_groups = find_combinations({}, 0, set(), corresponding_object_groups)

    return corresponding_object_groups


def dedup_corr_groups(corresponding_object_groups):

    def build_key(g):
        sorted_g = sorted(g.items())
        key = []
        for (k, v) in sorted_g:
            key.append(k)
            key.append(v)
        key = '-'.join(key)

        return key

    visited = set()
    # hash corresponding objects to string keys and take them if they are already not taken.
    deduped_corr_groups = []
    for group in corresponding_object_groups:
        group_key = build_key(group)
        if group_key not in visited:
            visited.add(group_key)
            deduped_corr_groups.append(group)

    return deduped_corr_groups


def project_3D_box_to_2D(obj_to_box):
    # box_vis_list_before = []
    # for obj, box in obj_to_box.items():
    #     box_viz = trimesh.creation.box(box.scale, box.transformation)
    #     box_viz.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#0000ff")
    #     box_vis_list_before.append(box_viz)
    # axis = trimesh.creation.axis(transform=np.eye(4))
    # box_vis_list_before.append(axis)
    # trimesh.Scene(box_vis_list_before).show()

    # project each 3D box to 2D.
    obj_to_2D_box = {}
    for obj, box in obj_to_box.items():
        vertices_2d = np.round(box.vertices[1:, :-1], 6)
        obj_to_2D_box[obj] = np.unique(vertices_2d, axis=0)

    return obj_to_2D_box


def translate_subscene(correspondence, anchor, obj_to_box, query=True):
    if query:
        filtered_obj_to_box = {obj: box for obj, box in obj_to_box.items() if obj in correspondence}
    else:
        anchor = correspondence[anchor]
        filtered_obj_to_box = {obj: box for obj, box in obj_to_box.items() if obj in correspondence.values()}

    # find the translation that brings the anchor to the origin and translate it.
    anchor_translation = -filtered_obj_to_box[anchor].translation
    filtered_obj_to_box[anchor] = translate_box(filtered_obj_to_box[anchor], anchor_translation)

    # translate the context objects accordingly.
    context_objects = [obj for obj in filtered_obj_to_box.keys() if obj != anchor]
    for context_object in context_objects:
        # translate the context object.
        filtered_obj_to_box[context_object] = translate_box(filtered_obj_to_box[context_object], anchor_translation)

    # project 3D boxes to 2D.
    obj_to_2D_box = project_3D_box_to_2D(filtered_obj_to_box)

    # combine the points
    N = len(obj_to_2D_box)
    points = np.zeros((2, N * 4), dtype=np.float32)
    for i, obj in enumerate(obj_to_2D_box):
        points[:, i*4: (i+1)*4] = obj_to_2D_box[obj].T

    return points, filtered_obj_to_box


def align(q_points, t_points):
    # find the covariance matrix between p and q
    H = t_points @ np.transpose(q_points)
    if np.linalg.matrix_rank(H) != 2:
        raise ValueError('Covariance matrix should have 2 but has rank {}'.format(np.linalg.matrix_rank(H)))
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    theta = np.arctan2(R[1, 0], R[0, 0])

    return theta


def compute_radial_error(q_anchor_box, q_context_box, t_anchor_box, t_candidate_box, normalization_const):
    r_q = np.linalg.norm(q_context_box.translation - q_anchor_box.translation)
    r_t = np.linalg.norm(t_candidate_box.translation - t_anchor_box.translation)

    return np.abs(r_q - r_t) / normalization_const


def compute_angular_error(q_anchor_box, q_context_box, t_anchor_box, t_candidate_box):
    # find the vector connecting obj to anchor.
    v_q = q_context_box.translation - q_anchor_box.translation
    v_t = t_candidate_box.translation - t_anchor_box.translation

    # find the angle for each vector.
    theta_q = np.arctan2(v_q[0], v_q[1])
    theta_t = np.arctan2(v_t[0], v_t[1])

    return np.abs(theta_t - theta_q) / (2 * np.pi)


def compute_error(args, correspondence, q_anchor, q_to_box_filtered, t_to_box_filtered, theta):
    # set up the rotation matrix
    rotation = np.asarray([[np.cos(theta), -np.sin(theta), 0],
                           [np.sin(theta), np.cos(theta), 0],
                           [0, 0, 1]], dtype=np.float32)

    # rotate the target subscene to align with the query subscene.
    context_objects = [q for q in correspondence.keys() if q != q_anchor]
    radial_error, angular_error = 0, 0
    for context_object in context_objects:
        t_anchor = correspondence[q_anchor]
        candidate = correspondence[context_object]

        # rotate the target box.
        t_to_box_filtered[candidate] = rotate_box(t_to_box_filtered[candidate], rotation)

        # compute the radial error.
        radial_error += compute_radial_error(q_to_box_filtered[q_anchor], q_to_box_filtered[context_object],
                                             t_to_box_filtered[t_anchor], t_to_box_filtered[candidate],
                                             args.max_coord_scene)

        # compute the angular error.
        angular_error += compute_angular_error(q_to_box_filtered[q_anchor], q_to_box_filtered[context_object],
                                               t_to_box_filtered[t_anchor], t_to_box_filtered[candidate])

    return angular_error, radial_error


def compute_chamfer_error(args, query_scene_name, target_scene_name, correspondence, theta):
    # set up the rotation matrix.
    rotation = np.asarray([[np.cos(theta), -np.sin(theta), 0],
                           [np.sin(theta), np.cos(theta), 0],
                           [0, 0, 1]], dtype=np.float32)

    # load the pc for each corresponding object and compute the chamfer distance.
    chamferDist = ChamferDistance()
    chamfer_error = 0
    for q, t in correspondence.items():
        # load pc
        q_file_name = '{}-{}.npy'.format(query_scene_name.split('.')[0], q)
        t_file_name = '{}-{}.npy'.format(target_scene_name.split('.')[0], t)
        pc_q = np.load(os.path.join(args.pc_dir, q_file_name))
        pc_t = np.load(os.path.join(args.pc_dir, t_file_name))

        # sample points.
        np.random.seed(0)
        sampled_indices = np.random.choice(range(len(pc_q)), args.num_points, replace=False)
        pc_q = np.expand_dims(pc_q[sampled_indices, :], axis=0)
        pc_q = torch.from_numpy(pc_q).cuda()

        np.random.seed(0)
        sampled_indices = np.random.choice(range(len(pc_t)), args.num_points, replace=False)
        pc_t = pc_t[sampled_indices, :]

        # rotate the target pc by theta
        pc_t = transform_pc(pc_t, np.zeros(3), rotation)

        # compute chamfer distance.
        pc_t = np.expand_dims(pc_t, axis=0)
        pc_t = torch.from_numpy(pc_t).cuda()
        dist = chamferDist(pc_q, pc_t, bidirectional=True).item()

        # normalize the distance
        chamfer_error += (dist / args.cd_norm_constant[q_file_name][args.cd_threshold])

    return chamfer_error


def find_best_target_subscenes(args, query_info, target_scene_names):
    # load the query info
    query_scene_name = query_info['example']['scene_name']
    query_node = query_info['example']['query']
    context_objects = query_info['example']['context_objects']
    query_scene = load_from_json(os.path.join(args.scene_dir, query_scene_name))
    q_to_box = load_boxes(query_scene, [query_node] + context_objects, box_type='aabb')

    # for each target scene find the potential group of objects corresponding to query subscene and align them.
    target_subscenes = []
    for target_scene_name in target_scene_names:
        # skip if the target scene is the same as the query scene.
        if target_scene_name == query_scene_name:
            continue

        # load the target scene and the boxes enclosing its objects.
        target_scene = load_from_json(os.path.join(args.scene_dir, target_scene_name))
        t_to_box = load_boxes(target_scene, target_scene.keys(), box_type='aabb')

        # map each query object to its potential candidates (only if they exist).
        q_to_candidates = find_query_candidates(query_node, q_to_box.keys(), query_scene, target_scene)
        for q, candidates in q_to_candidates.items():
            assert len(candidates) != 0
        q_objects = list(q_to_candidates.keys())

        # for each permutation of the query objects assign corresponding objects.
        q_objects_permutations = find_all_permutations(q_objects)
        corresponding_object_groups = []
        for q_objects_permutation in q_objects_permutations:
            corresponding_object_groups += find_corresponding_objects_groups(q_objects_permutation, q_to_candidates)

        # deduplicate the group of corresponding objects.
        corresponding_object_groups = dedup_corr_groups(corresponding_object_groups)

        # for each group of corresponding objects translate them to origin based on the anchor object and align them.
        for correspondence in corresponding_object_groups:
            if len(correspondence) == 0 or query_node not in correspondence:
                continue
            # translate centroid points for each object.
            q_points, q_to_box_filtered = translate_subscene(correspondence, query_node, q_to_box, query=True)
            t_points, t_to_box_filtered = translate_subscene(correspondence, query_node, t_to_box, query=False)

            # align the points and find the rotation angle
            theta = align(q_points, t_points)

            # rotate the target subscene and compute the radial and angular errors.
            angular_error, radial_error = compute_error(args, correspondence, query_node, q_to_box_filtered,
                                                        t_to_box_filtered, theta)

            # compute the chamfer distance between the corresponding objects.
            chamfer_error = compute_chamfer_error(args, query_scene_name, target_scene_name, correspondence, theta)

            # find total error.
            error = angular_error + radial_error + chamfer_error

            # record the target subscene.
            correspondence_rev = {c: q for q, c in correspondence.items()}
            target_subscene = {'scene_name': target_scene_name, 'target': correspondence[query_node],
                               'correspondence': correspondence_rev, 'theta': float(theta), 'error': float(error)}

            target_subscenes.append(target_subscene)

    # rank the target subscenes based on the number of correspondences and the overall error.
    target_subscenes = sorted(target_subscenes, reverse=True, key=lambda x: (len(x['correspondence']), -x['error']))

    return target_subscenes


def get_args():
    parser = argparse.ArgumentParser('Extracting and Ranking 3D Subscenes', add_help=False)

    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--mode', dest='mode', default='val', help='val | test')
    parser.add_argument('--pc_dir', default='../../data/{}/pc_objects')
    parser.add_argument('--scene_dir', default='../../results/{}/scenes')
    parser.add_argument('--query_dir', default='../../queries/{}/')
    parser.add_argument('--query_input_file_name', default='query_dict_top10.json')
    parser.add_argument('--cp_dir', default='../../results/{}')
    parser.add_argument('--results_folder_name',  default='BruteForce')
    parser.add_argument('--experiment_name', default='BruteForce')
    parser.add_argument('--cd_path', default='../../data/{}/cd_norm_constant.json')
    parser.add_argument('--cd_threshold', default='50', type=str)
    parser.add_argument('--max_coord_scene', default=13.07, type=float, help='13.07 for MP3D')
    parser.add_argument('--num_points', default=4096, type=int)
    parser.add_argument('--with_cat_predictions', action='store_true', default=False,
                        help='If true predicted categories are used')

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
    args.pc_dir = os.path.join(args.pc_dir, args.mode)
    args.query_dir = os.path.join(args.query_dir, args.mode)
    args.cd_norm_constant = load_from_json(args.cd_path.split('.json')[0] + '_{}.json'.format(args.mode))

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

    # retrieve best subscenes for each query
    t0 = time()
    for i, (query, query_info) in enumerate(query_dict.items()):
        if query not in ['lighting-31', 'plant-38', 'chair-23', 'chair-16', 'chair-9']:
            continue
        t = time()
        print('Processing query: {}'.format(query))
        target_subscenes = find_best_target_subscenes(args, query_info, target_scene_names)
        query_info['target_subscenes'] = target_subscenes
        duration = (time() - t) / 60
        print('Processing the query took {} minutes'.format(round(duration, 2)))
        print('*' * 50)

    duration_all = (time() - t0) / 60
    print('Processing all queries took {} minutes'.format(round(duration_all, 2)))

    # save the changes to query dict
    write_to_json(query_dict, query_dict_output_path)


if __name__ == '__main__':
    main()

