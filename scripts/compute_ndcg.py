import os
import argparse
import numpy as np
from chamferdist import ChamferDistance
import torch

from scripts.helper import load_from_json
from scripts.box import Box
from scripts.iou import IoU

# Supervised Point Transformer
sup_pt_config = {
    'cp_dir': '../results/{}/LearningBased',
    'results_folder_name': 'region_classification_transformer_2_32_4096_non_equal_full_region_top10',
    'experiment_name': 'supervised_point_transformer'
}

# 3D DINO
dino_config = {
    'cp_dir': '../results/{}/LearningBased',
    'results_folder_name': '3D_DINO_regions_non_equal_full_top10_seg',
    'experiment_name': '3D_DINO_point_transformer'
}

# 3D DINO epoch 10
dino_10_config = {
    'cp_dir': '../results/{}/LearningBased',
    'results_folder_name': '3D_DINO_regions_non_equal_full_top10_seg',
    'experiment_name': '3D_DINO_point_transformer_10'
}

# 3D DINO epoch 10
dino_v2_config = {
    'cp_dir': '../results/{}/LearningBased',
    'results_folder_name': '3D_DINO_regions_non_equal_full_top10_seg_v2',
    'experiment_name': '3D_DINO_point_transformer_v2'
}

# CSC config
csc_config = {
    'cp_dir': '/home/reza/Documents/research/ContrastiveSceneContexts/data_processed_subset/matterport3d/embeddings/',
    'results_folder_name': 'PointTransformerSeg',
    'experiment_name': 'CSC_point_transformer'
}

# CSC config epoch 10
csc_10_config = {
    'cp_dir': '/home/reza/Documents/research/ContrastiveSceneContexts/data_processed_subset/matterport3d/embeddings/',
    'results_folder_name': 'PointTransformerSeg10',
    'experiment_name': 'CSC_point_transformer_10'
}

# RandomRank config
random_config = {
    'cp_dir': '../results/{}/RandomRank',
    'results_folder_name': '',
    'experiment_name': 'RandomRank'
}

# CatRank config
cat_config = {
    'cp_dir': '../results/{}/CatRank',
    'results_folder_name': '',
    'experiment_name': 'CatRank'
}

# OracleRank config
oracle_config = {
    'cp_dir': '../results/{}/OracleRank',
    'results_folder_name': '',
    'experiment_name': 'OracleRank'
}

# GKRank config
gk_config = {
    'cp_dir': '../results/{}/GKRank',
    'results_folder_name': '',
    'experiment_name': 'GKRank'
}


# def find_relevance_score(gt_candidate, target_subscene):
#     rel_score = 0
#     for t_c, is_match in gt_candidate['match_info'].items():
#         if is_match:
#             if t_c in target_subscene['correspondence'] or t_c == target_subscene['target']:
#                 rel_score += 1
#
#     return rel_score


# def find_relevance_scores_ideal(args, gt_subscenes_q):
#     ideal_results = []
#     for ideal_subscene in gt_subscenes_q:
#         # find the image name
#         ideal_subscene_name = ideal_subscene['scene_name']
#         t = ideal_subscene['target']
#         img_name = '{}_{}.png'.format(ideal_subscene_name, t)
#
#         # compute relevance score
#         rel_score = np.sum(list(ideal_subscene['match_info'].values()))
#         ideal_results.append((img_name, rel_score))
#
#     # sort the ideal ranking based on the user's relevance score
#     ideal_results = sorted(ideal_results, reverse=True, key=lambda x: x[1])[:args.topk]
#
#     return ideal_results
#
#
# def find_relevance_scores_model(args, query_info, target_subscenes, gt_subscenes_q):
#     num_query_objects = 1 + len(query_info['context_objects'])
#
#     # find the best relevance score for each target subscene
#     model_results = []
#     for target_subscene in target_subscenes:
#         target_scene_name = target_subscene['scene_name'].split('.')[0]
#         t = target_subscene['target']
#         best_score = 0
#         for gt_candidate in gt_subscenes_q:
#             if gt_candidate['scene_name'] == target_scene_name:
#                 rel_score = find_relevance_score(gt_candidate, target_subscene)
#                 if rel_score > best_score:
#                     best_score = rel_score
#
#         # record the best relevance score for the model's target subscene.
#         img_name = '{}_{}.png'.format(target_scene_name, t)
#         model_results.append((img_name, best_score))
#
#     return model_results


def translate_obbox(obbox, translation):
    # build the transformation matrix
    transformation = np.eye(4)
    transformation[:3, 3] = translation

    # apply tranlsation to the obbox
    obbox = obbox.apply_transformation(transformation)

    return obbox


def compute_iou(obbox1, obbox2):
    # compute the iou
    iou_computer = IoU(obbox1, obbox2)
    iou = iou_computer.iou()

    return iou


def create_obb_at_origin(scene, obj, box_type):
    obbox_vertices = np.asarray(scene[obj][box_type])
    obbox = Box(obbox_vertices)
    obj_translation = -obbox.translation
    obbox = translate_obbox(obbox, obj_translation)

    return obbox, obj_translation


def find_cd_sim(args, query_scene, query_scene_name, q, target_scene, target_scene_name, t):
    # creat the path for each pc
    q_file_name = os.path.join(args.pc_dir, '-'.join([query_scene_name.split('.')[0], q]) + '.npy')
    t_file_name = os.path.join(args.pc_dir, '-'.join([target_scene_name.split('.')[0], t]) + '.npy')

    # load the pc for the query and target obj
    pc1 = np.load(q_file_name)
    pc2 = np.load(t_file_name)

    # sample prepare the pc for distance computation
    np.random.seed(0)
    sampled_indices = np.random.choice(range(len(pc1)), args.num_points, replace=False)
    pc1 = np.expand_dims(pc1[sampled_indices, :], axis=0)
    pc1 = torch.from_numpy(pc1).cuda()
    np.random.seed(0)
    sampled_indices = np.random.choice(range(len(pc2)), args.num_points, replace=False)
    pc2 = np.expand_dims(pc2[sampled_indices, :], axis=0)
    pc2 = torch.from_numpy(pc2).cuda()

    # compute the chamfer distance similarity between query and target obj
    chamferDist = ChamferDistance()
    dist_forward = chamferDist(pc1, pc2)
    dist = dist_forward.detach().cpu().item()

    q_cat = query_scene[q]['category'][0]
    if dist > args.cd_sim[q_cat][args.cd_threshold]:
        sim = 0
    else:
        sim = 1 / (1 + dist)

    return sim


def find_relevance_score_sim(args, query_scene, query_scene_name, q, q_obbox, target_scene, target_scene_name, t,
                             t_obbox):
    # skip if categories don't match
    q_cat = query_scene[q]['category'][0]
    t_cat = target_scene[t]['category'][0]
    if q_cat != t_cat:
        return 0

    # compute similarity based on CD
    sim_cd = find_cd_sim(args, query_scene, query_scene_name, q, target_scene, target_scene_name, t)

    # no point in counting IoU if objects are not similar.
    if sim_cd == 0:
        total_sim = 0
    else:
        # compute iou in the world coordinate
        iou = compute_iou(q_obbox, t_obbox)

        total_sim = sim_cd + iou

    return total_sim


def find_relevance_scores_ideal(args, query_info, gt_subscenes_q):
    ideal_results = []
    for ideal_subscene in gt_subscenes_q:
        # print(ideal_subscene)
        # find the image name
        ideal_subscene_name = ideal_subscene['scene_name']
        t = ideal_subscene['target']
        img_name = '{}_{}.png'.format(ideal_subscene_name, t)

        # load the query and target scenes.
        query_scene = load_from_json(os.path.join(args.scene_dir, query_info['scene_name']))
        target_scene = load_from_json(os.path.join(args.scene_dir, ideal_subscene_name + '.json'))

        # create a obbox object for the query object and translate it to the origin
        obbox_query, q_translation = create_obb_at_origin(query_scene, query_info['query'], box_type='obbox')

        # create the obbox for the target object and translate it to the origin
        obbox_target, t_translation = create_obb_at_origin(target_scene, t, box_type='obbox')

        # compute relevance score
        rel_score = 0
        for q_c, t_c in ideal_subscene['correspondence'].items():
            if t_c != '':
                if ideal_subscene['target'] == t_c:
                    rel_score += find_relevance_score_sim(args, query_scene, query_info['scene_name'],
                                                          query_info['query'], obbox_query, target_scene,
                                                          ideal_subscene_name, t, obbox_target)
                else:
                    # create a obbox object for the context object and translated it according to the query object
                    q_context_obbox_vertices = np.asarray(query_scene[q_c]['obbox'])
                    q_c_obbox = Box(q_context_obbox_vertices)
                    q_c_obbox = translate_obbox(q_c_obbox, q_translation)

                    # create a obbox object for the candidate and translated it according to the target object
                    t_context_obbox_vertices = np.asarray(target_scene[t_c]['obbox'])
                    t_c_obbox = Box(t_context_obbox_vertices)
                    t_c_obbox = translate_obbox(t_c_obbox, t_translation)
                    rel_score += find_relevance_score_sim(args, query_scene, query_info['scene_name'], q_c, q_c_obbox,
                                                          target_scene, ideal_subscene_name, t_c, t_c_obbox)

        num_query_objects = 1 + len(query_info['context_objects'])
        ideal_results.append((img_name, rel_score/num_query_objects))

    # sort the ideal ranking based on the user's relevance score
    ideal_results = sorted(ideal_results, reverse=True, key=lambda x: x[1])[:args.topk]

    return ideal_results


def find_relevance_scores_model(args, query_info, target_subscenes):
    # find the best relevance score for each target subscene
    model_results = []
    for target_subscene in target_subscenes:
        target_scene_name = target_subscene['scene_name'].split('.')[0]
        t = target_subscene['target']

        # load the query and target scenes.
        query_scene = load_from_json(os.path.join(args.scene_dir, query_info['scene_name']))
        target_scene = load_from_json(os.path.join(args.scene_dir, target_scene_name + '.json'))

        # create a obbox object for the query object and translate it to the origin
        obbox_query, q_translation = create_obb_at_origin(query_scene, query_info['query'], box_type='obbox')

        # create the obbox for the target object and translate it to the origin
        obbox_target, t_translation = create_obb_at_origin(target_scene, t, box_type='obbox')

        rel_score = find_relevance_score_sim(args, query_scene, query_info['scene_name'], query_info['query'],
                                             obbox_query, target_scene, target_scene_name, t, obbox_target)
        for t_c, q_c in target_subscene['correspondence'].items():
            # create a obbox object for the context object and translated it according to the query object
            q_context_obbox_vertices = np.asarray(query_scene[q_c]['obbox'])
            q_c_obbox = Box(q_context_obbox_vertices)
            q_c_obbox = translate_obbox(q_c_obbox, q_translation)

            # create a obbox object for the candidate and translated it according to the target object
            t_context_obbox_vertices = np.asarray(target_scene[t_c]['obbox'])
            t_c_obbox = Box(t_context_obbox_vertices)
            t_c_obbox = translate_obbox(t_c_obbox, t_translation)

            rel_score += find_relevance_score_sim(args, query_scene, query_info['scene_name'], q_c, q_c_obbox,
                                                  target_scene, target_scene_name, t_c, t_c_obbox)

        # record the best relevance score for the model's target subscene.
        img_name = '{}_{}.png'.format(target_scene_name, t)
        num_query_objects = 1 + len(query_info['context_objects'])
        model_results.append((img_name, rel_score/num_query_objects))

    return model_results


def compute_dcg(ranked_results):
    # read the relevance scores.
    rel_scores = np.asarray([score for _, score in ranked_results], dtype=np.float64)

    # compute the numerator of DCG
    numerator = 2**rel_scores - 1

    # compute the denominator of the DCG score.
    denominator = np.arange(2, len(ranked_results)+2)
    denominator = np.log2(denominator)

    # compute dcg.
    dcg = np.sum(numerator / denominator)

    return dcg


def evaluate(args, gt_file_path):
    # load the topk ranked results from the model
    query_results_path = os.path.join(args.cp_dir, args.results_folder_name,
                                      'query_dict_top10_{}_{}.json'.format(args.mode, args.experiment_name))
    query_results = load_from_json(query_results_path)

    # load the user's input.
    gt_subscenes = load_from_json(gt_file_path)

    # for each query and it's retrieved subscenes find the highest relevance score based on the user's input.
    dcg_ideal_scores, dcg_model_scores = [], []
    for query, results in query_results.items():
        # print(query)
        # if query != 'lighting-11':
        #     continue
        if query not in gt_subscenes:
            continue

        # rank the top 10 ideal results based on their relevance score.
        query_info = results['example']
        gt_subscenes_q = gt_subscenes[query]
        ideal_results = find_relevance_scores_ideal(args, query_info, gt_subscenes_q)

        # load the topk target subscenes from a model.
        target_subscenes = results['target_subscenes'][:args.topk]

        # find the relevance score based on user's input.
        model_results = find_relevance_scores_model(args, query_info, target_subscenes)

        # compute ndcg per query.
        dcg_ideal = compute_dcg(ideal_results)
        dcg_model = compute_dcg(model_results)
        dcg_ideal_scores.append(dcg_ideal)
        dcg_model_scores.append(dcg_model)

    # average ndcg across all queries.
    dcg_ideal_score = np.mean(dcg_ideal_scores)
    dcg_model_score = np.mean(dcg_model_scores)

    return dcg_ideal_score, dcg_model_score


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def get_args():
    parser = argparse.ArgumentParser('Evaluating 3D Subscene Retrieval', add_help=False)
    parser.add_argument('--ranking_strategy', default='3D DINO',
                        help='3D DINO | 3D DINO V2 | CSC | SUP PT | Random | Cat | Oracle | GK')
    parser.add_argument('--mode', default='test', help='val | test')
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--pc_dir', default='../data/{}/pc_regions')
    parser.add_argument('--scene_dir', default='../results/{}/scenes_top10')
    parser.add_argument('--query_input_file_name', default='query_dict_top10.json')
    parser.add_argument('--topk', default=10, type=int, help='number of top results for mAP computations.')
    parser.add_argument('--num_points', default=4096, type=int, help='number of points randomly sampled form the pc.')
    parser.add_argument('--cd_path', default='../data/{}/cd_thresholds.json')
    parser.add_argument('--cd_threshold', default='40')
    parser.add_argument('--gt_subscenes_dir', default='../results/{}/GroundTruthSubscenes')
    parser.add_argument('--gt_subscenes_file_names', default=[
                                                              "qimin-answers.json",
                                                              "linas-answers.json",
                                                              "saurabh-answers.json",
                                                              "tristan-answers.json",
                                                              "julia-answers.json",
                                                              "supriya-answers.json",
                                                              "tommaso-answers.json"
                                                              ],
                        type=str, nargs='+')

    return parser


def main():
    # read the args
    parser = argparse.ArgumentParser('Evaluating 3D Subscene Retrieval', parents=[get_args()])
    args = parser.parse_args()
    args.scene_dir = os.path.join(args.scene_dir, args.mode)
    args.pc_dir = os.path.join(args.pc_dir, args.mode)
    args.cd_path = args.cd_path.split('.json')[0] + '_{}.json'.format(args.mode)

    # determine the correct config for the pretraining strategy
    if args.ranking_strategy == '3D DINO':
        config = dino_config
    elif args.ranking_strategy == '3D DINO 10':
        config = dino_10_config
    elif args.ranking_strategy == '3D DINO V2':
        config = dino_v2_config
    elif args.ranking_strategy == 'CSC':
        config = csc_config
    elif args.ranking_strategy == 'CSC 10':
        config = csc_10_config
    elif args.ranking_strategy == 'SUP PT':
        config = sup_pt_config
    elif args.ranking_strategy == 'Random':
        config = random_config
    elif args.ranking_strategy == 'Cat':
        config = cat_config
    elif args.ranking_strategy == 'Oracle':
        config = oracle_config
    elif args.ranking_strategy == 'GK':
        config = gk_config

    # add the pretraining configs and apply 3dssr and adjust paths.
    for k, v in config.items():
        vars(args)[k] = v
    adjust_paths(args, exceptions=[])

    # load the CD thresholds
    args.cd_sim = load_from_json(args.cd_path)

    # compute ndcg for each user and average the results
    dcg_scores_ideal = []
    dcg_scores_model = []
    user_to_ndcg = {}
    for gt_file_name in args.gt_subscenes_file_names:
        # print(gt_file_name)
        gt_file_path = os.path.join(args.gt_subscenes_dir, gt_file_name)
        dcg_score_ideal, dcg_score_model = evaluate(args, gt_file_path)
        dcg_scores_ideal.append(dcg_score_ideal)
        dcg_scores_model.append(dcg_score_model)
        user_to_ndcg[gt_file_name] = dcg_score_ideal

    print('DCG score ideal mean and std is {}, {}'.format(np.mean(dcg_scores_ideal), np.std(dcg_scores_ideal)))
    print('DCG score model mean and std is {}, {}'.format(np.mean(dcg_scores_model), np.std(dcg_scores_model)))
    print(user_to_ndcg)


if __name__ == '__main__':
    main()

