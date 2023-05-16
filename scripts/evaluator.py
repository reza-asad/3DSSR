import os
import numpy as np
import pandas as pd
import torch
import shutil
from chamferdist import ChamferDistance

from scripts.helper import load_from_json, write_to_json
from scripts.box import Box
from scripts.iou import IoU


class Evaluate:
    def __init__(self, pc_dir, pc_dir_queries, query_results, evaluation_path, scene_dir, scene_dir_queries, curr_df,
                 mode, cd_sim, num_points, cat_threshold, bidirectional, df_metadata, fine_cat_field):
        self.pc_dir = pc_dir
        self.pc_dir_queries = pc_dir_queries
        self.query_results = query_results
        self.evaluation_path = evaluation_path
        self.scene_dir = scene_dir
        self.scene_dir_queries = scene_dir_queries
        self.curr_df = curr_df
        self.mode = mode
        self.cd_sim = cd_sim
        self.num_points = num_points
        self.metric = []
        self.cat_threshold = cat_threshold
        self.bidirectional = bidirectional
        self.df_metadata = df_metadata
        self.fine_cat_field = fine_cat_field
        self.max_angle_diff = np.pi/2
        self.theta_q = 0

    def map_obj_to_cat_fine(self, scene, scene_name):
        result = {}
        for obj in scene:
            key = '{}-{}'.format(scene_name.split('.')[0], obj)
            cat = self.df_metadata.loc[key, self.fine_cat_field]
            result[obj] = cat

        return result

    def map_obj_to_cat(self, scene, query=True):
        if query:
            result = {}
            for obj, obj_info in scene.items():
                result[obj] = obj_info['category'][0]
        else:
            result = {}
            cat_key = 'category'
            if self.cat_threshold is not None:
                cat_key = 'category_{}'.format(self.cat_threshold)
            for obj, obj_info in scene.items():
                result[obj] = obj_info[cat_key]

        return result

    @staticmethod
    def translate_obbox(obbox, translation):
        # build the transformation matrix
        transformation = np.eye(4)
        transformation[:3, 3] = translation

        # apply tranlsation to the obbox
        obbox = obbox.apply_transformation(transformation)

        return obbox

    @staticmethod
    def rotate_box(obbox, theta):
        # compute the rotation matrix.
        transformation = np.eye(4)
        rotation = np.asarray([[np.cos(theta), -np.sin(theta), 0],
                               [np.sin(theta), np.cos(theta), 0],
                               [0, 0, 1]])
        transformation[:3, :3] = rotation

        # apply rotation.
        obbox = obbox.apply_transformation(transformation)

        return obbox

    @staticmethod
    def rotate_pc(pc, theta):
        # compute the rotation matrix.
        transformation = np.eye(4)
        rotation = np.asarray([[np.cos(theta), -np.sin(theta), 0],
                               [np.sin(theta), np.cos(theta), 0],
                               [0, 0, 1]])
        transformation[:3, :3] = rotation

        # rotate pc
        pc_rot = np.ones((4, len(pc)), dtype=np.float64)
        pc_rot[0, :] = pc[:, 0]
        pc_rot[1, :] = pc[:, 1]
        pc_rot[2, :] = pc[:, 2]
        pc_rot = np.dot(transformation, pc_rot)
        pc_rot = pc_rot[:3, :].T

        return pc_rot

    @staticmethod
    def compute_iou(obbox1, obbox2):
        # compute the iou
        iou_computer = IoU(obbox1, obbox2)
        iou = iou_computer.iou()

        return iou

    def create_obb_at_origin(self, scene, obj, box_type):
        obbox_vertices = np.asarray(scene[obj][box_type])
        obbox = Box(obbox_vertices)
        obj_translation = -obbox.translation
        obbox = self.translate_obbox(obbox, obj_translation)

        return obbox, obj_translation

    def compute_rel_angle(self, t_c_obbox, q_c_obbox):
        # find the vector connecting the centroid of query/target boxes to origin.
        t_c_centroid = t_c_obbox.translation
        q_c_centroid = q_c_obbox.translation

        # find the 3D angle difference between query and target boxes in radians.
        cos_theta = np.dot(t_c_centroid, q_c_centroid) / (np.linalg.norm(t_c_centroid) * np.linalg.norm(q_c_centroid))

        return np.arccos(cos_theta) / self.max_angle_diff

    @staticmethod
    def compute_rel_distance(t_c_obbox, q_c_obbox):
        # find the distance of the query/target boxes to the origin.
        t_c_distance = np.linalg.norm(t_c_obbox.translation)
        q_c_distance = np.linalg.norm(q_c_obbox.translation)

        # compute distance relative to the query.
        return np.abs(t_c_distance - q_c_distance) / q_c_distance

    def sim_shape(self, query_scene_name, query_object, target_scene_name, target_object, obj_to_cat_query,
                  obj_to_cat_target, obj_to_cat_query_fine, obj_to_cat_target_fine, theta):
        if 'cat' in self.metric[0]:
            # pure category based matching
            if 'cd' not in self.metric[0]:
                return obj_to_cat_query[query_object] == obj_to_cat_target[target_object]
            # reject if categories don't match.
            elif obj_to_cat_query[query_object] not in obj_to_cat_target[target_object]:
                return False
            # reject if fine categorization does not match
            if self.fine_cat_field is not None:
                if obj_to_cat_query_fine[query_object] not in obj_to_cat_target_fine[target_object]:
                    return False

        # creat the path for each pc
        q_file_name = os.path.join(self.pc_dir_queries, '-'.join([query_scene_name.split('.')[0], query_object]) + '.npy')
        t_file_name = os.path.join(self.pc_dir, '-'.join([target_scene_name.split('.')[0], target_object]) + '.npy')

        # load the pc for the query and target obj
        pc1 = np.load(q_file_name)
        pc2 = np.load(t_file_name)

        # sample prepare the pc for distance computation
        np.random.seed(0)
        sampled_indices = np.random.choice(range(len(pc1)), self.num_points, replace=False)
        if self.theta_q != 0:
            pc1 = self.rotate_pc(pc1[sampled_indices, :], self.theta_q)
        else:
            pc1 = pc1[sampled_indices, :]
        pc1 = np.expand_dims(pc1, axis=0)
        pc1 = torch.from_numpy(pc1).cuda()

        # rotate the target pc if needed.
        np.random.seed(0)
        sampled_indices = np.random.choice(range(len(pc2)), self.num_points, replace=False)
        if theta is not None:
            pc2 = self.rotate_pc(pc2[sampled_indices, :], theta)
        else:
            pc2 = pc2[sampled_indices, :]
        pc2 = np.expand_dims(pc2, axis=0)
        pc2 = torch.from_numpy(pc2).cuda()

        # compute the chamfer distance similarity between query and target obj
        chamferDist = ChamferDistance()
        if self.bidirectional:
            dist_bidirectional = chamferDist(pc1, pc2, bidirectional=True)
            dist = dist_bidirectional.detach().cpu().item()
        else:
            dist_forward = chamferDist(pc1, pc2)
            dist = dist_forward.detach().cpu().item()

        # see if the distance is within the threshold
        threshold = self.metric[0].split('_')[-1]
        q_cat = obj_to_cat_query[query_object]

        return dist <= self.cd_sim[q_cat][threshold]

    def compute_dist_angle_match(self, query_scene_name, query_scene, query_object, context_objects, target_subscene):
        target_scene_name = target_subscene['scene_name']
        if '.json' not in target_subscene['scene_name']:
            target_scene_name = target_subscene['scene_name'] + '.json'

        # load the target scene
        target_scene = load_from_json(os.path.join(self.scene_dir, self.mode, target_scene_name))
        target_object = target_subscene['target']

        # map each object in the query and target scenes to their cat
        obj_to_cat_query = self.map_obj_to_cat(query_scene, query=True)
        obj_to_cat_query_fine = None
        if self.fine_cat_field is not None:
            obj_to_cat_query_fine = self.map_obj_to_cat_fine(query_scene, query_scene_name)

        obj_to_cat_target = None
        obj_to_cat_target_fine = None
        if 'cat' in self.metric[0]:
            obj_to_cat_target = self.map_obj_to_cat(target_scene, query=False)
            if self.fine_cat_field is not None:
                obj_to_cat_target_fine = self.map_obj_to_cat_fine(target_scene, target_scene_name)

        # create a obbox object for the query object and translate it to the origin
        obbox_query, q_translation = self.create_obb_at_origin(query_scene, query_object, box_type='aabb')

        # create the obbox for the target object and translate it to the origin
        obbox_target, t_translation = self.create_obb_at_origin(target_scene, target_object, box_type='aabb')

        # read rotation angle if it is computed
        theta = None
        if 'theta' in target_subscene:
            theta = target_subscene['theta']

        num_matches = 0
        if self.sim_shape(query_scene_name, query_object, target_scene_name, target_object,
                          obj_to_cat_query, obj_to_cat_target, obj_to_cat_query_fine, obj_to_cat_target_fine, theta):
            num_matches += 1

        # compute the relative distance between the query and target objects if their category match
        for candidate, context_object in target_subscene['correspondence'].items():
            # if the candidate and context objects have different categories, no match is counted.
            if not self.sim_shape(query_scene_name, context_object, target_scene_name, candidate, obj_to_cat_query,
                                  obj_to_cat_target, obj_to_cat_query_fine, obj_to_cat_target_fine, theta):
                continue

            # create a obbox object for the context object and translated it according to the query object
            q_context_obbox_vertices = np.asarray(query_scene[context_object]['aabb'])
            q_c_obbox = Box(q_context_obbox_vertices)
            q_c_obbox = self.translate_obbox(q_c_obbox, q_translation)

            # create a obbox object for the candidate and translated it according to the target object
            t_context_obbox_vertices = np.asarray(target_scene[candidate]['aabb'])
            t_c_obbox = Box(t_context_obbox_vertices)
            t_c_obbox = self.translate_obbox(t_c_obbox, t_translation)

            # rotate the candidate box if rotation angle theta is available.
            if theta is not None:
                t_c_obbox = self.rotate_box(t_c_obbox, theta)

            # compute the relative distnace between the candidate and context obboxes.
            if 'distance' in self.metric[0]:
                rel_distance = self.compute_rel_distance(t_c_obbox, q_c_obbox)
                # no match if euclidean distance is above the threshold.
                if rel_distance >= self.metric[2]:
                    continue
            if 'angle' in self.metric[0]:
                rel_angle = self.compute_rel_angle(t_c_obbox, q_c_obbox)
                # no match if the angle distance is above threshold.
                if rel_angle >= self.metric[2]:
                    continue

            # all thresholds are passed if we reach here
            num_matches += 1.0

        return num_matches / (len(context_objects) + 1)

    def compute_overlap_match(self, query_scene_name, query_scene, query_object, context_objects, target_subscene):
        target_scene_name = target_subscene['scene_name']
        if '.json' not in target_subscene['scene_name']:
            target_scene_name = target_subscene['scene_name'] + '.json'

        # load the target scene
        target_scene = load_from_json(os.path.join(self.scene_dir, self.mode, target_scene_name))
        target_object = target_subscene['target']

        # map each object in the query and target scenes to their cat
        obj_to_cat_query = self.map_obj_to_cat(query_scene, query=True)
        obj_to_cat_query_fine = None
        if self.fine_cat_field is not None:
            obj_to_cat_query_fine = self.map_obj_to_cat_fine(query_scene, query_scene_name)

        obj_to_cat_target = None
        obj_to_cat_target_fine = None
        if 'cat' in self.metric[0]:
            obj_to_cat_target = self.map_obj_to_cat(target_scene, query=False)
            if self.fine_cat_field is not None:
                obj_to_cat_target_fine = self.map_obj_to_cat_fine(target_scene, target_scene_name)

        # create a obbox object for the query object and translate it to the origin
        obbox_query, q_translation = self.create_obb_at_origin(query_scene, query_object, box_type='aabb')

        # create the obbox for the target object and translate it to the origin
        obbox_target, t_translation = self.create_obb_at_origin(target_scene, target_object, box_type='aabb')

        # read rotation angle if it is computed
        theta = None
        if 'theta' in target_subscene:
            theta = target_subscene['theta']

        # compute the iou between the query and target objects if their category match
        num_matches = 0
        if self.sim_shape(query_scene_name, query_object, target_scene_name, target_object,
                          obj_to_cat_query, obj_to_cat_target, obj_to_cat_query_fine, obj_to_cat_target_fine, theta):
            q_t_iou = self.compute_iou(obbox_target, obbox_query)
            if q_t_iou > self.metric[2]:
                num_matches += 1
        else:
            return 0

        # for each candidate object in the query scene, examine its corresponding context object in the query scene.
        # a match is detected if the IoU between the translated candidate and context objects are within a threshold.
        # the translation is dictated by the query scene and is the vector connecting the context object to query.
        for candidate, context_object in target_subscene['correspondence'].items():
            # if the candidate and context objects have different categories, no match is counted.
            if not self.sim_shape(query_scene_name, context_object, target_scene_name, candidate, obj_to_cat_query,
                                  obj_to_cat_target, obj_to_cat_query_fine, obj_to_cat_target_fine, theta):
                continue

            # create a obbox object for the context object and translated it according to the query object
            q_context_obbox_vertices = np.asarray(query_scene[context_object]['aabb'])
            q_c_obbox = Box(q_context_obbox_vertices)
            q_c_obbox = self.translate_obbox(q_c_obbox, q_translation)

            # create a obbox object for the candidate and translated it according to the target object
            t_context_obbox_vertices = np.asarray(target_scene[candidate]['aabb'])
            t_c_obbox = Box(t_context_obbox_vertices)
            t_c_obbox = self.translate_obbox(t_c_obbox, t_translation)

            # rotate the candidate box if rotation angle theta is available.
            if theta is not None:
                t_c_obbox = self.rotate_box(t_c_obbox, theta)

            # compute the IoU between the candidate and context obboxes.
            iou = self.compute_iou(t_c_obbox, q_c_obbox)

            # compute the threshold relative to the overlap between the query and context object
            if iou > self.metric[2]:
                num_matches += 1.0

        return num_matches / (len(context_objects) + 1)

    def compute_precision_at(self, query_scene, query_scene_name, query_object, context_objects, target_subscenes,
                             top, computed_precisions):
        accuracies = []
        for i in range(top):
            # the case where there are no longer results
            if i >= len(target_subscenes):
                acc = 0
            else:
                target_subscene_name = target_subscenes[i]['scene_name']
                target_object = target_subscenes[i]['target']

                # read the accuracy if you have computed it before.
                key = '-'.join([query_scene_name, query_object, target_subscene_name, target_object])
                if key in computed_precisions:
                    acc = computed_precisions[key]
                else:
                    acc = self.metric[1](query_scene_name, query_scene, query_object, context_objects,
                                         target_subscenes[i])
                    computed_precisions[key] = acc

            accuracies.append(acc)
        return np.mean(accuracies)

    def compute_mAP(self, query_name, model_name, experiment_id, topk=10):
        # load the query subscene and find the query and context objects
        query_scene_name = self.query_results[query_name]['example']['scene_name']
        query_scene = load_from_json(os.path.join(self.scene_dir_queries, self.mode, query_scene_name))
        query_object = self.query_results[query_name]['example']['query']
        context_objects = self.query_results[query_name]['example']['context_objects']

        metric_name = self.metric[0]
        # load the target subscenes up to topk
        target_subscenes = self.query_results[query_name]['target_subscenes'][:topk]

        # memorize the precisions that you already computed in a dict
        computed_precisions = {}
        precision_at = {i: 0 for i in range(1, topk+1)}
        for top in precision_at.keys():
            precision_at[top] = self.compute_precision_at(query_scene, query_scene_name, query_object,
                                                          context_objects, target_subscenes, top,
                                                          computed_precisions)
        # compute and record the mAP
        mAP = np.mean(list(precision_at.values()))
        threshold = np.round(self.metric[2], 3)
        self.query_results[query_name][metric_name]['mAP'].append((threshold, float(np.round(mAP * 100, 2))))

        query_model = (self.curr_df['query_name'] == query_name) & \
                      (self.curr_df['model_name'] == model_name) & \
                      (self.curr_df['experiment_id'] == experiment_id)
        self.curr_df.loc[query_model, metric_name] = mAP

    def add_to_tabular(self, model_name, query_name, experiment_id):
        # check if the row of data already exists
        query_model = (self.curr_df['query_name'] == query_name) & \
                      (self.curr_df['model_name'] == model_name) & \
                      (self.curr_df['experiment_id'] == experiment_id)

        metric_name = self.metric[0]
        if len(self.curr_df.loc[query_model, metric_name]) == 0:
            new_df = pd.DataFrame()
            new_df['model_name'] = [model_name]
            new_df['query_name'] = [query_name]
            new_df['experiment_id'] = [experiment_id]

            # update the dataframe that this evaluation represents.
            self.curr_df = pd.concat([self.curr_df, new_df])

    def to_tabular(self):
        self.curr_df.to_csv(self.evaluation_path, index=False)


def evaluate_subscene_retrieval(args):
    # check if this is evaluating an ablation model or baseline
    if args.ablations:
        evalaution_base_path = '../results/matterport3d/evaluations/ablations'
        if not os.path.exists(evalaution_base_path):
            os.makedirs(evalaution_base_path)
    else:
        evalaution_base_path = '../results/matterport3d/evaluations/{}'.format(args.mode)

    # if this is the first run create the evaluation directory
    if not os.path.exists(evalaution_base_path):
        os.makedirs(evalaution_base_path)
    evaluation_file_name = 'evaluation.csv'
    evaluation_aggregate_file_name = 'evaluation_aggregated.csv'
    if 'objects' in args.query_input_file_name:
        evaluation_file_name = 'evaluation_objects.csv'
        evaluation_aggregate_file_name = 'evaluation_aggregated_objects.csv'
    evaluation_path = os.path.join(evalaution_base_path, evaluation_file_name)

    # if evaluation csv file does not exist, copy it from the template.
    if not os.path.exists(evaluation_path):
        shutil.copy('../results/matterport3d/evaluations/evaluation_template.csv', evaluation_path)

    aggregated_csv_path = os.path.join(evalaution_base_path, evaluation_aggregate_file_name)

    # TODO: for removal consider and condition instead of or.
    curr_df = pd.read_csv(evaluation_path)
    if args.remove_model:
        model_to_remove = curr_df['model_name'] == args.model_name
        experiment_to_remove = curr_df['experiment_id'].apply(lambda x: x.split('-')[0] == args.experiment_name)
        curr_df = curr_df[~(model_to_remove & experiment_to_remove)]
        curr_df.to_csv(evaluation_path, index=False)
        print('Model {} with Experiment {} is removed'.format(args.model_name, args.experiment_name))
        return

    # define paths and parameters
    thresholds = np.linspace(0.05, 0.95, num=10)
    metrics = [
        'distance_cd_mAP_bi_5', 'distance_cd_mAP_bi_10', 'distance_cd_mAP_bi_20', 'distance_cd_mAP_bi_40',
        'angle_cd_mAP_bi_5', 'angle_cd_mAP_bi_10', 'angle_cd_mAP_bi_20', 'angle_cd_mAP_bi_40',
        'distance_angle_cd_mAP_bi_5', 'distance_angle_cd_mAP_bi_10', 'distance_angle_cd_mAP_bi_20', 'distance_angle_cd_mAP_bi_40',
        'distance_angle_cat_cd_mAP_bi_5', 'distance_angle_cat_cd_mAP_bi_10', 'distance_angle_cat_cd_mAP_bi_20', 'distance_angle_cat_cd_mAP_bi_40'
    ]

    # define the path for the results and evaluated results.
    query_results_dir = os.path.join(args.cp_dir.format(args.model_name), args.results_folder_name)
    query_results_input_file_name = args.query_input_file_name.split('.')[0] + '_{}_{}.json'.format(args.mode,
                                                                                                    args.experiment_name)
    query_results_output_file_name = query_results_input_file_name.split('.')[0] + '_evaluated.json'
    query_results_input_path = os.path.join(query_results_dir, query_results_input_file_name)
    query_results_output_path = os.path.join(query_results_dir, query_results_output_file_name)

    # read the results of a model and the evaluation csv file
    query_results = load_from_json(query_results_input_path)

    # filter the results by query if necessary
    evaluation_queries = ['all']
    if evaluation_queries[0] != 'all':
        query_results = {k: v for k, v in query_results.items() if k in evaluation_queries}

    # load the chamfer distance thresholds
    cd_sim = load_from_json(args.cd_path)

    # initialize the evaluator
    evaluator = Evaluate(pc_dir=args.pc_dir, pc_dir_queries=args.pc_dir_queries, query_results=query_results,
                         evaluation_path=evaluation_path, scene_dir=args.scene_dir,
                         scene_dir_queries=args.scene_dir_queries, curr_df=curr_df, mode=args.mode, cd_sim=cd_sim,
                         num_points=args.num_points, cat_threshold=args.cat_threshold, bidirectional=args.bidirectional,
                         df_metadata=args.df_metadata, fine_cat_field=args.fine_cat_field)

    # run evaluation and compute overlap mAP per query for each threshold.
    for i, (query_name, results_info) in enumerate(query_results.items()):
        print('Iteration {}/{}'.format(i+1, len(query_results)))
        # initialize the mAP for each query
        for metric in metrics:
            results_info[metric] = {'mAP': []}

        # TODO: randomly rotate the query (consistent for all models.)
        if args.rotate_query:
            np.random.seed(i)
            theta_q = np.random.choice([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4], 1)[0]
            evaluator.theta_q = theta_q

        for threshold in thresholds:
            experiment_id = args.experiment_name + '-' + str(np.round(threshold, 3))
            for metric in metrics:
                # distance, angle or IoU used for layout matching.
                if ('distance' in metric) or ('angle' in metric):
                    evaluator.metric = [metric, evaluator.compute_dist_angle_match, threshold]
                else:
                    evaluator.metric = [metric, evaluator.compute_overlap_match, threshold]
                evaluator.add_to_tabular(args.model_name, query_name, experiment_id)
                evaluator.compute_mAP(query_name, args.model_name, experiment_id, topk=args.topk)

    # save evaluation results in tabular format
    evaluator.to_tabular()
    # save the query dict with added precisions
    write_to_json(query_results, query_results_output_path)

    # find all metrics from the template csv.
    df_template = pd.read_csv('../results/matterport3d/evaluations/evaluation_template.csv')
    all_metrics = [e for e in df_template.keys() if 'mAP' in e]

    # average the evaluation results across all queries for each model
    curr_df = pd.read_csv(evaluation_path)
    groups = curr_df.groupby(['model_name', 'experiment_id'])
    df_mean = groups.agg({metric: 'mean' for metric in all_metrics})
    df_mean.reset_index(inplace=True)
    # convert to percentage and round up to 3 decimals
    for metric in all_metrics:
        df_mean[metric] = df_mean[metric].apply(lambda x: np.round(x * 100, 3))

    summary_resutls = df_mean.sort_values(by=['model_name', 'experiment_id']).reset_index(drop=True)
    summary_resutls.to_csv(aggregated_csv_path, index=False)


