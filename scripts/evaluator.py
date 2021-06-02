import os
import numpy as np
import pandas as pd
import shutil
from time import time
from optparse import OptionParser

from scripts.helper import load_from_json, write_to_json
from scripts.box import Box
from scripts.iou import IoU


class Evaluate:
    def __init__(self, query_results, evaluation_path, scene_dir, curr_df, mode, overlap_threshold, q_theta=0):
        self.query_results = query_results
        self.evaluation_path = evaluation_path
        self.scene_dir = scene_dir
        self.curr_df = curr_df
        self.mode = mode
        self.metrics = {'overlap_mAP': [self.compute_overlap_match, overlap_threshold]}
        self.q_theta = q_theta

    @staticmethod
    def map_obj_to_cat(scene):
        result = {}
        for obj, obj_info in scene.items():
            result[obj] = obj_info['category'][0]
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
    def rotate_obbox(obbox, theta):
        # build the transformation matrix
        transformation = np.eye(4)
        transformation[:3, :3] = np.asarray([[np.cos(theta), -np.sin(theta), 0],
                                             [np.sin(theta), np.cos(theta), 0],
                                             [0, 0, 1]])

        # apply tranlsation to the obbox
        obbox = obbox.apply_transformation(transformation)

        return obbox

    @staticmethod
    def compute_iou(obbox1, obbox2):
        # compute the iou
        iou_computer = IoU(obbox1, obbox2)
        iou = iou_computer.iou()

        return iou

    def compute_overlap_match(self, query_scene, query_object, context_objects, target_subscene):
        # load the target scene
        target_scene = load_from_json(os.path.join(self.scene_dir, self.mode, target_subscene['scene_name']))
        target_object = target_subscene['target']

        # map each object in the query and target scenes to their cat
        obj_to_cat_query = self.map_obj_to_cat(query_scene)
        obj_to_cat_target = self.map_obj_to_cat(target_scene)

        # create a obbox object for the query object and translate it to the origin
        query_obbox_vertices = np.asarray(query_scene[query_object]['obbox'])
        obbox_query = Box(query_obbox_vertices)
        q_translation = -obbox_query.translation
        obbox_query = self.translate_obbox(obbox_query, q_translation)

        # create the obboxes for the target object and translate it to the origin
        target_obbox_vertices = np.asarray(target_scene[target_object]['obbox'])
        obbox_target = Box(target_obbox_vertices)
        t_translation = -obbox_target.translation
        obbox_target = self.translate_obbox(obbox_target, t_translation)

        # rotate the target obbox if rotation angle is predicted and is greater than 0
        if 'theta' in target_subscene and target_subscene['theta'] > 0:
            obbox_target = self.rotate_obbox(obbox_target, target_subscene['theta'])

        # compute the iou between the query and target objects if their category match
        num_matches = 0
        query_target_iou = 0.0
        if obj_to_cat_query[query_object] == obj_to_cat_target[target_object]:
            query_target_iou = self.compute_iou(obbox_target, obbox_query)
            if query_target_iou > self.metrics['overlap_mAP'][1]:
                num_matches += 1

        # for each candidate object in the query scene, examine its corresponding context object in the query scene.
        # a match is detected if the IoU between the translated candidate and context objects are within a threshold.
        # the translation is dictated by the query scene and is the vector connecting the context object to query.
        for candidate, context_object in target_subscene['correspondence'].items():
            # if the candidate and context objects have different categories, no match is counted.
            if obj_to_cat_query[context_object] != obj_to_cat_target[candidate]:
                continue

            # create a obbox object for the context object and translated it according to the query object
            q_context_obbox_vertices = np.asarray(query_scene[context_object]['obbox'])
            q_c_obbox = Box(q_context_obbox_vertices)
            q_c_obbox = self.translate_obbox(q_c_obbox, q_translation)

            # create a obbox object for the candidate and translated it according to the target object
            t_context_obbox_vertices = np.asarray(target_scene[candidate]['obbox'])
            t_c_obbox = Box(t_context_obbox_vertices)
            t_c_obbox = self.translate_obbox(t_c_obbox, t_translation)

            # rotate the candidate obbox around the target object (origin), if rotation is predicted and is greater
            # than 0.
            if 'theta' in target_subscene and target_subscene['theta'] > 0:
                t_c_obbox = self.rotate_obbox(t_c_obbox, target_subscene['theta'])

            # compute the IoU between the candidate and context obboxes.
            iou = self.compute_iou(t_c_obbox, q_c_obbox)

            # compute the threshold relative to the overlap between the query and context object
            if iou > self.metrics['overlap_mAP'][1]:
                num_matches += 1.0

        return num_matches / (len(context_objects) + 1)

    def compute_precision_at(self, query_scene, query_scene_name, query_object, context_objects, target_subscenes,
                             top, metric, computed_precisions):
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
                    acc = self.metrics[metric][0](query_scene, query_object, context_objects, target_subscenes[i])
                    computed_precisions[key] = acc

            accuracies.append(acc)
        return np.mean(accuracies)

    def rotate_query(self, query_scene, query_obj, context_objects):
        # create a box for each query and context object. translate the query subscene to the origin.
        obj_to_obbox = {}
        vertices = np.asarray(query_scene[query_obj]['obbox'])
        obj_to_obbox[query_obj] = Box(vertices)
        q_translation = -obj_to_obbox[query_obj].translation
        obj_to_obbox[query_obj] = self.translate_obbox(obj_to_obbox[query_obj], q_translation)

        for c_obj in context_objects:
            vertices = np.asarray(query_scene[c_obj]['obbox'])
            obj_to_obbox[c_obj] = Box(vertices)
            obj_to_obbox[c_obj] = self.translate_obbox(obj_to_obbox[c_obj], q_translation)

        # rotate the query subscene
        transformation = np.eye(4)
        rotation = np.asarray([[np.cos(self.q_theta), -np.sin(self.q_theta), 0],
                               [np.sin(self.q_theta), np.cos(self.q_theta), 0],
                               [0, 0, 1]])
        transformation[:3, :3] = rotation
        for obj in obj_to_obbox.keys():
            obj_to_obbox[obj] = obj_to_obbox[obj].apply_transformation(transformation)

            # translate the query subscne back to where it was
            obj_to_obbox[obj] = self.translate_obbox(obj_to_obbox[obj], -q_translation)

            # record the obboxes for the rotated subscene.
            query_scene[obj]['obbox'] = obj_to_obbox[obj].vertices.tolist()

        return query_scene

    def compute_mAP(self, metric, query_name, model_name, experiment_id, topk=10):
        # load the query subscene and find the query and context objects
        query_scene_name = self.query_results[query_name]['example']['scene_name']
        query_scene = load_from_json(os.path.join(self.scene_dir, self.mode, query_scene_name))
        query_object = self.query_results[query_name]['example']['query']
        context_objects = self.query_results[query_name]['example']['context_objects']

        # rotate the query scene if theta is bigger than 0
        if self.q_theta > 0:
            query_scene = self.rotate_query(query_scene, query_object, context_objects)

        if len(context_objects) == 0:
            mAP = 1
            print('No context objects in the query, hence the mAP is trivially 1')
        else:
            # load the target subgraphs up to topk
            target_subscenes = self.query_results[query_name]['target_subscenes'][:topk]

            # memorize the precisions that you already computed in a dict
            computed_precisions = {}
            precision_at = {i: 0 for i in range(1, topk+1)}
            for top in precision_at.keys():
                precision_at[top] = self.compute_precision_at(query_scene, query_scene_name, query_object,
                                                              context_objects, target_subscenes, top, metric,
                                                              computed_precisions)
            # compute and record the mAP
            mAP = np.mean(list(precision_at.values()))
            threshold = np.round(self.metrics[metric][1], 1)
            self.query_results[query_name][metric]['mAP'].append((threshold, float(np.round(mAP * 100, 2))))

        query_model = (self.curr_df['query_name'] == query_name) & \
                      (self.curr_df['model_name'] == model_name) & \
                      (self.curr_df['experiment_id'] == experiment_id)
        column_name = metric
        self.curr_df.loc[query_model, column_name] = mAP

    def add_to_tabular(self, model_name, query_name, experiment_id):
        new_df = pd.DataFrame()
        new_df['model_name'] = [model_name]
        new_df['query_name'] = [query_name]
        new_df['experiment_id'] = [experiment_id]

        # update the dataframe that this evaluation represents.
        self.curr_df = pd.concat([self.curr_df, new_df])

    def to_tabular(self):
        self.curr_df.to_csv(self.evaluation_path, index=False)


def get_args():
    parser = OptionParser()
    parser.add_option('--mode', dest='mode', default='test', help='val|test')
    parser.add_option('--remove_model', action='store_true', dest='remove_model', default=False,
                      help='If True the model and its corresponding experiment are removed from the evaluation table.')
    parser.add_option('--ablations', dest='ablations', default='False',
                      help='If True the evaluation results are stored in the ablation folder.')
    parser.add_option('--model_name', dest='model_name', default='LearningBased', help='LearningBased|GKRank|CatRank|'
                                                                                    'SVDRank|RandomRank')
    parser.add_option('--experiment_name', dest='experiment_name', default='AlignRank', help='AlignRankOracle|AlignRank|'
                                                                                              'GKRank|CatRank|RandomRank|'
                                                                                              'SVDRank1D|SVDRank3D|'
                                                                                              'AlignRank[-Align]|'
                                                                                              'AlignRank[-GNN]')

    (options, args) = parser.parse_args()
    return options


def main():
    # load arguments
    args = get_args()
    ablations = args.ablations == 'True'

    # define paths and parameters
    q_theta = 0*np.pi/4
    thresholds = np.linspace(0.05, 0.95, num=10)
    metrics = ['overlap_mAP']
    query_results_input_path = '../results/matterport3d/{}/query_dict_{}_{}.json'.format(args.model_name, args.mode,
                                                                                         args.experiment_name)
    query_results_output_path = '../results/matterport3d/{}/query_dict_{}_{}_evaluated.json'.format(args.model_name,
                                                                                                    args.mode,
                                                                                                    args.experiment_name)
    # check if this is evaluating an ablation model or baseline
    if args.ablations:
        evalaution_base_path = '../results/matterport3d/evaluations/ablation'
        if not os.path.exists(evalaution_base_path):
            os.makedirs(evalaution_base_path)
    else:
        evalaution_base_path = '../results/matterport3d/evaluations/{}'.format(args.mode)

    # if this is the first run create the evaluation directory
    if not os.path.exists(evalaution_base_path):
        os.makedirs(evalaution_base_path)
    evaluation_path = os.path.join(evalaution_base_path, 'evaluation.csv')

    # if evaluation csv file does not exist, copy it from the template.
    if not os.path.exists(evaluation_path):
        shutil.copy('../results/matterport3d/evaluations/evaluation_template.csv', evaluation_path)

    aggregated_csv_path = os.path.join(evalaution_base_path, 'evaluation_aggregated.csv')
    scene_dir = '../data/matterport3d/scenes'

    # read the results of a model and the evaluation csv file
    query_results = load_from_json(query_results_input_path)
    curr_df = pd.read_csv(evaluation_path)

    if args.remove_model:
        model_exclude = curr_df['model_name'] != args.model_name
        experiment_exclude = curr_df['experiment_id'].apply(lambda x: x.split('-')[0] != args.experiment_name)
        curr_df = curr_df[model_exclude | experiment_exclude]
        curr_df.to_csv(evaluation_path, index=False)
        print('Model {} with Experiment {} is removed'.format(args.model_name, args.experiment_name))
        return

    # make sure the current run does not duplicate the previous runs.
    if len(curr_df) > 0:
        model_experiments = curr_df[['model_name', 'experiment_id']].\
            apply(lambda x: '-'.join([x['model_name'], x['experiment_id'].split('-')[0]]), axis=1).unique()
        curr_model_experiment = '-'.join([args.model_name, args.experiment_name])
        if curr_model_experiment in model_experiments:
            raise Exception('Model already evaluated. Try another experiment or removing the model')

    # filter the results by query if necessary
    evaluation_queries = ['all']
    if evaluation_queries[0] != 'all':
        query_results = {k: v for k, v in query_results.items() if k in evaluation_queries}

    # initialize the evaluator
    evaluator = Evaluate(query_results=query_results, evaluation_path=evaluation_path, scene_dir=scene_dir,
                         curr_df=curr_df, mode=args.mode, overlap_threshold=0, q_theta=q_theta)

    # run evaluation and compute overlap mAP per query for each threshold.
    for i, (query_name, results_info) in enumerate(query_results.items()):
        print('Iteration {}/{}'.format(i+1, len(query_results)))
        # initialize the mAP for each query
        for metric in metrics:
            results_info[metric] = {'mAP': []}

        for threshold in thresholds:
            experiment_id = args.experiment_name + '-' + str(np.round(threshold, 3))
            evaluator.add_to_tabular(args.model_name, query_name, experiment_id)
            for metric in metrics:
                evaluator.metrics[metric][1] = threshold
                evaluator.compute_mAP(metric, query_name, args.model_name, experiment_id)

    # save evaluation results in tabular format
    evaluator.to_tabular()
    # save the query dict with added precisions
    write_to_json(query_results, query_results_output_path)

    # average the evaluation results across all queries for each model
    curr_df = pd.read_csv(evaluation_path)
    groups = curr_df.groupby(['model_name', 'experiment_id'])
    df_mean = groups.agg({'overlap_mAP': 'mean'})
    df_mean.reset_index(inplace=True)
    # convert to percentage and round up to 3 decimals
    for metric in metrics:
        df_mean[metric] = df_mean[metric].apply(lambda x: np.round(x * 100, 3))

    summary_resutls = df_mean.sort_values(by=['model_name', 'experiment_id']).reset_index(drop=True)
    summary_resutls.to_csv(aggregated_csv_path, index=False)


if __name__ == '__main__':
    t = time()
    main()
    duration = time() - t
    print('Evaluation took {} minutes'.format(round(duration / 60, 2)))
