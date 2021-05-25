import os
import numpy as np
import pandas as pd
from time import time
from matplotlib import pyplot as plt

from scripts.helper import load_from_json, write_to_json
from scripts.box import Box
from scripts.iou import IoU

query_to_target_map = {}


class Evaluate:
    def __init__(self, query_results, evaluation_path, scene_graph_dir, curr_df, mode, distance_threshold,
                 overlap_threshold, precision_threshold=0.15, q_theta=0):
        self.query_results = query_results
        self.evaluation_path = evaluation_path
        self.scene_graph_dir = scene_graph_dir
        self.curr_df = curr_df
        self.mode = mode
        self.metrics = {'distance_mAP': [self.compute_distance_match, distance_threshold],
                        'overlap_mAP': [self.compute_overlap_match, overlap_threshold]
                        }
        self.precision_threshold = precision_threshold
        self.q_theta = q_theta

    @staticmethod
    def map_obj_to_cat(scene):
        result = {}
        for obj, obj_info in scene.items():
            result[obj] = obj_info['category'][0]
        return result

    @staticmethod
    def find_distance(query_scene, q, context_obj):
        q_cent = np.asarray(query_scene[q]['obbox'][0])
        context_obj_cent = np.asarray(query_scene[context_obj]['obbox'][0])
        return np.linalg.norm(q_cent - context_obj_cent)

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
        target_scene = load_from_json(os.path.join(self.scene_graph_dir, self.mode, target_subscene['scene_name']))
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
        min_iou = query_target_iou
        max_iou = query_target_iou
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

            # compute the threshold relative to the distance between the query and context object
            if iou > self.metrics['overlap_mAP'][1]:
                num_matches += 1.0

            # record the min and max of the IoUs for context and candidate
            if iou <= min_iou:
                min_iou = iou

            if iou >= max_iou:
                max_iou = iou

        min_max_ious = [float(np.round(min_iou, 2)), float(np.round(max_iou, 2))]

        return num_matches / (len(context_objects) + 1), min_max_ious

    def compute_distance_match(self, query_scene, query_object, context_objects, target_subscene):
        # load the target scene
        target_scene = load_from_json(os.path.join(self.scene_graph_dir, self.mode, target_subscene['scene_name']))

        # map each object in the query and target scenes to their cat
        obj_to_cat_query = self.map_obj_to_cat(query_scene)
        obj_to_cat_target = self.map_obj_to_cat(target_scene)

        # check if the target and query objects match
        target_object = target_subscene['target']
        num_matches = 0.0
        if obj_to_cat_query[query_object] == obj_to_cat_target[target_object]:
            num_matches += 1.0

        # for each candidate object in the target scene, examine its corresponding context object from the query scene.
        # a match is detected if the difference between the candidate-target and context-query distances are within a
        # threshold
        min_dist_diff = np.inf
        max_dist_diff = 0
        min_context_candidate_dist = (np.inf, np.inf)
        max_context_candidate_dist = (0, 0)
        for candidate, context_object in target_subscene['correspondence'].items():
            # if the candidate and context objects have different categories, no match is counted.
            if obj_to_cat_query[context_object] != obj_to_cat_target[candidate]:
                continue

            # distance between candidate and the target node.
            candidate_distance = self.find_distance(target_scene, target_subscene['target'], candidate)

            # distance between the context object and the query node.
            context_obj_distance = self.find_distance(query_scene, query_object, context_object)

            # check if the difference between the distances is within a threshold.
            relative_diff = abs(candidate_distance - context_obj_distance) / max(context_obj_distance,
                                                                                 0.00001)
            if relative_diff < self.metrics['distance_mAP'][1]:
                num_matches += 1.0

            # record the min and max of the distances for context and candidate
            if relative_diff < min_dist_diff:
                min_context_candidate_dist = (float(np.round(context_obj_distance, 2)),
                                              float(np.round(candidate_distance, 2)))
                min_dist_diff = relative_diff

            if relative_diff > max_dist_diff:
                max_context_candidate_dist = (float(np.round(context_obj_distance, 2)),
                                              float(np.round(candidate_distance, 2)))
                max_dist_diff = relative_diff

        min_max_distances = [min_context_candidate_dist, max_context_candidate_dist]

        return num_matches / (len(context_objects) + 1), min_max_distances

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
                    acc, min_max_metric = computed_precisions[key]
                else:
                    acc, min_max_metric = self.metrics[metric][0](query_scene, query_object, context_objects,
                                                                  target_subscenes[i])
                    computed_precisions[key] = (acc, min_max_metric)

                # record the precision and raw distance and overlap (only for min and max).
                if self.metrics[metric][1] == self.precision_threshold:
                    precision_key = metric.split('_')[0] + '_precision'
                    target_subscenes[i][precision_key] = acc

                    min_metric_key = 'min_' + metric.split('_')[0]
                    max_metric_key = 'max_' + metric.split('_')[0]
                    target_subscenes[i][min_metric_key] = min_max_metric[0]
                    target_subscenes[i][max_metric_key] = min_max_metric[1]

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
        query_scene = load_from_json(os.path.join(self.scene_graph_dir, self.mode, query_scene_name))
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

            # initialize precision values for the target subscenes
            if self.metrics[metric][1] == self.precision_threshold:
                precision_key = metric.split('_')[0] + '_precision'
                min_metric_key = 'min_' + metric.split('_')[0]
                max_metric_key = 'max_' + metric.split('_')[0]
                for target_subscene in target_subscenes:
                    target_subscene[precision_key] = None
                    target_subscene[min_metric_key] = None
                    target_subscene[max_metric_key] = None

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


def main():
    run_evalulation = False
    run_aggregation = False
    plot_comparisons = False
    plot_name = 'comparisons.png'
    remove_model = False
    visualize_loss = False

    thresholds = np.linspace(0.05, 0.95, num=10)
    metrics = ['distance_mAP', 'overlap_mAP']

    # define paths and parameters
    mode = 'test'
    evaluation_mode = 'test'
    model_name = 'SVDRank'
    experiment_name = 'with_cat_predictions_1d'
    q_theta = 0*np.pi/4
    query_results_input_path = '../results/matterport3d/{}/query_dict_{}_{}.json'.format(model_name, mode,
                                                                                         experiment_name)
    query_results_output_path = '../results/matterport3d/{}/query_dict_{}_{}_evaluated.json'.format(model_name, mode,
                                                                                                    experiment_name)
    evalaution_base_path = '../results/matterport3d/evaluations/{}'.format(evaluation_mode)
    evaluation_path = os.path.join(evalaution_base_path, 'evaluation.csv')
    aggregated_csv_path = os.path.join(evalaution_base_path, 'evaluation_aggregated.csv')
    scene_graph_dir = '../data/matterport3d/scene_graphs'

    # read the results of a model and the evaluation csv file
    query_results = load_from_json(query_results_input_path)
    curr_df = pd.read_csv(evaluation_path)

    # filter the results by query if necessary
    evaluation_queries = ['all']
    if evaluation_queries[0] != 'all':
        query_results = {k: v for k, v in query_results.items() if k in evaluation_queries}

    # initialize the evaluator
    evaluator = Evaluate(query_results=query_results, evaluation_path=evaluation_path, scene_graph_dir=scene_graph_dir,
                         curr_df=curr_df, mode=mode, distance_threshold=0, overlap_threshold=0, q_theta=q_theta)

    if run_evalulation:
        # compute distance or overlap mAP per query for each threshold
        for i, (query_name, results_info) in enumerate(query_results.items()):
            print('Iteration {}/{}'.format(i+1, len(query_results)))
            # initialize the mAP for each query
            for metric in metrics:
                results_info[metric] = {'mAP': []}

            for threshold in thresholds:
                experiment_id = experiment_name + '-' + str(np.round(threshold, 3))
                evaluator.add_to_tabular(model_name, query_name, experiment_id)
                for metric in metrics:
                    evaluator.metrics[metric][1] = threshold
                    evaluator.compute_mAP(metric, query_name, model_name, experiment_id)

        # save evaluation results in tabular format
        evaluator.to_tabular()
        # save the query dict with added precisions
        write_to_json(query_results, query_results_output_path)

    if run_aggregation:
        # average the evaluation results across all queries for each model
        groups = curr_df.groupby(['model_name', 'experiment_id'])
        df_mean = groups.agg({'distance_mAP': 'mean', 'overlap_mAP': 'mean'})
        df_mean.reset_index(inplace=True)
        # convert to percentage and round up to 3 decimals
        for metric in metrics:
            df_mean[metric] = df_mean[metric].apply(lambda x: np.round(x * 100, 3))

        summary_resutls = df_mean.sort_values(by=['model_name', 'experiment_id']).reset_index(drop=True)
        summary_resutls.to_csv(aggregated_csv_path, index=False)

    if plot_comparisons:
        # read the aggregated results
        summary_resutls = pd.read_csv(aggregated_csv_path)
        model_names = summary_resutls['model_name'].unique()

        # plot results for each metric
        for metric in metrics:
            # add results for each model to the plot
            fig, ax = plt.subplots()
            for model_name in model_names:
                summary_resutls_model = summary_resutls.loc[summary_resutls['model_name'] == model_name]
                # find all experiments under a model
                experiment_names = summary_resutls_model['experiment_id'].apply(lambda x: x.split('-')[0]).unique()
                for experiment_name in experiment_names:
                    # choose the experiment under the model name
                    this_experiment = summary_resutls_model['experiment_id'].apply(lambda x: x.split('-')[0] == experiment_name)
                    summary_resutls_model_experiment = summary_resutls_model.loc[this_experiment]
                    # x axis represents thresholds
                    x = summary_resutls_model_experiment['experiment_id'].apply(lambda x: np.float(x.split('-')[-1])).values
                    # y axis represents the mAP values
                    y = summary_resutls_model_experiment[metric].values

                    ax.plot(x, y, label='{}_{}'.format(model_name, experiment_name))
                    plt.title("Evaluating {}".format(metric))
                    plt.xlabel("Thresholds")
                    plt.ylabel(metric)
                    leg = ax.legend()
            plt.savefig('../results/matterport3d/evaluation_plots/{}_{}'.format(metric, plot_name))
            plt.show()

    if remove_model:
        model_exclude = curr_df['model_name'] != model_name
        experiment_exclude = curr_df['experiment_id'].apply(lambda x: x.split('-')[0] != experiment_name)
        curr_df = curr_df[model_exclude | experiment_exclude]
        curr_df.to_csv(evaluation_path, index=False)

    if visualize_loss:
        checkpoints_dir = '../results/matterport3d/{}/{}'.format(model_name, experiment_name)
        # load train and validation losses
        train_loss = np.load(os.path.join(checkpoints_dir, 'training_loss.npy'))
        valid_loss = np.load(os.path.join(checkpoints_dir, 'valid_loss.npy'))

        # plot the losses in one figure
        fig, ax = plt.subplots()
        ax.plot(range(len(train_loss)), train_loss, label='train loss')
        ax.plot(range(len(valid_loss)), valid_loss, label='valid loss')
        plt.title("Loss Evolution")
        plt.xlabel("Iterations")
        plt.ylabel('Loss')
        leg = ax.legend()

        # save and show the plot
        plt.savefig('../results/matterport3d/evaluation_plots/train_valid_losses_{}.png'.format(experiment_name))
        plt.show()


if __name__ == '__main__':
    t = time()
    main()
    duration = time() - t
    print('Evaluation took {} minutes'.format(round(duration / 60, 2)))
