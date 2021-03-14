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
                 overlap_threshold, precision_threshold=0.05):
        self.query_results = query_results
        self.evaluation_path = evaluation_path
        self.scene_graph_dir = scene_graph_dir
        self.curr_df = curr_df
        self.mode = mode
        self.metrics = {'distance_mAP': [self.compute_distance_match, distance_threshold, [0]],
                        'overlap_mAP': [self.compute_overlap_match, overlap_threshold,
                                        [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 1.5*np.pi, 7*np.pi/4]]
                        }
        self.precision_threshold = precision_threshold

    def sort_by_distance(self, scene_name, query_object, context_objects, mode):
        # load the query scene
        scene = load_from_json(os.path.join(self.scene_graph_dir, mode, scene_name))
        # find the centroid of the query object
        q_cent = np.asarray(scene[query_object]['obbox'][0])

        # record the distance between each context object and the query object
        dist_info = []
        for context_object in context_objects:
            context_obj_cent = np.asarray(scene[context_object]['obbox'][0])
            dist = np.linalg.norm(context_obj_cent - q_cent)
            dist_info.append((context_object, dist))
        sorted_obj_dist = sorted(dist_info, key=lambda x: x[1])
        sorted_context_objects = [obj for obj, _ in sorted_obj_dist]
        return sorted_context_objects

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

        # map each object in the query and target scenes to their cat
        obj_to_cat_query = self.map_obj_to_cat(query_scene)
        obj_to_cat_target = self.map_obj_to_cat(target_scene)

        # if target and query target don't match mAP is 0
        target_object = target_subscene['target']
        if obj_to_cat_query[query_object] != obj_to_cat_target[target_object]:
            return 0

        # create a obbox object for the query and target objects
        query_obbox_vertices = np.asarray(query_scene[query_object]['obbox'])
        obbox_query = Box(query_obbox_vertices)

        target_obbox_vertices = np.asarray(target_scene[target_object]['obbox'])
        obbox_target = Box(target_obbox_vertices)

        # compute the iou between the query and target objects (assumes query is at the origin)
        visited = set()
        target_obbox_vertices_copy = target_obbox_vertices.copy()
        obbox_target_copy = Box(target_obbox_vertices_copy)
        obbox_target_copy = self.translate_obbox(obbox_target_copy, -obbox_target_copy.translation)
        query_target_iou = self.compute_iou(obbox_target_copy, obbox_query)
        if query_target_iou > self.metrics['overlap_mAP'][1]:
            visited.add(target_object)

        # for each context object in the query scene find the object in the target subscene with maximum overlap that
        # satisfies the overlap threshold. Find the percentage of such matches relative to the total number of context
        # objects.
        for context_object in context_objects:
            context_object_cat = obj_to_cat_query[context_object]
            # create a obbox object for the context object
            context_obj_obbox_vertices = np.asarray(query_scene[context_object]['obbox'])
            obbox_context_object = Box(context_obj_obbox_vertices)

            # find the translation between the query and the current context object.
            translation_q_c = obbox_query.translation - obbox_context_object.translation

            # translate the context object to the origin
            translation_c = -obbox_context_object.translation
            obbox_context_object = self.translate_obbox(obbox_context_object, translation_c)

            # find the object in the target subscene with the same category that has maximum overlap with the context
            # object.
            best_iou = 0
            best_candidate = None
            for candidate in target_subscene['context_objects']:
                if candidate not in visited:
                    candidate_cat = obj_to_cat_target[candidate]
                    if candidate_cat == context_object_cat:
                        # create a obbox object for the candidate object
                        candidate_obj_obbox_vertices = np.asarray(target_scene[candidate]['obbox'])
                        obbox_candidate_object = Box(candidate_obj_obbox_vertices)

                        # translate the candidate object by the the vector that translates the target object to origin
                        obbox_candidate_object = self.translate_obbox(obbox_candidate_object, -obbox_target.translation)

                        # move the candidate object using the translation vector that brings context to query.
                        obbox_candidate_object = self.translate_obbox(obbox_candidate_object, translation_q_c)

                        # compute the iou between the context and the candidate objects
                        iou = self.compute_iou(obbox_context_object, obbox_candidate_object)

                        # compute the threshold relative to the distance between the query and context object
                        if iou > best_iou and iou > self.metrics['overlap_mAP'][1]:
                            best_iou = iou
                            best_candidate = candidate

            # add the best candidate to visited nodes
            if best_candidate is not None:
                visited.add(best_candidate)
        return len(visited) / (len(context_objects) + 1)

    def compute_distance_match(self, query_scene, query_object, context_objects, target_subscene):
        # load the target scene
        target_scene = load_from_json(os.path.join(self.scene_graph_dir, self.mode, target_subscene['scene_name']))

        # map each object in the query and target scenes to their cat
        obj_to_cat_query = self.map_obj_to_cat(query_scene)
        obj_to_cat_target = self.map_obj_to_cat(target_scene)

        # if target and query target don't match mAP is 0
        target_object = target_subscene['target']
        if obj_to_cat_query[query_object] != obj_to_cat_target[target_object]:
            return 0

        # for each context object in the query scene find the closet object in the target subscene that satisfies a
        # distance threshold. Find the percentage of such matches relative to the total number of context objects.
        visited = set()
        for context_object in context_objects:
            context_object_cat = obj_to_cat_query[context_object]
            # find the closest node in the target subscene with the same category as the current context object
            # that is not visited
            best_diff = 1000
            best_candidate = None
            for candidate in target_subscene['context_objects']:
                if candidate not in visited:
                    candidate_cat = obj_to_cat_target[candidate]
                    if candidate_cat == context_object_cat:
                        # find the difference between distances of a context object and its source (in both query and
                        # target subscenes)
                        candidate_distance = self.find_distance(target_scene, target_subscene['target'], candidate)
                        context_obj_distance = self.find_distance(query_scene, query_object, context_object)
                        relative_diff = abs(candidate_distance - context_obj_distance) / max(context_obj_distance,
                                                                                             0.00001)
                        # compute the threshold relative to the distance between the query and context object
                        if relative_diff < best_diff and relative_diff < self.metrics['distance_mAP'][1]:
                            best_diff = relative_diff
                            best_candidate = candidate

            # add the best candidate to visited nodes
            if best_candidate is not None:
                visited.add(best_candidate)
        return len(visited) / len(context_objects)

    def compute_precision_at(self, query_scene, query_scene_name, query_object, context_objects, target_subscenes,
                             top, metric, computed_precisions):
        accuracies = []
        for i in range(top):
            # the case where there are no longer results
            if i >= len(target_subscenes):
                best_acc = 0
                best_rotation = 0
            else:
                target_subscene_name = target_subscenes[i]['scene_name']
                target_object = target_subscenes[i]['target']
                # try different rotations of the query scene
                best_acc = 0
                best_rotation = 0
                for rotation in self.metrics[metric][2]:
                    # read the accuracy if you have computed it before.
                    key = '-'.join([query_scene_name, query_object, str(rotation), target_subscene_name, target_object])
                    if key in computed_precisions:
                        acc = computed_precisions[key]
                    else:
                        # rotate the query scene
                        query_scene_rotated = self.rotate_query_scene(query_scene, query_object, context_objects,
                                                                      rotation)
                        acc = self.metrics[metric][0](query_scene_rotated, query_object, context_objects,
                                                      target_subscenes[i])
                        computed_precisions[key] = acc

                    if acc > best_acc:
                        best_acc = acc
                        best_rotation = rotation

                # record the precision
                if self.metrics[metric][1] == self.precision_threshold:
                    precision_key = metric.split('_')[0] + '_precision'
                    rotation_key = metric.split('_')[0] + '_rotation'
                    target_subscenes[i][precision_key] = best_acc
                    target_subscenes[i][rotation_key] = best_rotation

            accuracies.append(best_acc)
        return np.mean(accuracies)

    def rotate_query_scene(self, query_scene, query_object, context_objects, rotation):
        q_and_context = [query_object] + context_objects
        query_scene_rotated = {obj: query_scene[obj].copy() for obj in q_and_context}

        # create the obbox of the query object
        query_obj_obbox_vertices = np.asarray(query_scene_rotated[query_object]['obbox'])
        obbox_query_obj = Box(query_obj_obbox_vertices)

        # translate and rotate each context object
        for obj in context_objects:
            # create a obbox of the context object
            context_obj_obbox_vertices = np.asarray(query_scene_rotated[obj]['obbox'])
            obbox_context_obj = Box(context_obj_obbox_vertices)

            # translate the context object according to the vector that translates the query object to the origin.
            obbox_context_obj = self.translate_obbox(obbox_context_obj, -obbox_query_obj.translation)

            # rotate the object based on the rotation angle
            obbox_context_obj = self.rotate_obbox(obbox_context_obj, rotation)

            # populate the rotated query scene
            query_scene_rotated[obj]['obbox'] = obbox_context_obj.vertices.tolist()

        # translate the query object to the origin and rotate it
        obbox_query_obj = self.translate_obbox(obbox_query_obj, -obbox_query_obj.translation)
        obbox_query_obj = self.rotate_obbox(obbox_query_obj, rotation)
        query_scene_rotated[query_object]['obbox'] = obbox_query_obj.vertices.tolist()

        return query_scene_rotated

    def compute_mAP(self, metric, query_name, model_name, experiment_id, topk=10):
        # Find the query and context objects
        query_scene_name = self.query_results[query_name]['example']['scene_name']
        query_scene = load_from_json(os.path.join(self.scene_graph_dir, self.mode, query_scene_name))
        query_object = self.query_results[query_name]['example']['query']
        context_objects = self.query_results[query_name]['example']['context_objects']
        if len(context_objects) == 0:
            mAP = 1
            print('No context objects in the query, hence the mAP is trivially 1')
        else:
            # sort the context objects in the query scene by their distance to the query (low to high)
            context_objects = self.sort_by_distance(query_scene_name, query_object, context_objects, mode=self.mode)

            # load the target subgraphs up to topk
            target_subscenes = self.query_results[query_name]['target_subscenes'][:topk]

            # initialize precision values for the target subscenes
            if self.metrics[metric][1] == self.precision_threshold:
                precision_key = metric.split('_')[0] + '_precision'
                rotation_key = metric.split('_')[0] + '_rotation'
                for target_subscene in target_subscenes:
                    target_subscene[precision_key] = None
                    target_subscene[rotation_key] = None

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
    plot_name = 'Random_BipartiteMatching.png'
    remove_model = False
    visualize_loss = False

    thresholds = np.linspace(0.05, 0.9, num=9)
    metrics = ['distance_mAP', 'overlap_mAP']

    # define paths and parameters
    mode = 'val'
    model_name = 'CategoryRandom'
    experiment_name = 'base'
    subring_matching_folder = 'subring_matching_{}'.format(experiment_name)
    query_results_input_path = '../results/matterport3d/{}/query_dict_{}_{}.json'.format(model_name, mode,
                                                                                         experiment_name)
    query_results_output_path = '../results/matterport3d/{}/query_dict_{}_{}_evaluated.json'.format(model_name, mode,
                                                                                          experiment_name)
    evaluation_path = '../results/matterport3d/evaluation.csv'
    scene_graph_dir = '../data/matterport3d/scene_graphs'
    aggregated_csv_path = '../results/matterport3d/evaluation_aggregated.csv'

    # read the results of a model and the evaluation csv file
    query_results = load_from_json(query_results_input_path)
    curr_df = pd.read_csv(evaluation_path)

    # filter the results by query if necessary
    evaluation_queries = ['all']
    if evaluation_queries[0] != 'all':
        query_results = {k: v for k, v in query_results.items() if k in evaluation_queries}

    # initialize the evaluator
    evaluator = Evaluate(query_results=query_results, evaluation_path=evaluation_path, scene_graph_dir=scene_graph_dir,
                         curr_df=curr_df, mode=mode, distance_threshold=0, overlap_threshold=0)

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
        checkpoints_dir = '../results/matterport3d/{}/{}'.format(model_name, subring_matching_folder)
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
        plt.savefig('../results/matterport3d/evaluation_plots/train_valid_losses.png')
        plt.show()


if __name__ == '__main__':
    t = time()
    main()
    duration = time() - t
    print('Evaluation took {} minutes'.format(round(duration / 60, 2)))
