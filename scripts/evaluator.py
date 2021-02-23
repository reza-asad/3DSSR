import os
import numpy as np
import pandas as pd
from time import time
from matplotlib import pyplot as plt

from scripts.helper import load_from_json
from scripts.box import Box
from scripts.iou import IoU


class Evaluate:
    def __init__(self, query_results, evaluation_path, scene_graph_dir, curr_df, mode, distance_threshold,
                 overlap_threshold):
        self.query_results = query_results
        self.evaluation_path = evaluation_path
        self.scene_graph_dir = scene_graph_dir
        self.curr_df = curr_df
        self.mode = mode

        self.metrics = {'distance_mAP': [self.compute_distance_match, distance_threshold, 'all'],
                        'overlap_mAP': [self.compute_overlap_match, overlap_threshold, 'all']
                        }

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

    def find_distance(self, scene_name, q, context_obj, mode):
        scene = load_from_json(os.path.join(self.scene_graph_dir, mode, scene_name))
        q_cent = np.asarray(scene[q]['obbox'][0])
        context_obj_cent = np.asarray(scene[context_obj]['obbox'][0])
        return np.linalg.norm(q_cent - context_obj_cent)

    @staticmethod
    def compute_translation(graph, source_node, nb):
        # load the source obbox and centroid
        source_obbox_vertices = np.asarray(graph[source_node]['obbox'])
        obbox_source = Box(source_obbox_vertices)

        # load the obbox and centroid of the neighbour
        nb_obbox_vertices = np.asarray(graph[nb]['obbox'])
        obbox_nb = Box(nb_obbox_vertices)

        # compute translation
        translation = obbox_source.translation - obbox_nb.translation

        return translation

    @staticmethod
    def compute_iou(graph, source_node, nb, translation):
        # load the source obbox and centroid
        source_obbox_vertices = np.asarray(graph[source_node]['obbox'])
        obbox_source = Box(source_obbox_vertices)

        # load the obbox and centroid of the neighbour
        nb_obbox_vertices = np.asarray(graph[nb]['obbox'])
        obbox_nb = Box(nb_obbox_vertices)
        transformation = np.eye(4)
        transformation[:3, 3] = translation
        obbox_nb = obbox_nb.apply_transformation(transformation)

        # compute the iou
        iou_computer = IoU(obbox_source, obbox_nb)
        iou = iou_computer.iou()

        return iou

    def compute_overlap_match(self, target_subscene, query_scene_name, query_object, context_objects):
        # load the query scene and target scenes
        query_scene = load_from_json(os.path.join(self.scene_graph_dir, self.mode, query_scene_name))
        target_scene = load_from_json(os.path.join(self.scene_graph_dir, self.mode, target_subscene['scene_name']))

        # map each object in the query and target scenes to their cat
        obj_to_cat_query = self.map_obj_to_cat(query_scene)
        obj_to_cat_target = self.map_obj_to_cat(target_scene)

        # if target and query target don't match mAP is 0
        target_object = target_subscene['target']
        if obj_to_cat_query[query_object] != obj_to_cat_target[target_object]:
            return 0

        # for each context object in the query scene find the object in the target subscene with maximum overlap that
        # satisfies the overlap threshold. Find the percentage of such matches relative to the total number of context
        # objects.
        visited = set()
        for context_object in context_objects:
            context_object_cat = obj_to_cat_query[context_object]
            # compute the translation vector and iou for the context object
            translation_q = self.compute_translation(query_scene, query_object, context_object)
            dist_q = np.linalg.norm(translation_q)
            iou_query = self.compute_iou(query_scene, query_object, context_object, translation_q)

            # find the object in the target subscene with the same category that has maximum overlap with the context
            # object.
            best_diff = 1000
            best_candidate = None
            for candidate in target_subscene['context_objects']:
                if candidate not in visited:
                    candidate_cat = obj_to_cat_target[candidate]
                    if candidate_cat == context_object_cat:
                        # compute the translation vector between the target node and the candidate node
                        translation_t = self.compute_translation(target_scene, target_object, candidate)
                        dir_t = translation_t / max(np.linalg.norm(translation_t), 0.00001)

                        # move the candidate dist_q units in the direction of dir_t.
                        translation = dir_t * dist_q

                        # use the rotated translation vector to compute iou for the candidate and target node.
                        iou_target = self.compute_iou(target_scene, target_object, candidate, translation)
                        # see if the target_iou is within a threshold of query_iou
                        relative_diff = abs(iou_query - iou_target) / max(iou_query,
                                                                          0.00001)
                        # compute the threshold relative to the distance between the query and context object
                        if relative_diff < best_diff and relative_diff < self.metrics['overlap_mAP'][1]:
                            best_diff = relative_diff
                            best_candidate = candidate

            # add the best candidate to visited nodes
            if best_candidate is not None:
                visited.add(best_candidate)
        return len(visited) / len(context_objects)

    def compute_distance_match(self, target_subscene, query_scene_name, query_object, context_objects):
        # load the query scene and target scenes
        query_scene = load_from_json(os.path.join(self.scene_graph_dir, self.mode, query_scene_name))
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
                        candidate_distance = self.find_distance(target_subscene['scene_name'], target_subscene['target'],
                                                                candidate, mode=self.mode)
                        context_obj_distance = self.find_distance(query_scene_name, query_object, context_object,
                                                                  mode=self.mode)
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

    def compute_target_to_query_corr(self, target_subscene, query_scene_name, query_object, context_objects):
        # sort the target context objects by their distance to the target object
        target_context_objects = self.sort_by_distance(target_subscene['scene_name'], target_subscene['target'],
                                                       target_subscene['context_objects'], mode=self.mode)
        # load the query scene and target scenes
        query_scene = load_from_json(os.path.join(self.scene_graph_dir, self.mode, query_scene_name))
        target_scene = load_from_json(os.path.join(self.scene_graph_dir, self.mode, target_subscene['scene_name']))

        # initialize the correspondence
        query_and_context = context_objects + [query_object]
        correspondence = {q: None for q in query_and_context}
        if query_scene[query_object]['category'][0] == target_scene[target_subscene['target']]['category'][0]:
            correspondence[query_object] = target_subscene['target']

        # assign each query context to a target context of the same category, following the computed order.
        visited = set()
        for query_c_node in context_objects:
            for target_c_node in target_context_objects:
                if target_c_node not in visited:
                    if target_scene[target_c_node]['category'][0] == query_scene[query_c_node]['category'][0]:
                        visited.add(target_c_node)
                        correspondence[query_c_node] = target_c_node

        return correspondence

    def compute_full_precision_at(self, top, metric, target_subscenes, query_scene_name, query_object, context_objects,
                                  computed_precisions, query_to_target_map):
        accuracies = []
        for i in range(top):
            # the case where there are no longer results
            if i >= len(target_subscenes):
                acc = 0
            else:
                # compute the correspondence between the current query and target scene if its not already computed
                target_subscene_name = target_subscenes[i]['scene_name']
                if target_subscene_name not in query_to_target_map:
                    query_to_target_map[target_subscene_name] = self.compute_target_to_query_corr(target_subscenes[i],
                                                                                                  query_scene_name,
                                                                                                  query_object,
                                                                                                  context_objects)
                query_to_target_corr = query_to_target_map[target_subscene_name]

                # consider each node in the query subgraph as a potential query
                query_and_context = context_objects + [query_object]
                target_scene_accuracies = []
                for j in range(len(query_and_context)):
                    # fix a query and a set of context objects
                    curr_query_object = query_and_context[j]
                    curr_context_objects = query_and_context[:j] + query_and_context[j+1:]

                    # find the corresponding target and cotnext objects
                    target_subscene = target_subscenes[i].copy()
                    target_subscene['target'] = query_to_target_corr[curr_query_object]
                    target_subscene['context_objects'] = [query_to_target_corr[n] for n in query_to_target_corr.keys()
                                                          if (query_to_target_corr[n] != target_subscene['target']) and
                                                          (query_to_target_corr[n] is not None)]

                    # if there is zero matching target object or zero matching context object, the acc is 0
                    if (target_subscene['target'] is None) or (len(target_subscene['context_objects']) == 0):
                        acc = 0
                    else:
                        # read the accuracy if you have computed it before.
                        key = '-'.join([query_scene_name, curr_query_object, target_subscene_name, target_subscene['target']])
                        if key in computed_precisions:
                            acc = computed_precisions[key]
                        else:
                            acc = self.metrics[metric][0](target_subscene, query_scene_name, curr_query_object,
                                                          curr_context_objects)
                            computed_precisions[key] = acc
                    target_scene_accuracies.append(acc)

                # average the computed accuracies for all possible target nodes in the target scene
                acc = np.mean(target_scene_accuracies)

            accuracies.append(acc)
        return np.mean(accuracies)

    def compute_precision_at(self, top, metric, target_subscenes, query_scene_name, query_object, context_objects,
                             computed_precisions):
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
                    acc = self.metrics[metric][0](target_subscenes[i], query_scene_name, query_object, context_objects)
                    computed_precisions[key] = acc

            accuracies.append(acc)
        return np.mean(accuracies)

    def compute_mAP(self, metric, query_name, model_name, experiment_id, topk=10):
        # Find the query and context objects
        query_scene_name = self.query_results[query_name]['example']['scene_name']
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

            # memorize the correspondence between each query and target scene
            target_to_query_map = {}

            # memorize the precisions that you already computed in a dict
            computed_precisions = {}
            precision_at = {i: 0 for i in range(1, topk+1)}
            for top in precision_at.keys():
                # case where we average precision over each node being the center of the ring.
                if self.metrics[metric][2] == 'all':
                    precision_at[top] = self.compute_full_precision_at(top, metric, target_subscenes, query_scene_name,
                                                                       query_object, context_objects,
                                                                       computed_precisions, target_to_query_map)
                else:
                    precision_at[top] = self.compute_precision_at(top, metric, target_subscenes, query_scene_name,
                                                                  query_object, context_objects, computed_precisions)
            mAP = np.mean(list(precision_at.values()))
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
    run_aggregation = True
    plot_comparisons = True
    plot_name = 'GK++_GNN_Randoms.png'
    remove_model = False
    visualize_loss = False

    thresholds = np.linspace(0.1, 0.9, num=9)
    metrics = ['distance_mAP', 'overlap_mAP']

    # define paths and parameters
    mode = 'val'
    model_name = 'GNN'
    experiment_name = 'cat_dir'
    subring_matching_folder = 'subring_matching_{}'.format(experiment_name)
    query_results_path = '../results/matterport3d/{}/query_dict_{}_{}.json'.format(model_name, mode, experiment_name)
    evaluation_path = '../results/matterport3d/evaluation.csv'
    scene_graph_dir = '../data/matterport3d/scene_graphs'
    aggregated_csv_path = '../results/matterport3d/evaluation_aggregated.csv'

    # read the results of a model and the evaluation csv file
    query_results = load_from_json(query_results_path)
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
            for threshold in thresholds:
                experiment_id = experiment_name + '-' + str(np.round(threshold, 3))
                evaluator.add_to_tabular(model_name, query_name, experiment_id)
                for metric in metrics:
                    evaluator.metrics[metric][1] = threshold
                    evaluator.compute_mAP(metric, query_name, model_name, experiment_id)
        evaluator.to_tabular()

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
        curr_df = curr_df[curr_df['model_name'] != model_name]
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
        # plt.savefig('../results/matterport3d/evaluation_plots/train_valid_losses.png')
        plt.show()


if __name__ == '__main__':
    t = time()
    main()
    duration = time() - t
    print('Evaluation took {} minutes'.format(round(duration / 60, 2)))
