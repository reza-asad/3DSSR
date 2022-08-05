import os
import argparse
import pandas as pd
import numpy as np

from scripts.helper import load_from_json
from find_gt_subscene_candidatesv2 import build_subscene_key


def add_gt(args):
    # load the input df
    df_input = pd.read_csv(os.path.join(args.rendering_path, 'metadata_{}.csv'.format(args.batch_id)))

    # convert the gt labels to string
    df_input['gt_label'] = df_input['gt_label'].apply(lambda x: label_map[x])

    # load the user responses.
    df_responses = pd.read_csv(os.path.join(args.rendering_path, 'metadata_{}_responses.csv'.format(args.batch_id)))

    # add gt label from input to responses and record the results.
    df_comb = pd.merge(df_input[['img_names', 'gt_label']], df_responses, left_on='img_names', right_on='Input.img_names')
    df_comb['Input.gt_label'] = df_comb['gt_label']
    df_comb.drop(['gt_label', 'img_names'], axis=1, inplace=True)
    df_comb.to_csv(os.path.join(args.rendering_path, 'metadata_{}_responses.csv'.format(args.batch_id)), index=False)


def reject_approve(args):
    # load the user responses.
    df_responses = pd.read_csv(os.path.join(args.rendering_path, 'metadata_{}_responses.csv'.format(args.batch_id)))

    # group by the records by worker id and compute accuracy.
    def compute_accuracy(g):
        num_correct = np.sum(g['Input.gt_label'] == g['Answer.category.label'])
        num_total = len(g)
        result = {'accuracy': num_correct / num_total * 100, 'num_responses': num_total}

        return pd.Series(result)

    # find per user accuracy.
    grouped = df_responses.groupby(['WorkerId'])
    per_user_acc = grouped.apply(compute_accuracy)
    print(per_user_acc)

    # populate the approve and reject columns.
    worker_ids = per_user_acc.index.tolist()
    for worker_id in worker_ids:
        # check if the worker is approved to rejected.
        worker_records = df_responses['WorkerId'] == worker_id
        wrong_answers = df_responses['Input.gt_label'] != df_responses['Answer.category.label']
        if per_user_acc.loc[worker_id, 'accuracy'] < args.acceptance_acc_threshold:
            reject_msg = '{} questions were responded and accuracy was {}. Unfortunately, the score is too low for ' \
                         'us to accept all your responses. Please read the instructions more carefully.'.\
                format(int(per_user_acc.loc[worker_id, 'num_responses']),
                       np.round(per_user_acc.loc[worker_id, 'accuracy'], 2))
            df_responses.loc[worker_records & wrong_answers, 'Reject'] = reject_msg
            df_responses.loc[worker_records & (~wrong_answers), 'Approve'] = 'X'
        else:
            df_responses.loc[worker_records, 'Approve'] = 'X'

    # print the number of approvals.
    num_approved = df_responses.loc[df_responses['Approve'] == 'X'].shape[0]
    print('{}/{} was apporved'.format(num_approved, len(df_responses)))

    # record the final results.
    df_responses.to_csv(os.path.join(args.rendering_path, 'metadata_{}_responses.csv'.format(args.batch_id)),
                        index=False)


def print_stats(args):
    # load the user responses.
    # df_responses = pd.read_csv(os.path.join(args.rendering_path, 'metadata_{}_responses.csv'.format(args.batch_id)))
    #
    # # filter to approved assignments.
    # df_responses = df_responses.loc[df_responses['Approve'] == 'X']
    #
    # # group the results by the image name and count the number of approved responses per image-pair.
    # df_responses['count'] = 0
    # grouped = df_responses.groupby(['Input.img_names'])
    # print('Unique number of approved responses per image: {}'.format(grouped.count()['count'].unique()))

    # count the number of unique responses.
    all_worker_ids = set()
    for i in range(args.num_unique_answers):
        df_responses = pd.read_csv(os.path.join(args.rendering_path, 'metadata_{}_responses.csv'.format(i+1)))
        df_responses = df_responses.loc[df_responses['Approve'] == 'X']
        curr_worker_ids = set(df_responses['WorkerId'].values)
        all_worker_ids = all_worker_ids.union(curr_worker_ids)

    print('Number of unique users: {}'.format(len(all_worker_ids)))


def combine_answers(args):
    # initialize the values for the combined frame.
    frame_dict = {'img_names': []}
    for i in range(args.num_unique_answers):
        frame_dict['user_response_{}'.format(i+1)] = []

    for batch_id in args.batch_ids:
        # if batch_id != 1:
        #     continue
        # find the number of expected responses.
        df_metadaa = pd.read_csv(os.path.join(args.rendering_path, 'metadata_{}.csv'.format(batch_id)))
        num_records = len(df_metadaa) * args.num_unique_answers

        # load the responses and filter to only include approved responses.
        df_responses = pd.read_csv(os.path.join(args.rendering_path, 'metadata_{}_responses.csv'.format(batch_id)))
        df_responses = df_responses.loc[df_responses['Approve'] == 'X']
        print('{}/{} records are approved'.format(len(df_responses), num_records))

        # map the labels back to 0, 1 and 2.
        df_responses['Answer.category.label'] = df_responses['Answer.category.label'].apply(lambda x: label_map_inv[x])
        df_responses['Input.gt_label'] = df_responses['Input.gt_label'].apply(lambda x: label_map_inv[x])

        # find the questions with less than expected unique answers.
        df_responses['count'] = 0
        grouped = df_responses.groupby(['Input.img_names'])
        img_name_count = grouped.count()[['count']].reset_index()
        not_complete = img_name_count['count'] < args.num_unique_answers
        incomplete_img_names = img_name_count.loc[not_complete, 'Input.img_names']
        num_remaining = args.num_unique_answers - img_name_count.loc[not_complete, 'count']
        img_name_to_num_remaining_records = dict(zip(incomplete_img_names, num_remaining))
        df_responses.drop(columns=['count'], inplace=True)

        # fill in the unapproved responses with gt.
        img_names = []
        gt_labels = []
        worker_ids = []
        for img_name, num_remaining in img_name_to_num_remaining_records.items():
            for i in range(num_remaining):
                img_names.append(img_name)
                gt_labels.append(df_metadaa.loc[df_metadaa['img_names'] == img_name, 'gt_label'].values[0])
                worker_ids.append('root')
        df_unapproved = pd.DataFrame({'Input.img_names': img_names, 'Answer.category.label': gt_labels,
                                      'WorkerId': worker_ids})
        df_responses = pd.concat([df_unapproved, df_responses])
        print('{}/{} records are ready for aggregation'.format(len(df_responses), num_records))
        assert len(df_responses) == num_records

        # sort the result by the img_name and worker id
        df_responses.sort_values(by=['Input.img_names', 'WorkerId'], inplace=True)

        # flatten the responses per group.
        grouped = df_responses.groupby(['Input.img_names'])
        for img_name, group in grouped:
            frame_dict['img_names'].append(img_name)
            for i, response in enumerate(group['Answer.category.label'].values):
                frame_dict['user_response_{}'.format(i+1)].append(response)

    # create the combined frame and save it.
    df_combined = pd.DataFrame(frame_dict)
    for i in range(args.num_unique_answers):
        df_combined['user_response_{}'.format(i+1)] = df_combined['user_response_{}'.format(i+1)].astype(int)
    # pd.set_option('display.max_columns', None)
    df_combined.to_csv(os.path.join(args.rendering_path, 'all_user_responses.csv'), index=False)


def load_retrieved_results(args, model_name, retrieved_results):
    # take topk results from each model.
    retrieved_subscenes = {}
    for config_name, config_info in retrieved_results.items():
        if config_name == model_name:
            print('Extracting subscenes from {}'.format(config_name))
            # create the path to the model's retrieved results.
            for k, v in config_info.items():
                vars(args)[k] = v
            adjust_paths(args, exceptions=[])
            query_results_dir = os.path.join(args.cp_dir.format(args.model_name), args.results_folder_name)
            query_results_input_file_name = args.query_input_file_name.split('.')[0] + '_{}_{}.json' \
                .format(args.mode, args.experiment_name)
            query_results_input_path = os.path.join(query_results_dir, query_results_input_file_name)

            # load the results of the model
            query_result = load_from_json(query_results_input_path)

            # for each query take the top 10 results from the query result
            retrieved_subscenes = {query: [] for query in args.query_list}
            for query, results_info in query_result.items():
                if query in args.query_list:
                    # build the query key.
                    query_key = 'query_{}_{}.png'.format(results_info['example']['scene_name'].split('.')[0],
                                                         results_info['example']['query'])
                    target_subscenes = results_info['target_subscenes'][:args.topk]
                    for target_subscene in target_subscenes:
                        target_key = build_subscene_key(target_subscene)
                        img_name = '{}*{}.png'.format(query_key, target_key)
                        retrieved_subscenes[query].append(img_name)

    return retrieved_subscenes


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


def compute_dcg_per_user_per_model(img_name_to_rel, retrieved_subscenes):
    # compute dcg per query.
    dc_all_queries = []
    queries = sorted(retrieved_subscenes.keys())
    for query in queries:
        ranked_results = []
        img_names = retrieved_subscenes[query]
        for img_name in img_names:
            if img_name in img_name_to_rel:
                ranked_results.append([img_name, img_name_to_rel[img_name]])
            else:
                ranked_results.append([None, 0])

        # compute dcg for the query.
        dcg = compute_dcg(ranked_results)
        dc_all_queries.append(dcg)

    return np.asarray(dc_all_queries)


def compute_ideal_dcg(args, df_combined, user_id, retrieved_subscenes):
    # compute ideal dcg per query.
    dc_all_queries = []
    queries = sorted(retrieved_subscenes.keys())
    for query in queries:
        query_img_name = retrieved_subscenes[query][0].split('*')[0]
        curr_query = df_combined['img_names'].apply(lambda x: query_img_name in x)
        scores = df_combined.loc[curr_query, user_id]
        top_scores = scores.sort_values(ascending=False).values[:args.topk]
        ideal_results = [(None, score) for score in top_scores]
        dcg = compute_dcg(ideal_results)
        dc_all_queries.append(dcg)

    return np.asarray(dc_all_queries)


def compute_ndcg(args, retrieved_results):
    # map model name to its results.
    model_name_to_results = {}
    for model_name in args.model_names:
        retrieved_subscenes = load_retrieved_results(args, model_name, retrieved_results)
        model_name_to_results[model_name] = retrieved_subscenes

    # load the user answers.
    df_combined = pd.read_csv(os.path.join(args.rendering_path, 'all_user_responses.csv'))

    # for each user group compute ndcg.
    user_ids = ['user_response_{}'.format(i+1) for i in range(args.num_unique_answers)]
    model_name_to_ndcg = {model_name: [] for model_name in model_name_to_results.keys()}
    for user_id in user_ids:
        # find a mapping from img name to its relevance score.
        img_name_to_rel = dict(zip(df_combined['img_names'], df_combined[user_id]))

        # compute the ideal dcg and the one for each model.
        for model_name in model_name_to_results.keys():
            # compute the ideal dcg.
            dcg_ideal_all_queries = compute_ideal_dcg(args, df_combined, user_id, model_name_to_results[model_name])

            dcg_all_queries = compute_dcg_per_user_per_model(img_name_to_rel, model_name_to_results[model_name])
            ndcg = dcg_all_queries / dcg_ideal_all_queries

            # average the ndcg across queries.
            ndcg = np.mean(ndcg)
            model_name_to_ndcg[model_name].append(ndcg)

    # average the ndcg across users.
    for model_name, ndcg_scores in model_name_to_ndcg.items():
        print('For {} NDCG mean and std are {} {}'.format(model_name, np.mean(ndcg_scores), np.std(ndcg_scores)))


def get_args():
    parser = argparse.ArgumentParser('Compiling list of ground truth 3D subscenes', add_help=False)

    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--action', default='ndcg', help='add gt | reject approve | stats | combine | ndcg')
    parser.add_argument('--mode', dest='mode', default='test', help='val or test')
    parser.add_argument('--results_dir', default='../results/{}/')
    parser.add_argument('--results_folder_name',  default='UserStudy')
    parser.add_argument('--experiment_name', default='GroundTruthSubscenes')
    parser.add_argument('--rendering_folder_name', default='rendered_results')
    parser.add_argument('--num_unique_answers', default=8, type=int)
    parser.add_argument('--batch_id', default=12, type=int)
    parser.add_argument('--batch_ids', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], type=int, nargs='+')
    parser.add_argument('--acceptance_acc_threshold', default=40, type=float)
    parser.add_argument('--topk', default=10, type=int)
    parser.add_argument('--query_input_file_name', default='query_dict_top10.json')
    parser.add_argument('--model_config_filename', default='3dssr_model_configs.json')
    # parser.add_argument('--model_names', default=["sup_pt_config", "dino_config", "csc_config", "random_config",
    #                                               "cat_config", "oracle_config", "gk_config", "brute_force_config"],
    #                     type=str, nargs='+')
    parser.add_argument('--model_names', default=["sup_pt_full_config", "dino_full_config", "random_full_config",
                                                  "cat_full_config", "oracle_full_config", "gk_full_config",
                                                  "brute_force_config"],
                        type=str, nargs='+')
    parser.add_argument('--query_list', default=["table-43", "curtain-7", "mirror-10", "cushion-20", "cabinet-18",
                                                 "chair-45", "cushion-33", "lighting-11", "picture-39", "lighting-46"],
                        type=str, nargs='+')

    return parser


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def main():
    # get the arguments
    parser = argparse.ArgumentParser('Compiling list of ground truth 3D subscenes', parents=[get_args()])
    args = parser.parse_args()
    adjust_paths(args, exceptions=[])

    # make sure to retrieve data from the requested mode
    args.rendering_path = os.path.join(args.results_dir, args.results_folder_name, args.rendering_folder_name,
                                       args.mode, args.experiment_name)

    if args.action == 'add gt':
        add_gt(args)
    elif args.action == 'reject approve':
        reject_approve(args)
    elif args.action == 'stats':
        print_stats(args)
    elif args.action == 'combine':
        combine_answers(args)
    elif args.action == 'ndcg':
        retrieved_results = load_from_json(args.model_config_filename)
        compute_ndcg(args, retrieved_results)
    else:
        raise NotImplementedError('Action {} is not implemented'.format(args.action))


if __name__ == '__main__':
    label_map = {0: 'Not Similar', 1: 'Somewhat Similar', 2: 'Very Similar'}
    label_map_inv = {label: i for i, label in label_map.items()}
    main()
