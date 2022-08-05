import os
import argparse

from scripts.helper import load_from_json, write_to_json


def find_best_matching_subscene(args, answers_file_names, query, target_subscene_template):
    # for each answer query find the target subscene matching the template.
    matching_target_subscens = []
    for answer_file_name in answers_file_names:
        answers = load_from_json(os.path.join(args.query_results_path, answer_file_name))
        target_subscenes = answers[query]
        for target_subscene in target_subscenes:
            same_scene_name = target_subscene['scene_name'] == target_subscene_template['scene_name']
            same_target_obj = target_subscene['target'] == target_subscene_template['target']
            if same_scene_name and same_target_obj:
                matching_target_subscens.append(target_subscene)

    # find the best matching subscene through voting.
    votes = {}
    for target_subscene in matching_target_subscens:
        for q, t in target_subscene['correspondence'].items():
            if t != '':
                if t not in votes:
                    votes[t] = [q, 1]
                else:
                    votes[t][1] += 1

    num_users = len(answers_file_names)
    best_mathcing_subscene = {'scene_name': target_subscene_template['scene_name'], 'target': '', 'num_matches': 0}
    num_matches = 0
    correspondence = {}
    for t, (q, vote) in votes.items():
        # majority voting.
        if vote > (num_users // 2):
            if t == target_subscene_template['target']:
                best_mathcing_subscene['target'] = t
                num_matches += 1
            else:
                correspondence[t] = q
                num_matches += 1
    best_mathcing_subscene['correspondence'] = correspondence
    best_mathcing_subscene['num_matches'] = num_matches

    return best_mathcing_subscene


def find_most_popular_answer(args, query_dict):
    # create the output path.
    query_output_file_name = args.query_input_file_name.split('.')[0] + '_{}_{}.json'.format(args.mode,
                                                                                             args.experiment_name)
    query_dict_output_path = os.path.join(args.query_results_path, query_output_file_name)

    # for each query and target subscene find correspondence through voting.
    query_results = {}
    for query in args.query_list:
        query_results[query] = query_dict[query]
        query_results[query]['target_subscenes'] = []

        # load an answer dict.
        answers_file_names = [e for e in os.listdir(args.query_results_path) if e!= query_output_file_name]
        template_answer = load_from_json(os.path.join(args.query_results_path, answers_file_names[0]))

        for target_subscene_template in template_answer[query]:
            # take votes among the answers to find the best matching subscene.
            best_matching_subscene = find_best_matching_subscene(args, answers_file_names, query,
                                                                 target_subscene_template)
            query_results[query]['target_subscenes'].append(best_matching_subscene)

        # sort the target subscenes by the number of correspondences.
        query_results[query]['target_subscenes'] = sorted(query_results[query]['target_subscenes'], key=lambda x: x['num_matches'], reverse=True)

    # save the results.
    write_to_json(query_results, query_dict_output_path)


def get_args():
    parser = argparse.ArgumentParser('Format answers', add_help=False)

    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--mode', dest='mode', default='test', help='val or test')
    parser.add_argument('--query_dir', default='../queries/{}/')
    parser.add_argument('--query_input_file_name', default='query_dict_top10.json')
    # parser.add_argument('--query_list', default=["cabinet-18"], type=str, nargs='+')
    parser.add_argument('--query_list', default=["table-43", "curtain-7", "mirror-10", "cushion-20", "cabinet-18",
                                                 "chair-45", "cushion-33", "lighting-11", "picture-39", "lighting-46"],
                        type=str, nargs='+')
    parser.add_argument('--results_dir', default='../results/{}')
    parser.add_argument('--results_folder_name',  default='BruteForce')
    parser.add_argument('--experiment_name', default='brute_force')

    return parser


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def main():
    # get the arguments
    parser = argparse.ArgumentParser('Format answers', parents=[get_args()])
    args = parser.parse_args()
    adjust_paths(args, exceptions=[])
    args.query_results_path = os.path.join(args.results_dir, args.results_folder_name)
    args.query_dir = os.path.join(args.query_dir, args.mode)

    # load the query dict
    query_dict_input_path = os.path.join(args.query_dir, args.query_input_file_name)
    query_dict = load_from_json(query_dict_input_path)

    # take voting among the answers to take the best brute force model.
    find_most_popular_answer(args, query_dict)


if __name__ == '__main__':
    main()
