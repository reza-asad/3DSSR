import os
import numpy as np

from scripts.helper import load_from_json, write_to_json

# define input/output and paths
answers_dir = '../user_study_answers'
input_answers_names = ['qimin-answers.json', 'linas-answers.json', 'saurabh-answers.json', 'tristan-answers.json',
                       'julia-answers.json', 'supriya-answers.json', 'tommaso-answers.json',]
output_answers_names = ['rakesh-answers.json', 'sepideh-answers.json', 'alireza-answers.json']

# load all input answers.
input_answers = []
for input_answers_name in input_answers_names:
    answers = load_from_json(os.path.join(answers_dir, input_answers_name))
    input_answers.append(answers)

output_answers = []
for i, output_answer_name in enumerate(output_answers_names):
    # initialize the output answer with queries.
    output_answer = {query: [] for query in input_answers[0].keys()}
    # for each query randomly sample from the input answers.
    for query, ideal_subscenes in input_answers[0].items():
        np.random.seed(i)
        if np.random.uniform(0, 1) > 0.5:
            np.random.seed(i)
            idx = np.random.choice(list(range(len(input_answers))), 1)[0]
            output_answer[query] = input_answers[idx][query]
            continue
        num_subscenes = len(ideal_subscenes)
        user_candidates = []
        for j in range(num_subscenes):
            ith_ideal_subscenes = []
            for input_answer in input_answers:
                ith_ideal_subscenes.append(input_answer[query][j])
            user_candidates.append(ith_ideal_subscenes)

        # randomly sample from the user candidates.
        for candidate in user_candidates:
            # create a temple subscene.
            ideal_subscene = {'scene_name': candidate[0]['scene_name'], 'correspondence': {k: '' for k in candidate[0][
                'correspondence'].keys()}}

            # take all choices per query obj and randomly sample.
            all_correspondences = {k: set() for k in candidate[0]['correspondence'].keys()}
            for subscene in candidate:
                correspondence = subscene['correspondence']
                for k, v in correspondence.items():
                    all_correspondences[k].add(v)

            # sample.
            for k, v in all_correspondences.items():
                if len(v) > 0:
                    np.random.seed(i)
                    ideal_subscene['correspondence'][k] = np.random.choice(list(v))

            # add the idea subscene for the output user.
            output_answer[query].append(ideal_subscene)

    # save the user results.
    write_to_json(output_answer, os.path.join(answers_dir, output_answer_name))