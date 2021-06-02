import os
from optparse import OptionParser
from subprocess import Popen
from time import sleep


def get_args():
    parser = OptionParser()
    parser.add_option('--mode', dest='mode', default='test', help='val|test')
    parser.add_option('--ablations', dest='ablations', default='False',
                      help='If True the evaluation results are stored in the ablation folder.')
    (options, args) = parser.parse_args()
    return options


def main():
    # load the arguments
    args = get_args()
    ablations = args.ablations == 'True'
    # run evaluations to compare AlignRank against baselines.
    if ablations:
        evalaution_base_path = '../results/matterport3d/evaluations/ablation'
        model_name_experiment = [('LearningBased', 'AlignRank'),
                                 ('LearningBased', 'AlignRank[-GNN]'),
                                 ('LearningBased', 'AlignRank[-Align]'),
                                 ('SVDRank', 'SVDRank1D'),
                                 ('SVDRank', 'SVDRank3D')]
    else:
        evalaution_base_path = '../results/matterport3d/evaluations/{}'.format(args.mode)
        model_name_experiment = [('LearningBased', 'AlignRankOracle'),
                                 ('LearningBased', 'AlignRank'),
                                 ('GKRank', 'GKRank'),
                                 ('CatRank', 'CatRank'),
                                 ('RandomRank', 'RandomRank')]

    # delete previous evaluation results
    evaluation_csv_path = os.path.join(evalaution_base_path, 'evaluation.csv')
    evaluation_summary_path = os.path.join(evalaution_base_path, 'evaluation_aggregated.csv')
    for path in [evaluation_csv_path, evaluation_summary_path]:
        if os.path.exists(path):
            os.remove(path)

    for model_name, experiment_name in model_name_experiment:
        command = 'python3 evaluator.py --mode {} --ablations {} --model_name {} --experiment_name {}'\
            .format(args.mode, args.ablations, model_name, experiment_name)
        evaluation_process = Popen(command, shell=True)
        while evaluation_process.poll() is None:
            sleep(10)


if __name__ == '__main__':
    main()
