import numpy as np
import pandas as pd
from sklearn.metrics import auc
from time import time
from optparse import OptionParser
from matplotlib import pyplot as plt


def plot_results(metrics, model_name_exp_map, summary_results, plot_name):
    # plot results for each metric
    auc_dic = {}
    for metric in metrics:
        auc_dic[metric] = {}
        # add results for each model to the plot
        fig, ax = plt.subplots()
        for (model_name, experiment_id), display_name in model_name_exp_map.items():
            summary_resutls_model = summary_results.loc[summary_results['model_name'] == model_name]
            # choose the experiment under the model name
            this_experiment = summary_resutls_model['experiment_id'].apply(
                lambda x: x.split('-0')[0] == experiment_id)
            summary_resutls_model_experiment = summary_resutls_model.loc[this_experiment]
            # x axis represents thresholds
            x = summary_resutls_model_experiment['experiment_id'].apply(lambda x: np.float64(x.split('-')[-1])).values
            # y axis represents the mAP values
            y = summary_resutls_model_experiment[metric].values

            ax.plot(x, y, marker='*', label=display_name)
            plt.xlabel("Thresholds")
            plt.ylabel('mAP %')
            leg = ax.legend()

            # compute the area under the curve if necessary
            auc_dic[metric][display_name] = np.round(auc(x, y), 2)

        plt.grid()
        plt.savefig('../results/matterport3d/evaluation_plots/{}'.format(plot_name))
        plt.show()
        print(auc_dic)


def get_args():
    parser = OptionParser()
    parser.add_option('--mode', dest='mode', default='test', help='val|test')
    parser.add_option('--ablations', dest='ablations', default='False',
                      help='If True the ablation results are plotted.')

    (options, args) = parser.parse_args()
    return options


def main():
    # load the arguments
    args = get_args()
    ablations = args.ablations == 'True'

    # define the paths and params and model names.
    if ablations:
        aggregated_csv_path = '../results/matterport3d/evaluations/ablations/evaluation_aggregated.csv'
        plot_name = 'alignment_ablation_global.png'
        model_name_exp_map = {('LearningBased', 'AlignRank'): 'AlignRank',
                              ('LearningBased', 'AlignRank[-GNN]'): 'AlignRank[-GNN]',
                              ('LearningBased', 'AlignRank[-Align]'): 'AlignRank[-Align]',
                              ('SVDRank', 'SVDRank1D'): 'SVDRank1D',
                              ('SVDRank', 'SVDRank3D'): 'SVDRank3D'}
    else:
        aggregated_csv_path = '../results/matterport3d/evaluations/{}/evaluation_aggregated.csv'.format(args.mode)
        plot_name = 'mAP_comparisons.png'
        model_name_exp_map = {('LearningBased', 'AlignRankOracle'): 'AlignRankOracle',
                              ('LearningBased', 'AlignRank'): 'AlignRank',
                              ('GKRank', 'GKRank'): 'GKRank',
                              ('CatRank', 'CatRank'): 'CatRank',
                              ('RandomRank', 'RandomRank'): 'RandomRank'}

    # read the aggregated results and choose the metric to plot
    summary_results = pd.read_csv(aggregated_csv_path)
    metrics = ['overlap_mAP']

    # plot the quantiative results
    plot_results(metrics, model_name_exp_map, summary_results, plot_name)


if __name__ == '__main__':
    t = time()
    main()
    duration = time() - t
    print('Preparing quantitative results took {} minutes'.format(round(duration / 60, 2)))
