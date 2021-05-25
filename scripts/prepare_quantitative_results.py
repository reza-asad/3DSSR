import numpy as np
import pandas as pd
from sklearn.metrics import auc
from time import time
from matplotlib import pyplot as plt


def plot_results(metrics, model_name_exp_map, summary_results, plot_name, with_auc):
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
                lambda x: x.split('-')[0] == experiment_id)
            summary_resutls_model_experiment = summary_resutls_model.loc[this_experiment]
            # x axis represents thresholds
            x = summary_resutls_model_experiment['experiment_id'].apply(lambda x: np.float(x.split('-')[-1])).values
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


def main():
    # define the paths and params
    mode = 'test'
    aggregated_csv_path = '../results/matterport3d/evaluations/{}/evaluation_aggregated.csv'.format(mode)

    # read the aggregated results and choose the metric to plot
    summary_results = pd.read_csv(aggregated_csv_path)
    metrics = ['overlap_mAP']

    # define the models that you would like to plot and the name that you will display
    # model_name_exp_map = {('LearningBased', 'lstm_with_cats'): 'AlignRankOracle',
    #                       ('LearningBased', 'lstm_top1_predictions'): 'AlignRank',
    #                       ('GK++', 'base'): 'GKRank',
    #                       ('CategoryRandom', 'base'): 'CatRank',
    #                       ('LearningBased', 'lstm_top1_predictions_kmeans'): 'AlignRankKmeans',
    #                       ('Random', 'base'): 'Random'}

    model_name_exp_map = {('LearningBased', 'lstm_top1_predictions'): 'AlignRank',
                          ('LearningBased', 'lstm_top1_predictions_linear_classifier'): 'AlignRank[-GNN]',
                          ('LearningBased', 'no_alignment'): 'AlignRank[-Align]',
                          ('SVDRank', 'with_cat_predictions_3d'): 'SVDRank3D',
                          ('SVDRank', 'with_cat_predictions_1d'): 'SVDRank1D'}

    plot_name = 'alignment_ablation_global.png'.format(mode)

    # plot the quantiative results
    plot_results(metrics, model_name_exp_map, summary_results, plot_name, with_auc=True)


if __name__ == '__main__':
    t = time()
    main()
    duration = time() - t
    print('Preparing quantitative results took {} minutes'.format(round(duration / 60, 2)))
