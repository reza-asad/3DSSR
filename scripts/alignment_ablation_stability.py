import numpy as np
import pandas as pd
from sklearn.metrics import auc
from matplotlib import pyplot as plt


def find_experiment_name(x):
    potential_experiment_name = x.split('-')[0]
    if potential_experiment_name.split('_')[-1].isdigit():
        potential_experiment_name = '_'.join(potential_experiment_name.split('_')[:-1])
    return potential_experiment_name


def find_theta(x):
    potential_theta = x.split('-')[0].split('_')[-1]
    if potential_theta.isdigit():
        return potential_theta
    else:
        return '0'


def compute_auc(g):
    x = g['threshold'].values
    y = g['overlap_mAP'].values * 100

    return auc(x, y)


def compute_variance(g):
    return np.var(g['AUC'])


def main():
    # define the paths
    pd.set_option('max_columns', None)
    evaluation_path = '../results/matterport3d/evaluations/ablation/evaluation.csv'

    # read the evaluation csv
    df = pd.read_csv(evaluation_path)

    # find the experiment_name, theta and IoU threshold for each record
    df['experiment_name'] = df['experiment_id'].apply(lambda x: find_experiment_name(x))
    df['theta'] = df['experiment_id'].apply(lambda x: find_theta(x))
    df['threshold'] = df['experiment_id'].apply(lambda x: float(x.split('-')[-1]))

    # group the results by query name, experiment_name and theta. for each group compute the AUC.
    grouped = df.groupby(['query_name', 'experiment_name', 'theta'])

    # compute the AUC for each group
    agg_auc = grouped.apply(lambda g: compute_auc(g))
    agg_auc = agg_auc.reset_index()
    agg_auc = agg_auc.rename(columns={0: 'AUC'})

    # group the results by query and experiment names
    grouped = agg_auc.groupby(['query_name', 'experiment_name'])

    # compute the variance
    agg_auc_variance = grouped.apply(lambda g: compute_variance(g))
    agg_auc_variance = agg_auc_variance.reset_index()
    agg_auc_variance = agg_auc_variance.rename(columns={0: 'std'})

    # map each query to an index
    query_names = agg_auc_variance['query_name'].unique()
    query_name_to_idx = dict(zip(query_names, range(len(query_names))))
    agg_auc_variance['query_idx'] = agg_auc_variance['query_name'].apply(lambda q: query_name_to_idx[q])

    # sort results by query index
    agg_auc_variance = agg_auc_variance.sort_values(by='query_idx')

    # extract the variances for each experiment name.
    with_alignment = agg_auc_variance[agg_auc_variance['experiment_name'] == 'lstm_with_cats']
    without_alignment = agg_auc_variance[agg_auc_variance['experiment_name'] == 'no_alignment_with_cats']
    # experiments = {'AlignRankOracle': with_alignment, 'AlignRankOracle-': without_alignment}
    all_data = [with_alignment['std'].values.tolist()] + [without_alignment['std'].values.tolist()]
    labels = ['', '']
    # add the variances to a plot
    ig, ax = plt.subplots()
    bplot1 = ax.boxplot(all_data,
                        positions=[1, 1.3],
                        vert=False,  # vertical box alignment
                        patch_artist=True,  # fill with color
                        labels=labels)
    # ax1.set_title('Rectangular box plot')
    colors = ['lightblue', 'pink']
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)

    ax.xaxis.grid(True)
    ax.set_ylabel('Models')
    ax.set_xlabel('Varianve of AUC')
    leg = ax.legend([bplot1["boxes"][0], bplot1["boxes"][1]], ['AlignRankOracle', 'AlignRankOracle[-Align]'],
                    loc='upper right')
    ax.set_ylim(0.8, 1.6)
    plt.savefig('../results/matterport3d/evaluation_plots/{}'.format('alignment_ablation_local_transposed.png'))
    plt.show()

    # print(agg_auc.head())


if __name__ == '__main__':
    main()
