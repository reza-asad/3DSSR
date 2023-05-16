import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import auc


def adjust_paths(args, exceptions):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in vars(args).items():
        if (type(v) is str) and ('/' in v) and k not in exceptions:
            v = v.format(args.dataset)
            vars(args)[k] = os.path.join(base_dir, v)


def get_args():
    parser = argparse.ArgumentParser('Metric Correlation', add_help=False)
    parser.add_argument('--dataset', default='matterport3d')
    parser.add_argument('--metric', default='cat_cd_mAP_bi')
    parser.add_argument('--query_list', default=["table-43", "curtain-7", "mirror-10", "cushion-20", "cabinet-18",
                                                 "chair-45", "cushion-33", "lighting-11", "picture-39", "lighting-46"],
                        type=str, nargs='+')
    parser.add_argument('--evaluation_csv_path', default='../results/{}/evaluations/test/evaluation.csv', )

    return parser


def main():
    parser = argparse.ArgumentParser('Metric Correlation', parents=[get_args()])
    args = parser.parse_args()
    adjust_paths(args, exceptions=[])

    # open the evaluation csv
    df = pd.read_csv(args.evaluation_csv_path)

    # filter the csv to only include the models of interest.
    df['model_name_experiment_name'] = df[['model_name', 'experiment_id']].apply\
        (lambda x: '{}-{}'.format(x['model_name'], x['experiment_id'].split('-')[0]), axis=1)
    accepted_models = df['model_name_experiment_name'].apply(lambda x: x in ndcg_dict.keys())
    df = df.loc[accepted_models]

    # filter the csv to only include queries from the query list.
    accepted_query = df['query_name'].apply(lambda x: x in set(args.query_list))
    df = df.loc[accepted_query]

    # filter the csv to only include the metric of interest.
    base_keys = ['query_name', 'model_name', 'experiment_id']
    metric_keys = [e for e in df.keys() if args.metric in e]
    df = df[base_keys + metric_keys]

    # compute the mean metric for each model across the selected queries.
    groups = df.groupby(['model_name', 'experiment_id'])
    df_mean = groups.agg({metric: 'mean' for metric in metric_keys})
    df_mean.reset_index(inplace=True)
    for metric in metric_keys:
        df_mean[metric] = df_mean[metric].apply(lambda x: np.round(x * 100, 3))

    summary_resutls = df_mean.sort_values(by=['model_name', 'experiment_id']).reset_index(drop=True)
    summary_resutls['model_name_experiment_name'] = summary_resutls[['model_name', 'experiment_id']].apply\
        (lambda x: '{}-{}'.format(x['model_name'], x['experiment_id'].split('-')[0]), axis=1)

    # compute the AUC for each model.
    auc_dic = {metric: None for metric in metric_keys}
    for metric in metric_keys:
        # given a metric compute auc for each model
        auc_dic[metric] = {}
        for k in ndcg_dict.keys():
            # find x and y for auc
            y = summary_resutls.loc[summary_resutls['model_name_experiment_name'] == k, metric].values
            x = summary_resutls.loc[summary_resutls['model_name_experiment_name'] == k, 'experiment_id'].values
            x = [float(e.split('-')[-1]) for e in x]
            auc_dic[metric][k] = auc(x, y)

    # compute correlation between each metric and ndcg.
    correlation_dict = {metric: None for metric in metric_keys}
    for metric in metric_keys:
        x = [ndcg_dict[k] for k in sorted(ndcg_dict.keys())]
        y = [auc_dic[metric][k] for k in sorted(ndcg_dict.keys())]
        correlation_dict[metric] = np.corrcoef(x, y)[0, 1]

    print(correlation_dict)


if __name__ == '__main__':
    # # ndcg top10 model vs top10 ndcg dictionary.
    # ndcg_dict = {
    #     'LearningBased-3D_DINO_point_transformer': 0.29,
    #     'LearningBased-supervised_point_transformer': 0.31,
    #     'OracleRank-OracleRank': 0.31,
    #     'CatRank-CatRank': 0.22,
    #     'PointTransformerSeg-CSC_point_transformer': 0.21,
    #     'GKRank-GKRank': 0.21,
    #     'RandomRank-RandomRank': 0.05
    # }
    # # ndcg full_models vs top10 ndcg dictionary.
    # ndcg_dict = {
    #     'LearningBased-3D_DINO_point_transformer_full': 0.29,
    #     'LearningBased-supervised_point_transformer_full': 0.31,
    #     'OracleRank-OracleRankFull': 0.31,
    #     'CatRank-CatRankFull': 0.22,
    #     # 'PointTransformerSeg-CSC_point_transformer': 0.21,
    #     'GKRank-GKRankFull': 0.21,
    #     'RandomRank-RandomRankFull': 0.05
    # }

    # ndcg full models vs full ndcg dictionary.
    ndcg_dict = {
        'LearningBased-3D_DINO_point_transformer_full': 0.16,
        'LearningBased-supervised_point_transformer_full': 0.13,
        'OracleRank-OracleRank': 0.28,
        'CatRank-CatRankFull': 0.10,
        # 'PointTransformerSeg-CSC_point_transformer': 0.21,
        'GKRank-GKRankFull': 0.07,
        'RandomRank-RandomRankFull': 0.00
    }
    main()


