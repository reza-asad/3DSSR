import numpy as np
import pandas as pd
from sklearn.metrics import auc
from time import time
from optparse import OptionParser
from matplotlib import pyplot as plt


def plot_results(metrics, model_name_exp_map, summary_results, mode, name, same_scale=False, title=None):
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

            if same_scale:
                # 38 or 55
                ax.set_ylim([0, 80])
                ax.set_title('CD threshold: {}%'.format(metric.split('_')[-1]))
            elif title is not None:
                ax.set_title(title)
            ax.plot(x, y, marker='*', label=display_name)
            plt.xlabel("IoU thresholds")
            plt.ylabel('mAP %')
            leg = ax.legend()

            # compute the area under the curve if necessary
            auc_dic[metric][display_name] = np.round(auc(x, y), 2)

        # plt.grid()
        # plt.savefig('../results/matterport3d/evaluation_plots/{}_{}_{}.png'.format(mode, metric, name))
        # plt.show()

    for k, v in auc_dic.items():
        print('{}: {}'.format(k, v))


def get_args():
    parser = OptionParser()
    parser.add_option('--mode', dest='mode', default='test', help='val|test')
    parser.add_option('--ablations', dest='ablations', default='False',
                      help='If True the ablation results are plotted.')
    parser.add_option('--object_retrieval', dest='object_retrieval', default='False')

    (options, args) = parser.parse_args()
    return options


def main():
    # load the arguments
    args = get_args()
    ablations = args.ablations == 'True'
    object_retrieval = args.object_retrieval == 'True'

    evaluation_aggregate_file_name = 'evaluation_aggregated.csv'
    if object_retrieval:
        evaluation_aggregate_file_name = 'evaluation_aggregated_objects.csv'

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
        aggregated_csv_path = '../results/matterport3d/evaluations/{}/{}'.format(args.mode,
                                                                                 evaluation_aggregate_file_name)
        # For cat + IoU
        # model_name_exp_map = {
        #     ('OracleRank', 'OracleRank'): 'OracleCatRank[+IoU]',
        #     ('CatRank', 'CatRank'): 'OracleCatRank',
        #     ('GKRank', 'GKRank'): 'OracleGKRank',
        #     ('LearningBased', 'supervised_point_transformer'): 'TransformerRank',
        #     ('LearningBased', '3D_DINO_point_transformer'): 'PointCropRank',
        #     ('PointTransformerSeg', 'CSC_point_transformer'): 'CSCRank',
        #     ('LearningBased', 'supervised_point_transformer_with_boxes'): 'TransformerRankV2',
        #     ('LearningBased', '3D_DINO_point_transformer_with_boxes'): 'PointCropRankV2',
        #     ('RandomRank', 'RandomRank'): 'RandomRank',
        #     ('PointTransformerSeg', 'CSC_point_transformer_with_boxes'): 'CSCRankV2'
        # }
        # For cd + IoU
        # model_name_exp_map = {
        #     ('LearningBased', '3D_DINO_point_transformer'): 'PointCropRank',
            # ('LearningBased', '3D_DINO_point_transformer_with_boxes_large_nms'): 'PointCropRankV2',
            # ('LearningBased', 'supervised_point_transformer'): 'TransformerRank',
        #     ('LearningBased', 'supervised_point_transformer_with_boxes_large_nms'): 'TransformerRankV2',
        #     ('PointTransformerSeg', 'CSC_point_transformer'): 'CSCRank',
        #     ('OracleRank', 'OracleRank'): 'OracleCatRank[+IoU]',
        #     ('GKRank', 'GKRank'): 'OracleGKRank',
        #     ('CatRank', 'CatRank'): 'OracleCatRank',
        #     ('RandomRank', 'RandomRank'): 'RandomRank',
        #     ('PointTransformerSeg', 'CSC_point_transformer_with_boxes_large_nms'): 'CSCRankV2'
        # }
        # For cd + distance + angle + cat + IoU with rotation
        # model_name_exp_map = {
            # ('LearningBased', '3D_DINO_point_transformer'): 'PointCropRank',
            # ('LearningBased', '3D_DINO_point_transformer_with_rotation'): 'PointCropRankRot',
            # ('LearningBased', 'supervised_point_transformer'): 'TransformerRank',
            # ('LearningBased', 'supervised_point_transformer_with_rotation'): 'TransformerRankRot',
            # ('PointTransformerSeg', 'CSC_point_transformer'): 'CSCRank',
            # ('PointTransformerSeg', 'CSC_point_transformer_with_rotation'): 'CSCRankRot',
            # ('OracleRank', 'OracleRank'): 'OracleCatRank[+IoU]',
            # ('OracleRank', 'OracleRankRot'): 'OracleCatRankRot[+IoU]',
            # ('GKRank', 'GKRank'): 'OracleGKRank',
            # ('CatRank', 'CatRank'): 'OracleCatRank',
            # ('RandomRank', 'RandomRank'): 'RandomRank',
        #     ('PointTransformerSeg', 'CSC_point_transformer_with_boxes_large_nms'): 'CSCRankV2'
        # }
        # For cd + distance + angle including scannet training
        # model_name_exp_map = {
        #     ('LearningBased', '3D_DINO_point_transformer_scannet'): 'PointCropRank',
        #     ('LearningBased', 'supervised_point_transformer_scannet'): 'TransformerRank',
        #     ('PointTransformerSeg', 'CSC_point_transformer_scannet'): 'CSCRank',
        #     ('OracleRank', 'OracleRank'): 'OracleCatRank[+IoU]',
        #     ('GKRank', 'GKRank'): 'OracleGKRank',
        #     ('CatRank', 'CatRank'): 'OracleCatRank',
        #     ('RandomRank', 'RandomRank'): 'RandomRank',
        # }
        # For cd + distance + angle object retrieval
        # model_name_exp_map = {
        #     ('LearningBased', '3D_DINO_point_transformer'): 'PointCropRank',
        #     ('LearningBased', 'supervised_point_transformer'): 'TransformerRank',
        #     ('PointTransformerSeg', 'CSC_point_transformer'): 'CSCRank',
        #     ('CatRank', 'CatRank'): 'OracleCatRank',
        #     ('RandomRank', 'RandomRank'): 'RandomRank',
        # }
        # For cd + IoU full classes
        model_name_exp_map = {
            ('LearningBased', '3D_DINO_point_transformer_full'): 'PointCropRank',
            ('LearningBased', '3D_DINO_point_transformer_50_full'): 'PointCropRank50',
            # ('LearningBased', '3D_DINO_point_transformer_with_boxes_large_nms_full'): 'PointCropRankV2',
            ('LearningBased', 'supervised_point_transformer_full'): 'TransformerRank',
            # ('LearningBased', 'supervised_point_transformer_with_boxes_large_nms_full'): 'TransformerRankV2',
            ('PointTransformerSeg', 'CSC_point_transformer_full'): 'CSCRank',
            ('OracleRank', 'OracleRankFull'): 'OracleCatRank[+IoU]',
            # ('CatRank', 'CatRank_with_boxes_full'): 'CatRankV2[+IoU]',
            ('GKRank', 'GKRankFull'): 'OracleGKRank',
            ('CatRank', 'CatRankFull'): 'OracleCatRank',
            ('RandomRank', 'RandomRankFull'): 'RandomRank',
            # ('PointTransformerSeg', 'CSC_point_transformer_with_boxes_large_nms_full'): 'CSCRankV2'
        }
        # For cat + cd + IoU
        # model_name_exp_map = {
        #     ('LearningBased', '3D_DINO_point_transformer'): 'PointCropRank',
        #     ('LearningBased', 'supervised_point_transformer'): 'TransformerRank',
        #     ('OracleRank', 'OracleRank'): 'OracleCatRank[+IoU]',
        #     ('CatRank', 'CatRank'): 'OracleCatRank',
        #     ('PointTransformerSeg', 'CSC_point_transformer'): 'CSCRank',
        #     ('GKRank', 'GKRank'): 'OracleGKRank',
        #     ('LearningBased', 'supervised_point_transformer_with_boxes_large_nms'): 'TransformerRankV2',
        #     ('LearningBased', '3D_DINO_point_transformer_with_boxes_large_nms'): 'PointCropRankV2',
        #     ('PointTransformerSeg', 'CSC_point_transformer_with_boxes_large_nms'): 'CSCRankV2',
        #     ('RandomRank', 'RandomRank'): 'RandomRank'
        # }
        # model_name_exp_map = {
        #     ('LearningBased', '3D_DINO_point_transformer_10'): 'PointCropRank[+9]',
        #     ('LearningBased', 'supervised_point_transformer'): 'TransformerRank',
        #     ('PointTransformerSeg', 'CSC_point_transformer_10'): 'CSCRank[+9]',
        #     ('OracleRank', 'OracleRank'): 'OracleCatRank[+IoU]',
        #     ('GKRank', 'GKRank'): 'OracleGKRank',
        #     ('CatRank', 'CatRank'): 'OracleCatRank',
        #     ('RandomRank', 'RandomRank'): 'RandomRank'
        # }
        # Comparing TransformerRank with and without CAT
        # model_name_exp_map = {
        #     ('LearningBased', '3D_DINO_point_transformer_full'): 'PointCropRank',
        #     ('LearningBased', 'supervised_point_transformer_full'): 'TransformerRank',
        #     ('PointTransformerSeg', 'CSC_point_transformer'): 'CSCRank',
        #     ('LearningBased', 'supervised_point_transformer_with_cats_full'): 'TransformerRankCat',
        #     ('OracleRank', 'OracleRankFull'): 'OracleCatRank[+IoU]',
        # }

    # read the aggregated results and choose the metric to plot
    summary_results = pd.read_csv(aggregated_csv_path)
    metrics = ['distance_cd_mAP_bi_5', 'distance_cd_mAP_bi_10', 'distance_cd_mAP_bi_20', 'distance_cd_mAP_bi_40']
    metrics = ['angle_cd_mAP_bi_5', 'angle_cd_mAP_bi_10', 'angle_cd_mAP_bi_20', 'angle_cd_mAP_bi_40']
    metrics = ['distance_angle_cd_mAP_bi_5', 'distance_angle_cd_mAP_bi_10', 'distance_angle_cd_mAP_bi_20', 'distance_angle_cd_mAP_bi_40']
    metrics = ['distance_angle_cat_cd_mAP_bi_5', 'distance_angle_cat_cd_mAP_bi_10', 'distance_angle_cat_cd_mAP_bi_20', 'distance_angle_cat_cd_mAP_bi_40']

    # plot the quantiative results
    plot_results(metrics, model_name_exp_map, summary_results, args.mode, '1st_epoch_same_scale',
                 same_scale=True, title=None)


if __name__ == '__main__':
    t = time()
    main()
    duration = time() - t
    print('Preparing quantitative results took {} minutes'.format(round(duration / 60, 2)))
