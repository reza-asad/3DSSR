import os
import numpy as np
from matplotlib import pyplot as plt

from scripts.helper import load_from_json


def box_plot(data, title):
    ig, ax = plt.subplots()
    labels = ['', '', '']
    bplot = ax.boxplot([v for v in data.values()],
                       positions=[1, 1.2, 1.4],
                       vert=False,  # vertical box alignment
                       patch_artist=True,  # fill with color
                       labels=labels)

    # ax1.set_title('Rectangular box plot')
    colors = ['lightblue', 'pink', 'lightgreen']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    ax.xaxis.grid(True)
    ax.set_ylabel('Models')
    ax.set_xlabel('Mean Absolute Alignment Error (degrees)')
    boxes = []
    legend_names = []
    for i, (k, v) in enumerate(data.items()):
        boxes.append(bplot["boxes"][i])
        legend_names.append(k)
    leg = ax.legend(boxes, legend_names, loc='upper right')
    ax.set_ylim(0.8, 1.6)
    plt.savefig('../results/matterport3d/evaluation_plots/{}'.format('{}.png'.format(title)))
    plt.show()


def main():
    # define the path to the alignment error files for each model
    mode = 'test'
    exact_rotations = False
    base_path = '../results/matterport3d'
    alignment_errors_path = '../results/matterport3d/evaluations/ablation/alignment_error_align_vs_svd.json'
    model_name_exp_map = {('LearningBased', 'alignment_error_1d'): 'AlignmentModule',
                          ('SVDRank', 'alignment_error_3d'): 'SVD3D',
                          ('SVDRank', 'alignment_error_1d'): 'SVD1D'}
    plot_title = 'alignment_error_non_exact'

    # extract alignment errors for each model
    model_name_exp_to_errors = {model_name_exp: [] for model_name_exp in model_name_exp_map.keys()}
    for model_name_exp, final_name in model_name_exp_map.items():
        if exact_rotations:
            # load the alignment errors for each rotated query.
            query_dict = load_from_json(os.path.join(base_path, model_name_exp[0],
                                                     'query_dict_{}_{}'.format(mode, model_name_exp[1] + '.json'),))
            for query in query_dict.keys():
                model_name_exp_to_errors[model_name_exp].append(query_dict[query]['mean_error'] * 180 /np.pi)
        else:
            # load the alignment errors for non exact rotations of the query subscene.
            alignment_errors = load_from_json(alignment_errors_path)
            model_name_exp_to_errors[model_name_exp] = alignment_errors[final_name]

    # map the model names
    model_to_errros = {model_name_exp_map[k]: v for k, v in model_name_exp_to_errors.items()}

    # plot a box plot containing all mean errors for each model.
    box_plot(model_to_errros, plot_title)


if __name__ == '__main__':
    main()
