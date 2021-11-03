import os
import json
import numpy as np
from matplotlib import pyplot as plt

from scripts.helper import load_from_json, vanilla_plot

# define the model name and paths
results_dir = '../../../results/matterport3d/LearningBased'
model_names = [
    '3D_DINO_exact_regions_transformer_default',
    '3D_DINO_exact_regions_transformer_normalized',
    '3D_DINO_exact_regions_transformer_fixed_crop_normalized',
    '3D_DINO_exact_regions_transformer_fixed_crop_normalized_aug',
    '3D_DINO_exact_regions_transformer_variable_crops_normalized'
]

log_file_name = 'log.txt'
knn_acc_file_name = 'knn_accuracies.json'
metrics_file_name = 'metrics.json'

# set the number of knn neighbours.
nb_knn = '20'
with_loss = True
with_knn = True
with_metrics = True

tr_loss_list = []
epoch_accuracies_list = []
num_unique_classes_list = []
num_data_skipped_list = []
for model_name in model_names:
    print('Model Name: {}'.format(model_name))
    cp_dir = os.path.join(results_dir, model_name)
    if with_loss:
        # load the training loss
        tr_loss = []
        with open(os.path.join(cp_dir, log_file_name)) as f:
            for epoch_log in f.readlines():
                epoch_log = json.loads(epoch_log)
                tr_loss.append(epoch_log['train_loss'])
        tr_loss_list.append(tr_loss)

    if with_knn:
        # load the knn accuracies.
        epoch_accuracies = {'TopK Avg': [], 'Macro Avg': [], 'Micro Avg': []}
        knn_accuracies = load_from_json(os.path.join(cp_dir, knn_acc_file_name))
        for i, epoch_accuracy in enumerate(knn_accuracies):
            epoch = str(i)
            for metric, acc in epoch_accuracy[epoch][nb_knn].items():
                epoch_accuracies[metric].append(acc * 100)
        epoch_accuracies_list.append(epoch_accuracies)

        # report the accuracy at multiple scales.
        for metric, accuracies in epoch_accuracies.items():
            print('{}: {}'.format(metric, np.max(accuracies)))

    if with_metrics:
        # load the metrics file
        metrics = load_from_json(os.path.join(cp_dir, metrics_file_name))
        num_unique_classes = []
        num_data_skipped = []
        for metric in metrics:
            num_unique_classes.append(metric['num_unique_classes'])
            num_data_skipped.append(metric['num_data_skipped'])
        num_unique_classes_list.append(num_unique_classes)
        num_data_skipped_list.append(num_data_skipped)
    print('*' * 50)

# find the max training loss, min and max accuracy
max_tr_loss = 0
max_acc = 0
min_acc = 0
for i in range(len(model_names)):
    if with_loss:
        max_tr_loss = np.maximum(max_tr_loss, np.max(tr_loss_list[i]))
    if with_knn:
        max_acc = np.maximum(max_acc, np.max([epoch_accuracies_list[i][k] for k in epoch_accuracies.keys()]))
        min_acc = np.minimum(min_acc, np.min([epoch_accuracies_list[i][k] for k in epoch_accuracies.keys()]))

# plot loss and knn accuracies
for i in range(len(model_names)):
    cp_dir = os.path.join(results_dir, model_names[i])
    if with_loss:
        # plt.ylim(0, max_tr_loss)
        # plot train and validation loss
        vanilla_plot(tr_loss_list[i], cp_dir)
        plt.close()

    if with_knn:
        # plot the accuracies all together
        # plt.ylim(0, max_acc)
        epoch_accuracies = epoch_accuracies_list[i]
        for metric, accuracies in epoch_accuracies.items():
            # plot the accuracies
            vanilla_plot(accuracies, cp_dir, plot_label=metric, xlabel='Epochs', ylabel='Accuracies (Percentage)',
                         plot_name='knn_accuracies.png', with_legend=True)
        plt.close()

    if with_metrics:
        # plot the metrics
        vanilla_plot(num_unique_classes_list[i], cp_dir, plot_label='num_unique_classes', xlabel='Epochs',
                     ylabel='count', plot_name='num_unique_classes.png', with_legend=True, scatter=True)
        plt.close()
        vanilla_plot(num_data_skipped_list[i], cp_dir, plot_label='num_data_skipped', xlabel='Epochs',
                     ylabel='count', plot_name='num_data_skipped.png', with_legend=True, scatter=True)
        plt.close()
