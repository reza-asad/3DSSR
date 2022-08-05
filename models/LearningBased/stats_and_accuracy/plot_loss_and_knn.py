import os
import json
import numpy as np

from matplotlib import pyplot as plt


def plot_tr_val_loss(tr_loss, val_loss, cp_dir):
    plt.plot(range(1, len(tr_loss)+1), tr_loss, label='Train')
    if len(val_loss) > 0:
        plt.plot(range(1, len(tr_loss)+1), val_loss, label='Val')
        plt.legend()

    # add label
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.savefig(os.path.join(cp_dir, 'loss.png'))


# set the paths
dataset = 'matterport3d'
log_file_name = 'log.txt'
results_folder_name = '3D_DINO_regions_non_equal_full_top10_pret'
cp_dir = '../../../results/{}/LearningBased/{}'.format(dataset, results_folder_name)
epoch_offset = 0
multiples = 10
with_knn = True
with_loss = True

if with_knn:
    # add knn accuracies
    with open(os.path.join(cp_dir, 'knn_accuracies.json')) as f:
        knn_accuracies_dict = json.load(f)
    knn_accuracies = {}
    nb_knn = '20'
    for checkpoint, accuracy in knn_accuracies_dict.items():
        checkpoint_number = int(checkpoint.split('.')[0][10:])
        knn_accuracies[checkpoint_number] = accuracy[nb_knn]
    knn_accuracies = list(zip(*sorted(knn_accuracies.items(), key=lambda x: x[0])))[1]
    print('Maximum KNN ACC is {} from Epoch {}'.format(np.max(knn_accuracies), (np.argmax(knn_accuracies) + epoch_offset) * multiples))

if with_loss:
    # load the training loss
    tr_loss = []
    with open(os.path.join(cp_dir, log_file_name)) as f:
        for epoch_log in f.readlines():
            epoch_log = json.loads(epoch_log)
            tr_loss.append(epoch_log['train_loss'])

    # plot train loss
    plot_tr_val_loss(tr_loss, [], cp_dir)
    plt.close()

if with_knn:
    # plot knn accuracies
    plt.plot([multiples * i for i in range(epoch_offset, len(knn_accuracies)+epoch_offset)], knn_accuracies, label='KNN Acc')

    # add legend and label
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (Percentage)')

    # save the plot
    plt.savefig(os.path.join(cp_dir, 'knn_accuracies.png'))


