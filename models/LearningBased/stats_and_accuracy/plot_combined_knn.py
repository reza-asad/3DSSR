import os
import json

from matplotlib import pyplot as plt

# set the paths
results_dir = '../../../results/matterport3d/LearningBased'
results_folder_names = ['3D_DINO_regions_non_equal_full_top10', '3D_DINO_regions_non_equal_full_top10_pret']
epoch_offset = 0
multiples = 1
nb_knn = '20'

# add legend and label
plt.xlabel('Epoch')
plt.ylabel('Accuracy (Percentage)')
# plt.axis([0, 100, 0, 100])

# read the knn accuracies for each model.
for results_folder_name in results_folder_names:
    # load the knn accuracies
    with open(os.path.join(results_dir, results_folder_name, 'knn_accuracies.json')) as f:
        knn_accuracies_dict = json.load(f)

    # populate the accuracies for a desired neighbour
    knn_accuracies = {}
    for checkpoint, accuracy in knn_accuracies_dict.items():
        checkpoint_number = int(checkpoint.split('.')[0][10:])
        knn_accuracies[checkpoint_number] = accuracy[nb_knn]
    knn_accuracies_sorted = list(zip(*sorted(knn_accuracies.items(), key=lambda x: x[0])))[1]

    # plot knn accuracies
    plt.plot([multiples * i for i in range(epoch_offset, len(knn_accuracies_sorted)+epoch_offset)], knn_accuracies_sorted,
             label='{}'.format(results_folder_name))

# save the plot
plt.legend()
# plt.show()
plt.savefig('knn_accuracies.png')



