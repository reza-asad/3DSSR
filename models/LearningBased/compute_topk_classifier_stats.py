import os
import numpy as np

from scripts.helper import load_from_json

# set the paths
topk = 10
model_name = 'region_classification_transformer'
cat_to_freq_path = '../../data/matterport3d/accepted_cats_to_frequency.json'
per_class_accuracy_path = os.path.join('../../results/matterport3d/LearningBased', model_name, 'per_class_accuracy.json')

# load the cat to frequency data and per class accuracy
cat_to_freq = load_from_json(cat_to_freq_path)
per_class_accuracy = load_from_json(per_class_accuracy_path)

# take the topk categories by frequency
topk_cats = [e[0] for e in sorted(cat_to_freq.items(), reverse=True, key=lambda x: x[1])[:topk]]

# print accuracy for topk, mean accuracy for topk and mean for all cats
topk_accuracy = [per_class_accuracy[cat] for cat in topk_cats]
topk_mean_accuracy = np.mean(topk_accuracy)
mean_accuracy = np.mean(list(per_class_accuracy.values()))

print('topk accuracy: {}'.format(topk_accuracy))
print('topk mean accuracy: {}'.format(topk_mean_accuracy))
print('mean accuracy: {}'.format(mean_accuracy))
