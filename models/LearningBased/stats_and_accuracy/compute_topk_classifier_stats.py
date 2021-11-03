import os
import numpy as np
import pandas as pd
from collections import Counter

from scripts.helper import load_from_json, vanilla_plot

# set the paths for reading the results of the model.
topk = 10
mode = 'val'
with_loss = False
dataset = 'matterport3d'
model_name = 'region_classification_transformer_random_init'
cp_dir = os.path.join('../../results/{}/LearningBased'.format(dataset), model_name)
per_class_accuracy_path = os.path.join(cp_dir, 'per_class_accuracy_fixed_region.json')

# load the cat to frequency data and per class accuracy
cat_to_freq_path = '../../data/matterport3d/accepted_cats_to_frequency.json'.format(dataset)
df_metadata = pd.read_csv('../../data/matterport3d/metadata.csv'.format(dataset))
cat_to_freq = load_from_json(cat_to_freq_path)
per_class_accuracy = load_from_json(per_class_accuracy_path)

# take the topk categories by frequency
topk_cats = [e[0] for e in sorted(cat_to_freq.items(), reverse=True, key=lambda x: x[1])[:topk]]

# compute macro average accuracy for topk.
topk_accuracy = [per_class_accuracy[cat] for cat in topk_cats]
topk_mean_accuracy = np.mean(topk_accuracy)

# filter the metadata to only include test objects and accpeted cats
df_metadata = df_metadata.loc[df_metadata['split'] == mode]
df_metadata = df_metadata.loc[df_metadata['mpcat40'].apply(lambda x: x in cat_to_freq)]
# find the number of objects per accepted category.
cat_to_freq_test = Counter(df_metadata['mpcat40'])

# add any missing category to the per_class_accuracy dictionary.
for cat in cat_to_freq_test.keys():
    if cat not in per_class_accuracy:
        per_class_accuracy[cat] = 0

# compute macro average for all accepted classes
mean_accuracy = np.mean(list(per_class_accuracy.values()))

num_total_test = df_metadata.shape[0]
num_correct = 0
for cat, freq in cat_to_freq_test.items():
    num_correct += per_class_accuracy[cat] * freq
micro_accuracy = num_correct / num_total_test

# map the topk accuracy to what percentage of the data they are responsible for
num_total_train = np.sum([num for num in cat_to_freq.values()])
topk_acc_freq = [(cat, per_class_accuracy[cat], cat_to_freq[cat]/num_total_train) for cat in topk_cats]

print('topk accuracy: {}'.format(topk_acc_freq))
print('topk mean accuracy: {}'.format(topk_mean_accuracy))
print('Macro average accuracy: {}'.format(mean_accuracy))
print('Micro average accuracy: {}'.format(micro_accuracy))

# plot train and validation loss
if with_loss:
    tr_loss = np.load(os.path.join(cp_dir, 'training_loss.npy'))
    vanilla_plot(tr_loss, cp_dir)
    val_loss = np.load(os.path.join(cp_dir, 'valid_loss.npy'))
    vanilla_plot(val_loss, cp_dir)
