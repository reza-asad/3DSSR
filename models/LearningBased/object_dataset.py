import os
import torch
from torch.utils.data import Dataset
import numpy as np

from find_closest_shape_cluster import add_context_objects


class Object(Dataset):
    def __init__(self, scene_graph_dir, objects, closest_shape_clusters, latent_caps_dir, mode, cat_to_filenames=None,
                 hard_negative_prob=0.5, topk_neg=3, topk_context=7):
        self.scene_graph_dir = scene_graph_dir
        self.objects = objects
        self.cat_to_filenames = cat_to_filenames
        self.closest_shape_clusters = closest_shape_clusters
        self.latent_caps_dir = latent_caps_dir
        self.mode = mode
        self.hard_negative_prob = hard_negative_prob
        self.topk_neg = topk_neg
        self.topk_context = topk_context

    def __len__(self):
        return len(self.objects)

    def prepare_latent_caps(self, file_names):
        nb_latent_caps = []
        for file_name in file_names:
            latent_caps = np.load(os.path.join(self.latent_caps_dir, file_name))
            latent_caps = latent_caps.reshape((1, -1))
            latent_caps = torch.from_numpy(latent_caps)
            nb_latent_caps.append(latent_caps)
        nb_latent_caps = torch.cat(nb_latent_caps, dim=0)

        return nb_latent_caps

    def __getitem__(self, idx):
        # read the category of the selected object
        main_obj, main_obj_cat = self.objects[idx]

        # find the context objects for the main object.
        main_obj_and_contexts = add_context_objects(self.scene_graph_dir, main_obj, self.mode, self.topk_context)

        data = {'latent_caps_main': self.prepare_latent_caps(main_obj_and_contexts), 'file_name': main_obj}
        # for training sample positives and negatives
        if self.cat_to_filenames is not None:
            # sample a positive for the selected object
            positive = main_obj
            while positive == main_obj:
                positive = np.random.choice(self.cat_to_filenames[main_obj_cat], 1)[0]

            # sample a negative. at least half of the time, select an object that is in top k closest clusters to main
            # obj.
            negative_cats = self.closest_shape_clusters[main_obj_cat]
            if np.random.uniform(0, 1) > self.hard_negative_prob:
                neg_idx = np.random.choice(range(0, self.topk_neg), 1)[0]
            else:
                neg_idx = np.random.choice(range(len(negative_cats)), 1)[0]
            neg_cat = negative_cats[neg_idx][0]
            negative = np.random.choice(self.cat_to_filenames[neg_cat], 1)[0]

            # find the context objects for positive and negative.
            pos_and_contexts = add_context_objects(self.scene_graph_dir, positive, self.mode, self.topk_context)
            neg_and_contexts = add_context_objects(self.scene_graph_dir, negative, self.mode, self.topk_context)

            # read the latent caps for positive and negative.
            data['latent_caps_positive'] = self.prepare_latent_caps(neg_and_contexts)
            data['latent_caps_negative'] = self.prepare_latent_caps(pos_and_contexts)

        return data







