import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from scripts.helper import load_from_json, visualize_scene


class Object(Dataset):
    def __init__(self, scene_graph_dir, imgs_dir, scene_to_pos_negatives, models_dir, latent_caps_dir, mode,
                 query_scene_name=None, query_nodes=None, min_neg_examples=5, max_neg_examples=10,
                 fov=np.pi / 6):
        self.scene_graph_dir = scene_graph_dir
        self.imgs_dir = imgs_dir
        self.scene_to_pos_negatives = load_from_json(scene_to_pos_negatives)
        self.models_dir = models_dir
        self.latent_caps_dir = latent_caps_dir
        self.mode = mode
        self.file_names = os.listdir(os.path.join(self.scene_graph_dir, self.mode))
        self.query_scene_name = query_scene_name
        self.query_nodes = query_nodes

        self.min_neg_examples = min_neg_examples
        self.max_neg_examples = max_neg_examples
        self.fov = fov

    def __len__(self):
        return len(self.file_names)

    @staticmethod
    def normalize_imgs(imgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        for i in range(3):
            imgs[:, i, ...] = (imgs[:, i, ...] - mean[i]) / std[i]

        return imgs

    @staticmethod
    def img_to_tensor(img):
        img = np.array(img, dtype=np.float)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img.unsqueeze_(dim=0)

        return img

    def read_latent_caps(self, file_name, obj):
        latent_caps_file_name = file_name.split('.')[0] + '-' + obj + '.npy'
        latent_caps = np.load(os.path.join(self.latent_caps_dir, latent_caps_file_name))
        latent_caps = torch.from_numpy(latent_caps.reshape(-1))
        latent_caps.unsqueeze_(dim=0)

        return latent_caps

    def resolution_to_distance(self, camera_height):
        theta = self.fov / 2.0

        return np.tan(theta) * camera_height

    @staticmethod
    def find_context_objects_3d(graph, center_obj, max_distance, replace_ratio=0.25, train=True):
        context_objects = []
        non_context_objects = []
        # take all objects that are at most at max_distance from the center object
        for metrics in graph[center_obj]['ring_info'].values():
            for obj, distance in metrics['distance']:
                if distance <= max_distance:
                    context_objects.append((obj, distance))
                else:
                    non_context_objects.append((obj, distance))

        if len(context_objects) == 0:
            return []

        # if there are at least two context object, randomly select a small subset of them.
        if len(context_objects) > 1 and train:
            num_replace = int(np.ceil(len(context_objects) * replace_ratio))
            # replace some of the context objects with object not in the context.
            # TODO: also try randomly replacing any object, not only the far ones.
            replace_indices = np.random.choice(range(len(context_objects)//2, len(context_objects)), num_replace,
                                               replace=False)
            if len(non_context_objects) > 0:
                non_context_objects = sorted(non_context_objects, key=lambda x: x[1])
                i = 0
                while i < len(non_context_objects) and i < len(replace_indices):
                    idx = replace_indices[i]
                    context_objects[idx] = non_context_objects[i]
                    i += 1

        # sort the context object from furthest to closest.
        context_objects = sorted(context_objects, key=lambda x: x[1], reverse=True)
        context_objects = list(zip(*context_objects))[0]
        # non_context_objects = list(zip(*non_context_objects))[0]

        return context_objects

    def __getitem__(self, idx, test=False, test_obj='32'):
        if test:
            self.file_names[idx] = 'Z6MFQCViBuw_room0.json'

        # load the clean graph for selecting positive/negative examples and latent codes for 3D context.
        scene_name = self.file_names[idx].split('.')[0]
        graph_path = os.path.join(self.scene_graph_dir, self.mode, self.file_names[idx])
        graph = load_from_json(graph_path)
        objects = list(graph.keys())

        data = {'file_name': self.file_names[idx], 'skip': False}
        # if query is not determined, select a random object from the target graph.
        if self.query_scene_name is None:
            # sample a positive object randomly
            if test:
                pos_obj = test_obj
            else:
                pos_obj = np.random.choice(objects, 1)[0]
                # if self.file_names[idx] == 'Z6MFQCViBuw_room0.json':
                #     print(pos_obj)

            # read negative objects associated with the positive. if no negatives found, skip.
            all_neg_objects = self.scene_to_pos_negatives[scene_name]['pos_to_negatives'][pos_obj]
            if len(all_neg_objects) == 0:
                data['skip'] = True
                return data
            # take a subset of the negative examples.
            num_neg_objects = np.random.randint(self.min_neg_examples, self.max_neg_examples+1)
            num_neg_objects = np.minimum(num_neg_objects, len(all_neg_objects))
            neg_objects = all_neg_objects[:num_neg_objects]

            # find an angle to randomly rotate the images.
            random_theta = np.random.choice([0, 90, 180, 270], 1)[0]

            # include the positive object with some probability
            pos_obj_idx = -1
            if np.random.uniform(0, 1) > 0.3:
                pos_negatives = [pos_obj] + neg_objects
            else:
                pos_negatives = neg_objects
            np.random.shuffle(pos_negatives)

            # read the object and context images for the negatives from lowest to highest IoU.
            scene_imgs_dir = os.path.join(self.imgs_dir, scene_name)
            rendered_obj_imgs, rendered_context_imgs = [], []
            for i, obj in enumerate(pos_negatives):
                # read the object and context images
                sub_folder = 'negatives'
                if obj == pos_obj:
                    sub_folder = 'positives'
                    pos_obj_idx = i
                obj_img_path = os.path.join(scene_imgs_dir, sub_folder, 'object_images', obj + '.png')
                context_img_path = os.path.join(scene_imgs_dir, sub_folder, 'context_images', obj + '.png')
                obj_img = Image.open(obj_img_path)
                context_img = Image.open(context_img_path)

                # if obj == pos_obj:
                #     obj_img.show()
                #     context_img.show()
                #     t=y

                # randomly rotate the obj and context images.
                obj_img = obj_img.rotate(random_theta)
                context_img = context_img.rotate(random_theta)

                # convert the images into tensor
                rendered_obj_imgs.append(self.img_to_tensor(obj_img))
                rendered_context_imgs.append(self.img_to_tensor(context_img))

            # combine the rendered images
            rendered_obj_imgs = torch.cat(rendered_obj_imgs, dim=0)
            rendered_context_imgs = torch.cat(rendered_context_imgs, dim=0)

            # normalize the imgs
            rendered_obj_imgs = self.normalize_imgs(rendered_obj_imgs)
            rendered_context_imgs = self.normalize_imgs(rendered_context_imgs)

            # add labels for the rendered images. the last image is assumed to be positive.
            data['labels'] = torch.zeros(len(pos_negatives), 1)
            if pos_obj_idx > 0:
                data['labels'][pos_obj_idx, 0] = 1

            # find the context objects in the 3D space that fit in the object center img.
            camera_pose = np.asarray(self.scene_to_pos_negatives[scene_name]['camera_pose'][pos_obj])
            max_distnace = self.resolution_to_distance(camera_pose[2, 3])
            context_objects_3d = self.find_context_objects_3d(graph, pos_obj, max_distance=max_distnace)

            # read the latent code for the context objects. assume they are ordered from furthest to closest.
            latent_caps = []
            for obj in context_objects_3d:
                latent_caps.append(self.read_latent_caps(self.file_names[idx], obj))
            # add the latent caps for the object itself
            latent_caps.append(self.read_latent_caps(self.file_names[idx], pos_obj))
            data['latent_caps'] = torch.cat(latent_caps, dim=0)
            # print(len(context_objects_3d), data['latent_caps'].shape)

            # visualize the context objects in 3D
            # path = os.path.join(self.scene_graph_dir, self.mode)
            # visualize_scene(scene_graph_dir=path, models_dir=self.models_dir, scene_name=data['file_name'],
            #                 objects=context_objects_3d, with_backbone=True, as_obbox=False)
            # t=y
        else:
            # read the positive and negative images for the query and context objects in the query scene.
            query_scene_imgs_dir = os.path.join(self.imgs_dir, self.query_scene_name.split('.')[0])
            rendered_obj_imgs, rendered_context_imgs = [], []
            for obj in self.query_nodes:
                obj_img_path = os.path.join(query_scene_imgs_dir, 'positives', 'object_images', obj + '.png')
                context_img_path = os.path.join(query_scene_imgs_dir, 'positives', 'context_images', obj + '.png')
                obj_img = Image.open(obj_img_path)
                context_img = Image.open(context_img_path)

                # convert the images into tensor
                rendered_obj_imgs.append(self.img_to_tensor(obj_img))
                rendered_context_imgs.append(self.img_to_tensor(context_img))

            # combine the rendered images
            rendered_obj_imgs = torch.cat(rendered_obj_imgs, dim=0)
            rendered_context_imgs = torch.cat(rendered_context_imgs, dim=0)

            # normalize the imgs
            rendered_obj_imgs = self.normalize_imgs(rendered_obj_imgs)
            rendered_context_imgs = self.normalize_imgs(rendered_context_imgs)

            # load the latent caps for each object and its context in the target scene.
            all_latent_caps = []
            for center_obj in objects:
                # read the camera pose for each object in the target scene and find its 3D context objects
                camera_pose = np.asarray(self.scene_to_pos_negatives[scene_name]['camera_pose'][center_obj])
                max_distnace = self.resolution_to_distance(camera_pose[2, 3])
                context_objects_3d = self.find_context_objects_3d(graph, center_obj, max_distance=max_distnace,
                                                                  train=False)

                # save the latent caps for the center object and its 3D context objects.
                latent_caps_center_obj = []
                for context_object_3d in context_objects_3d:
                    latent_caps_center_obj.append(self.read_latent_caps(self.file_names[idx], context_object_3d))
                latent_caps_center_obj.append(self.read_latent_caps(self.file_names[idx], center_obj))
                latent_caps_center_obj = torch.cat(latent_caps_center_obj, dim=0)

                # record the results and try another object in the target scene.
                all_latent_caps.append(latent_caps_center_obj)

            data['latent_caps'] = all_latent_caps
            data['candidates'] = objects

        # package the object and context imgs into tensors.
        data['obj_imgs'] = rendered_obj_imgs
        data['context_imgs'] = rendered_context_imgs

        return data







