import os
from collections import Counter
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import trimesh

from scripts.helper import load_from_json
from scripts.box import Box


class SubScene(Dataset):
    def __init__(self, scene_dir, pc_dir, metadata_path, accepted_cats_path, max_coord_scene, file_name_to_idx,
                 mode='train', num_points=4096, num_objects=5, with_transform=False, random_subscene=False,
                 batch_one=True):
        self.scene_dir = os.path.join(scene_dir, mode)
        self.pc_dir = os.path.join(pc_dir, mode)
        self.max_coord_scene = max_coord_scene
        self.file_name_to_idx = file_name_to_idx
        self.mode = mode
        self.num_points = num_points
        self.num_objects = num_objects
        self.with_transform = with_transform
        self.random_subscene = random_subscene
        self.batch_one = batch_one

        # scenes to process.
        self.accepted_cats = sorted(load_from_json(accepted_cats_path))
        self.cat_to_idx = {cat: i for i, cat in enumerate(self.accepted_cats)}
        self.file_names = self.find_file_names(metadata_path, mode)

        # parameters for radial perturbation.
        self.min_radial_threshold = 0.3
        self.max_radial_threshold = 0.5

    def find_file_names(self, metadata_path, mode):
        # load the metadata and accepted cats.
        df = pd.read_csv(metadata_path)

        # filter the metadata by the accepted cats
        is_accepted = df['mpcat40'].apply(lambda x: x in self.accepted_cats)
        df = df.loc[is_accepted]

        # filter the metadata by the mode
        df = df.loc[df['split'] == mode]
        print(Counter(df['mpcat40']))

        # filter the remaining scenes, so they have at least 1 object.
        scene_name_to_obj_count = df.groupby(['room_name'])['objectId'].count()
        if self.batch_one:
            scene_filter_criteria = scene_name_to_obj_count > 0
        else:
            # ensure consistent batch during training.
            scene_filter_criteria = scene_name_to_obj_count > (self.num_objects - 1)
        scene_name_to_obj_count = scene_name_to_obj_count[scene_filter_criteria]

        # take objects in the filtered scenes.
        scene_names = set(scene_name_to_obj_count.keys())
        df = df.loc[df['room_name'].apply(lambda x: x in scene_names)]
        df['key'] = df[['room_name', 'objectId']].apply(lambda x: '-'.join([str(x[0]), str(x[1])]) + '.npy', axis=1)

        return df['key'].values.tolist()

    def __len__(self):
        return len(self.file_names)

    def find_random_subscene(self, scene, anchor_obj):
        # shuffle the context objects.
        context_objects = [obj for obj in scene.keys() if obj != anchor_obj]
        if self.mode in ['val', 'test']:
            np.random.seed(0)
        np.random.shuffle(context_objects)

        # sample.
        subscene_objects = [anchor_obj] + context_objects[: self.num_objects - 1]

        # populate the properties of the subscene and return it.
        subscene = {}
        for obj in subscene_objects:
            subscene[obj] = scene[obj].copy()

        return subscene

    def find_radial_subscene(self, scene, anchor_obj):
        # find the closest N objects to the anchor object to build the subscene.
        anchor_centroid = np.asarray(scene[anchor_obj]['aabb'][0], dtype=np.float32)
        distances = []
        for obj in scene.keys():
            obj_centroid = np.asarray(scene[obj]['aabb'][0], dtype=np.float32)
            distance = np.linalg.norm(obj_centroid - anchor_centroid)
            distances.append((obj, distance))

        # sort the distances from smallest to largest and take the subscene objects
        distances = sorted(distances, key=lambda x: x[1])
        subscene_objects = [e[0] for e in distances[:self.num_objects]]

        # populate the properties of the subscene and return it.
        subscene = {}
        for obj in subscene_objects:
            subscene[obj] = scene[obj].copy()

        return subscene

    def populate_scene_info(self, scene, scene_name):
        # for each object in the scene load the object's dense point cloud along with translation and rotation
        # parameters for the object.
        scene_to_pc = {}
        for obj, obj_info in scene.items():
            # load the dense point cloud for the obj.
            pc_filename = '-'.join([scene_name, obj]) + '.npy'
            obj_pc = np.load(os.path.join(self.pc_dir, pc_filename))

            # find the axis-aligned bounding box for the object.
            aabb = Box(np.asarray(obj_info['aabb'], dtype=np.float32))

            # record normalized results.
            scene_to_pc[obj] = {'pc': obj_pc, 'aabb': aabb}

        return scene_to_pc

    @staticmethod
    def build_colored_pc(pc):
        radii = np.linalg.norm(pc, axis=1)
        colors = trimesh.visual.interpolate(radii, color_map='viridis')
        pc_vis = trimesh.points.PointCloud(pc, colors=colors)

        return pc_vis

    @staticmethod
    def visualize_scene_boxes(anchor, curr_scene, prev_scene):
        color = "#0000FF"
        if len(prev_scene) != 0:
            color = "#ff0000"
        for obj, obj_info in curr_scene.items():
            box = trimesh.creation.box(obj_info['aabb'].scale, obj_info['aabb'].transformation)
            if obj == anchor:
                box.visual.vertex_colors = trimesh.visual.color.hex_to_rgba("#00FF00")
            else:
                box.visual.vertex_colors = trimesh.visual.color.hex_to_rgba(color)
            prev_scene.append(box)

        return prev_scene

    def sample_points(self, scene_pc):
        for obj, obj_info in scene_pc.items():
            # for validation use the same sample.
            if self.mode in ['val', 'test']:
                np.random.seed(0)
            sampled_indices = np.random.choice(range(len(obj_info['pc'])), self.num_points, replace=False)
            obj_info['pc'] = obj_info['pc'][sampled_indices, :]

        return scene_pc

    @staticmethod
    def new_box(old_box, translation, rotation):
        transformation = np.eye(4)
        transformation[:3, 3] = translation
        transformation[:3, :3] = rotation
        new_box = old_box.apply_transformation(transformation)
        # boxes = trimesh.Scene([trimesh.creation.box(old_box.scale, old_box.transformation),
        #                        trimesh.creation.box(new_box.scale, new_box.transformation)])

        return new_box

    @staticmethod
    def transform_pc(pc, translation, rotation):
        transformation = np.eye(4)
        transformation[:3, 3] = translation
        transformation[:3, :3] = rotation

        new_pc = np.ones((len(pc), 4), dtype=np.float)
        new_pc[:, 0] = pc[:, 0]
        new_pc[:, 1] = pc[:, 1]
        new_pc[:, 2] = pc[:, 2]

        new_pc = np.dot(transformation, new_pc.T).T
        new_pc = new_pc[:, :3]

        return new_pc

    def perturb_radial(self, aabb_obj, aabb_anchor):
        # find the centroids of the boxes.
        obj_c = aabb_obj.translation
        anchor_c = aabb_anchor.translation

        # find the vector connecting anchor to obj.
        anchor_to_obj = obj_c - anchor_c

        # compute the norm of the distance between obj and anchor
        anchor_to_obj_norm = np.linalg.norm(anchor_to_obj)

        # pick a positive or negative direction along the vec.
        alpha = 1
        if np.random.uniform(0, 1) > 0.5:
            alpha = -1

        # randomly pick the amount of perturbation.
        perturbation_magnitude = np.random.uniform(self.min_radial_threshold * anchor_to_obj_norm,
                                                   self.max_radial_threshold * anchor_to_obj_norm)

        # move the centroid of the context object according to the perturbation magnitude and direction.
        translation = alpha * perturbation_magnitude * (anchor_to_obj / anchor_to_obj_norm)

        # build the new aabb
        return self.new_box(aabb_obj, translation, np.eye(3))

    def perturb_angular(self, subscene_pc, anchor, context_obj, theta):
        # create the rotation matrix.
        rotation = np.asarray([[np.cos(theta), -np.sin(theta), 0],
                               [np.sin(theta), np.cos(theta), 0],
                               [0, 0, 1]])

        # translate the object's aabb so the centroid of the anchor object is at the origin.
        anchor_translation = -subscene_pc[anchor]['aabb'].translation
        subscene_pc[context_obj]['aabb'] = self.new_box(subscene_pc[context_obj]['aabb'], anchor_translation, np.eye(3))

        # rotate the object's aabb by theta.
        subscene_pc[context_obj]['aabb'] = self.new_box(subscene_pc[context_obj]['aabb'], np.zeros(3), rotation)

        # translate the object's aabb back to the world coordinate.
        subscene_pc[context_obj]['aabb'] = self.new_box(subscene_pc[context_obj]['aabb'], -anchor_translation, np.eye(3))

        # vis only
        # pc1 = self.build_colored_pc(subscene_pc[context_obj]['pc'])
        # pc1.show()

        # rotate the pc
        subscene_pc[context_obj]['pc'] = self.transform_pc(subscene_pc[context_obj]['pc'], np.zeros(3), rotation)

        # pc2 = self.build_colored_pc(subscene_pc[context_obj]['pc'])
        # pc2.show()

        return subscene_pc[context_obj]

    def perturb(self, subscene_pc, anchor):
        # visualize the query boxes.
        # prev_scene = self.visualize_scene_boxes(anchor, subscene_pc, [])

        # choose a random rotation for the subscene.
        theta = np.random.uniform(0, 2*np.pi)

        # for each object in the query, perturb its radial and angle relative to the anchor object
        context_objects = [obj for obj in subscene_pc.keys() if obj != anchor]
        for context_object in context_objects:
            # perturb radially relative to the anchor obj.
            subscene_pc[context_object]['aabb'] = self.perturb_radial(subscene_pc[context_object]['aabb'],
                                                                      subscene_pc[anchor]['aabb'])

            # perturb angular relative to the anchor.
            subscene_pc[context_object] = self.perturb_angular(subscene_pc, anchor, context_object, theta)

        # angular perturbation of the anchor object.
        subscene_pc[anchor] = self.perturb_angular(subscene_pc, anchor, anchor, theta)

        # visualize the query boxes after perturbation compared to no perturbation.
        # combined_scene = self.visualize_scene_boxes(anchor, subscene_pc, prev_scene)
        # trimesh.Scene(combined_scene).show()

        return subscene_pc

    def bring_to_local(self, subscene_pc, anchor):
        # translate the object's aabb so the centroid of the anchor object is at the origin.
        anchor_translation = -subscene_pc[anchor]['aabb'].translation
        for obj in subscene_pc.keys():
            subscene_pc[obj]['aabb'] = self.new_box(subscene_pc[obj]['aabb'], anchor_translation, np.eye(3))

        # translate each object (point cloud) according to the aabbs.
        for obj in subscene_pc.keys():
            translation = subscene_pc[obj]['aabb'].translation
            subscene_pc[obj]['pc'] = self.transform_pc(subscene_pc[obj]['pc'], translation, np.eye(3))

        return subscene_pc

    def normalize_scene(self, scene):
        for obj, obj_info in scene.items():
            # normalize the pc.
            obj_info['pc'] /= self.max_coord_scene

        return scene

    def prepare_tensor_data(self, scene, scene_pc, anchor_obj):
        pc_data = np.zeros((len(scene_pc), self.num_points, 3), dtype=np.float32)
        label_data = np.zeros((len(scene_pc)))

        objects = sorted(scene_pc.keys(), key=int)
        for i, obj in enumerate(objects):
            # add pc data.
            pc_data[i, ...] = scene_pc[obj]['pc']

            # add label
            label_data[i] = self.cat_to_idx[scene[obj]['category'][0]]

        # find the index of the anchor object.
        anchor_idx = np.zeros(1, dtype=np.long)
        for idx, obj in enumerate(objects):
            if obj == anchor_obj:
                anchor_idx[0] = idx
                break

        # convert data to tensor.
        pc_data = torch.from_numpy(pc_data)
        label_data = torch.from_numpy(label_data)
        anchor_idx = torch.from_numpy(anchor_idx)

        return pc_data, label_data, anchor_idx

    def __getitem__(self, idx):
        # load the anchor object and the scene it belongs to.
        file_name = self.file_names[idx]
        scene_name = file_name.split('-')[0]
        scene = load_from_json(os.path.join(self.scene_dir, scene_name + '.json'))

        # take a subscene.
        anchor_obj = file_name.split('.')[0].split('-')[-1]
        if self.random_subscene:
            subscene = self.find_random_subscene(scene, anchor_obj)
        else:
            subscene = self.find_radial_subscene(scene, anchor_obj)

        # populate dense pc, translation and rotation for each object in the scene.
        subscene_pc = self.populate_scene_info(subscene, scene_name)

        # sample points from each object's point cloud.
        subscene_pc = self.sample_points(subscene_pc)

        # apply transformations to the subscene if necessary.
        if self.with_transform:
            subscene_pc = self.perturb(subscene_pc, anchor_obj)

        # build the subscene where anchor object is at the origin of the scene's coordinate frame.
        subscene_pc = self.bring_to_local(subscene_pc, anchor_obj)

        # normalize the pc and centroid
        subscene_pc = self.normalize_scene(subscene_pc)

        # prepare the data for training.
        pc, label, anchor_idx = self.prepare_tensor_data(scene, subscene_pc, anchor_obj)

        # prepare file names for the anchor object.
        file_name_idx = torch.zeros(1)
        file_name_idx[0] = self.file_name_to_idx[file_name]

        data = {'pc': pc, 'label': label, 'file_name': file_name_idx, 'anchor_idx': anchor_idx}

        return data
